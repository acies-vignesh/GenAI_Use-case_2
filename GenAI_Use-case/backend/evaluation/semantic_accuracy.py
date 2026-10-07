"""Score a generated semantic model against the steward-written reference for the demo DB.

    cd backend
    .venv/Scripts/python -m evaluation.semantic_accuracy                       # latest generated version
    .venv/Scripts/python -m evaluation.semantic_accuracy --model ../data/semantic_models/<file>.json

Entities are matched by table name and attributes by column, so ids/names don't need to agree.
"""
import argparse
import json
import sys
from pathlib import Path

from app.config import PROJECT_ROOT
from app.schemas.semantic_model import SemanticModel

REFERENCE = PROJECT_ROOT / "samples" / "semantic_models" / "demo_retail.reference.json"
MODELS_DIR = PROJECT_ROOT / "data" / "semantic_models"


def _ratio(hit: int, total: int) -> str:
    return f"{hit}/{total} ({hit / total:.0%})" if total else "n/a"


def evaluate(gen: SemanticModel, ref: SemanticModel) -> dict:
    gen_by_table = {e.table: e for e in gen.entities}
    ref_eid_to_table = {e.entity_id: e.table for e in ref.entities}
    gen_eid_to_table = {e.entity_id: e.table for e in gen.entities}
    out: dict = {"mismatches": []}

    def miss(kind: str, what: str, expected, got) -> None:
        out["mismatches"].append({"kind": kind, "what": what, "expected": expected, "got": got})

    counts = {k: [0, 0] for k in ("entity_type", "business_key", "semantic_type", "role", "allowed_values")}
    pii = {"tp": 0, "fn": 0, "fp": 0}
    for re_ in ref.entities:
        ge = gen_by_table.get(re_.table)
        if ge is None:
            miss("entity", re_.table, "present", "missing")
            continue
        counts["entity_type"][1] += 1
        if ge.entity_type == re_.entity_type:
            counts["entity_type"][0] += 1
        else:
            miss("entity_type", re_.table, re_.entity_type.value, ge.entity_type.value)
        counts["business_key"][1] += 1
        if set(ge.business_key) == set(re_.business_key):
            counts["business_key"][0] += 1
        else:
            miss("business_key", re_.table, re_.business_key, ge.business_key)

        gattrs = {a.column: a for a in ge.attributes}
        for ra in re_.attributes:
            ga = gattrs.get(ra.column)
            where = f"{re_.table}.{ra.column}"
            if ga is None:
                miss("attribute", where, "present", "missing")
                continue
            for key in ("semantic_type", "role"):
                counts[key][1] += 1
                if getattr(ga, key) == getattr(ra, key):
                    counts[key][0] += 1
                else:
                    miss(key, where, getattr(ra, key).value, getattr(ga, key).value)
            if ra.pii and ga.pii:
                pii["tp"] += 1
            elif ra.pii:
                pii["fn"] += 1
                miss("pii", where, True, False)
            elif ga.pii:
                pii["fp"] += 1
                miss("pii", where, False, True)
            if ra.allowed_values:
                counts["allowed_values"][1] += 1
                got = set(ga.allowed_values.values) if ga.allowed_values else set()
                if got == set(ra.allowed_values.values):
                    counts["allowed_values"][0] += 1
                else:
                    miss("allowed_values", where, sorted(map(str, ra.allowed_values.values)), sorted(map(str, got)))

    def rel_key(r, eid_to_table):
        return eid_to_table.get(r.from_entity), tuple(r.from_attributes), eid_to_table.get(r.to_entity)

    ref_rels = {rel_key(r, ref_eid_to_table) for r in ref.relationships}
    gen_rels = {rel_key(r, gen_eid_to_table) for r in gen.relationships}
    out["relationships"] = _ratio(len(ref_rels & gen_rels), len(ref_rels))
    for r in ref_rels - gen_rels:
        miss("relationship", f"{r[0]}.{','.join(r[1])} -> {r[2]}", "present", "missing")
    out["extra_relationships"] = len(gen_rels - ref_rels)

    hier_ok = [0, 0]
    gen_hier = {gen_eid_to_table.get(hy.entity): hy for hy in gen.hierarchies}
    for rh in ref.hierarchies:
        table = ref_eid_to_table[rh.entity]
        gh = gen_hier.get(table)
        for check, expected, got in (
            ("levels", rh.levels, gh.levels if gh else None),
            ("attach", sorted((ref_eid_to_table[a.entity], a.attribute, a.must_attach_at) for a in rh.attached_entities),
             sorted((gen_eid_to_table[a.entity], a.attribute, a.must_attach_at) for a in gh.attached_entities)
             if gh else None),
        ):
            hier_ok[1] += 1
            if expected == got:
                hier_ok[0] += 1
            else:
                miss(f"hierarchy_{check}", table, expected, got)

    for k, (hit, total) in counts.items():
        out[k] = _ratio(hit, total)
    out["hierarchies"] = _ratio(*hier_ok)
    tp, fn, fp = pii["tp"], pii["fn"], pii["fp"]
    out["pii_recall"] = _ratio(tp, tp + fn)
    out["pii_precision"] = _ratio(tp, tp + fp)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, help="Generated model JSON (default: latest version)")
    args = parser.parse_args()
    path = args.model or max(MODELS_DIR.glob("semantic_model_*.json"),
                             key=lambda f: int(f.stem.rsplit("_v", 1)[1]))
    gen = SemanticModel.model_validate_json(path.read_text(encoding="utf-8"))
    ref = SemanticModel.model_validate_json(REFERENCE.read_text(encoding="utf-8"))
    result = evaluate(gen, ref)

    print(f"Model: {path.name}")
    for k in ("entity_type", "business_key", "semantic_type", "role", "pii_recall", "pii_precision",
              "allowed_values", "relationships", "hierarchies"):
        print(f"  {k:<15} {result[k]}")
    print(f"  extra relationships: {result['extra_relationships']}")
    print("Mismatches:")
    for m in result["mismatches"]:
        print(f"  [{m['kind']}] {m['what']}: expected {m['expected']}, got {m['got']}")

    out = PROJECT_ROOT / "data" / "evaluation" / f"semantic_accuracy_{path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
