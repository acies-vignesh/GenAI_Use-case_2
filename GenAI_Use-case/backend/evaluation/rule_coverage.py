"""How many planted issues would the generated rules catch? (structural matching against the reference rules)

    cd backend
    .venv/Scripts/python -m evaluation.rule_coverage                       # proposed+approved rules in the app DB
    .venv/Scripts/python -m evaluation.rule_coverage --rules ../data/evaluation/x.json --label no-context

A generated rule "covers" a reference rule when it targets the same entity, checks the same column(s)
with the same kind of check, and its parameters are at least as strict (allowed values a subset,
range bounds as tight, the same hierarchy level...). Approximate by design - the real test is running
the rules against the data, which comes with the rule executor.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from app.config import PROJECT_ROOT
from app.persistence.repositories import unit_of_work
from app.rules.expressions import column_names, parse_expression
from app.schemas.rule import Rule
from app.schemas.semantic_model import SemanticModel

REF_RULES = PROJECT_ROOT / "samples" / "rules" / "demo_retail.reference_rules.json"
REF_MODEL = PROJECT_ROOT / "samples" / "semantic_models" / "demo_retail.reference.json"
MANIFEST = PROJECT_ROOT / "samples" / "source_db" / "planted_issues.json"

FAMILY = {
    "not_null": "completeness", "null_rate_max": "completeness", "allowed_values": "allowed",
    "pattern": "format", "length_range": "format", "range": "range", "unique": "unique",
    "date_not_future": "future", "not_placeholder": "placeholder", "row_condition": "row", "conditional": "row",
    "foreign_key": "fk", "has_children": "children", "aggregate_compare": "aggregate",
    "hierarchy_parent_level": "h_parent", "hierarchy_acyclic": "h_acyclic", "hierarchy_attach_level": "h_attach",
    "custom_sql": "custom",
}


def _cols(text: str | None) -> set[str]:
    if not text:
        return set()
    try:
        return column_names(parse_expression(text))
    except Exception:
        return set()


def row_columns(r: Rule) -> set[str]:
    c = r.check
    cols = {x.lower() for x in r.target.columns} | _cols(r.filter)
    for f in ("expression", "when", "then", "parent_expression"):
        cols |= _cols(getattr(c, f, None))
    return cols


def _stricter_min(g, gi, r, ri) -> bool:
    return g is not None and (g > r or (g == r and (not gi or ri)))


def _stricter_max(g, gi, r, ri) -> bool:
    return g is not None and (g < r or (g == r and (not gi or ri)))


def matches(ref: Rule, gen: Rule, ref_h: dict[str, str], gen_h: dict[str, str]) -> bool:
    if gen.target.entity != ref.target.entity:
        return False
    fr, fg = FAMILY[ref.check.type], FAMILY[gen.check.type]
    rc, gc = ref.check, gen.check
    same_cols = {c.lower() for c in ref.target.columns} == {c.lower() for c in gen.target.columns}

    if fg == "row" and fr in ("row", "range"):        # "total_amount >= 0" written as a row condition
        return row_columns(ref) <= row_columns(gen)
    if fr != fg:
        return False
    if fr == "completeness":
        if not same_cols:
            return False
        limit = rc.max_rate if rc.type == "null_rate_max" else 0.0
        return gc.type == "not_null" or gc.max_rate <= max(limit, 0.0)
    if fr == "allowed":
        return same_cols and gc.case_sensitive and set(map(str, gc.values)) <= set(map(str, rc.values))
    if fr == "range":
        if not same_cols:
            return False
        ok_min = rc.min is None or _stricter_min(gc.min, gc.inclusive, rc.min, rc.inclusive)
        ok_max = rc.max is None or _stricter_max(gc.max, gc.inclusive, rc.max, rc.inclusive)
        return ok_min and ok_max
    if fr == "placeholder":
        return same_cols and bool(set(map(str, gc.values)) & set(map(str, rc.values)))
    if fr == "fk":
        return same_cols and gc.ref_entity == rc.ref_entity
    if fr == "children":
        return gc.child_entity == rc.child_entity
    if fr == "aggregate":
        return (gc.child_entity == rc.child_entity and bool(_cols(gc.child_expression) & _cols(rc.child_expression))
                and bool(_cols(gc.parent_expression) & _cols(rc.parent_expression)))
    if fr in ("h_parent", "h_acyclic"):
        return gen_h.get(gc.hierarchy) == ref_h.get(rc.hierarchy)
    if fr == "h_attach":
        return same_cols and gc.level == rc.level and gen_h.get(gc.hierarchy) == ref_h.get(rc.hierarchy)
    if fr == "custom":
        return False
    return same_cols                                    # unique, format, future


def evaluate(generated: list[Rule], gen_model: SemanticModel) -> dict:
    ref_items = json.loads(REF_RULES.read_text(encoding="utf-8"))["rules"]
    manifest = {i["issue_id"]: i for i in json.loads(MANIFEST.read_text(encoding="utf-8"))["issues"]}
    ref_model = SemanticModel.model_validate_json(REF_MODEL.read_text(encoding="utf-8"))
    ref_h = {h.hierarchy_id: h.entity for h in ref_model.hierarchies}
    gen_h = {h.hierarchy_id: h.entity for h in gen_model.hierarchies}

    used: set[str] = set()
    issues: dict[str, dict] = {pi: {"covered_by": []} for pi in manifest}
    for item in ref_items:
        ref = Rule.model_validate(item["rule"])
        hits = [g for g in generated if matches(ref, g, ref_h, gen_h)]
        used |= {g.rule_id for g in hits}
        for pi in item["covers"]:
            issues[pi]["covered_by"] += [f"[{g.origin.value}] {g.name}" for g in hits]

    rows = []
    for pi, info in issues.items():
        m = manifest[pi]
        origins = {c.split("]")[0][1:] for c in info["covered_by"]}
        rows.append({"issue_id": pi, "dimension": m["dimension"], "table": m["table"],
                     "description": m["description"], "covered": bool(info["covered_by"]),
                     "by": sorted(origins), "covered_by": sorted(set(info["covered_by"]))})
    extras = [f"[{g.origin.value}] {g.target.entity}: {g.name} ({g.check.type})"
              for g in generated if g.rule_id not in used]
    return {"issues": rows, "extras": extras, "total_rules": len(generated)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=Path, help="JSON list of rules (default: proposed+approved in the app DB)")
    parser.add_argument("--source", help="Source id (default: the only source in the app DB)")
    parser.add_argument("--label", default="db", help="Name for this evaluation in the output file")
    args = parser.parse_args()

    with unit_of_work() as repo:
        source_id = args.source or repo.sources.list()[0].id
        gen_model = repo.semantic_models.latest(source_id)
        if args.rules:
            generated = [Rule.model_validate(r) for r in json.loads(args.rules.read_text(encoding="utf-8"))]
        else:
            generated = [r for r in repo.rules.list(source_id) if r.status.value != "rejected"]
    result = evaluate(generated, gen_model)

    rows = result["issues"]
    for r in rows:
        mark = "[x]" if r["covered"] else "[ ]"
        print(f"{mark} {r['issue_id']} {r['dimension']:<22} {r['table']}: {r['description']}")
        if r["covered"]:
            print(f"        {r['covered_by'][0]}" + (f"  (+{len(r['covered_by']) - 1} more)" if len(r['covered_by']) > 1 else ""))
    print("\nBy dimension (covered / total):")
    total, hit = Counter(r["dimension"] for r in rows), Counter(r["dimension"] for r in rows if r["covered"])
    for dim in total:
        print(f"  {dim:<22} {hit[dim]:>2} / {total[dim]}")
    covered = [r for r in rows if r["covered"]]
    only_llm = sum(1 for r in covered if r["by"] == ["llm"])
    print(f"\nCoverage: {len(covered)}/{len(rows)} planted issues   "
          f"(by baseline: {sum(1 for r in covered if 'heuristic' in r['by'])}, only by LLM: {only_llm})")
    print(f"Rules evaluated: {result['total_rules']}   not matching any reference rule: {len(result['extras'])}")
    for e in result["extras"]:
        print(f"  ? {e}")

    out = PROJECT_ROOT / "data" / "evaluation" / f"rule_coverage_{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
