"""How many planted issues leave visible evidence in the data profile?

    cd backend
    .venv/Scripts/python -m evaluation.profile_coverage [--metadata ../data/metadata/metadata_<id>.json]

For each planted issue, look at the profile of its columns for the signals that matter for its
dimension (e.g. completeness -> nulls, uniqueness -> near-unique column). Cross-column dimensions
(consistency, business_rule, hierarchy) can't be proven by single-column stats - those are the
cases the semantic layer, business context and the LLM must cover.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from app.config import PROJECT_ROOT
from app.schemas.metadata import ColumnMetadata, SourceMetadata, TableMetadata

MANIFEST = PROJECT_ROOT / "samples" / "source_db" / "planted_issues.json"
RARE_SHARE = 0.02          # a value/pattern this rare next to dominant ones is suspicious
NEAR_UNIQUE = 0.95

SIGNALS_BY_DIMENSION = {
    "completeness": {"nulls"},
    "validity": {"rare_values", "minority_patterns", "format_mismatch", "negatives", "zeros"},
    "uniqueness": {"near_unique"},
    "referential_integrity": {"orphans"},
    "accuracy": {"placeholders", "outliers", "rare_values"},
    "timeliness": {"future_dates"},
    "consistency": set(),       # cross-column
    "business_rule": set(),     # cross-column / domain knowledge
    "hierarchy": set(),         # needs semantic understanding of levels
}


def column_signals(table: TableMetadata, col: ColumnMetadata) -> dict[str, str]:
    p = col.profile
    s: dict[str, str] = {}
    if p.null_count:
        s["nulls"] = f"{p.null_count} nulls ({p.null_rate:.1%})"
    if p.uniqueness is not None and NEAR_UNIQUE <= p.uniqueness < 1:
        s["near_unique"] = f"uniqueness {p.uniqueness:.4f}"
    if p.top_values and len(p.top_values) > 1:
        rare = [v.value for v in p.top_values if v.value is not None and v.share < RARE_SHARE]
        dominant = sum(v.share for v in p.top_values if v.share >= RARE_SHARE)
        if rare and dominant >= 0.9:
            s["rare_values"] = f"rare values {rare[:6]}"
    if p.patterns and len(p.patterns) > 1:
        minority = [pt.pattern for pt in p.patterns[1:] if pt.share < RARE_SHARE]
        if minority:
            s["minority_patterns"] = f"minority patterns {minority[:4]}"
    if p.detected_format and p.detected_format.match_rate < 1:
        s["format_mismatch"] = f"{p.detected_format.format} match {p.detected_format.match_rate:.2%}"
    if p.numeric:
        if p.numeric.negative_count:
            s["negatives"] = f"{p.numeric.negative_count} negative"
        if p.numeric.zero_count:
            s["zeros"] = f"{p.numeric.zero_count} zero"
        if p.numeric.outlier_count:
            s["outliers"] = f"max {p.numeric.max} vs p99 {p.numeric.p99}"
    if p.date:
        if p.date.placeholder_count:
            s["placeholders"] = f"{p.date.placeholder_count} placeholder dates (min {p.date.min})"
        if p.date.future_count:
            s["future_dates"] = f"{p.date.future_count} future dates (max {p.date.max})"
    for fk in table.foreign_keys:
        if col.name in fk.columns and fk.orphan_count:
            s["orphans"] = f"{fk.orphan_count} orphans -> {fk.referred_table}"
    return s


def evaluate(metadata: SourceMetadata, manifest: dict) -> list[dict]:
    results = []
    for issue in manifest["issues"]:
        table = metadata.table(issue["table"])
        wanted = SIGNALS_BY_DIMENSION[issue["dimension"]]
        direct, incidental = [], []
        for name in issue["columns"]:
            for tag, text in column_signals(table, table.column(name)).items():
                (direct if tag in wanted else incidental).append(f"{name}: {text}")
        status = "visible" if direct else ("hint" if incidental else "not visible")
        results.append({**{k: issue[k] for k in ("issue_id", "table", "dimension", "description")},
                        "status": status, "evidence": direct or incidental})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path,
                        help="Profiled metadata JSON (default: the only file in data/metadata)")
    args = parser.parse_args()
    path = args.metadata or next((PROJECT_ROOT / "data" / "metadata").glob("metadata_*.json"))
    metadata = SourceMetadata.model_validate_json(path.read_text(encoding="utf-8"))
    if metadata.profiled_at is None:
        raise SystemExit("Metadata has no profile - run `python -m app.cli profile` first.")

    results = evaluate(metadata, json.loads(MANIFEST.read_text(encoding="utf-8")))

    for r in results:
        mark = {"visible": "[x]", "hint": "[~]", "not visible": "[ ]"}[r["status"]]
        print(f"{mark} {r['issue_id']} {r['dimension']:<22} {r['table']}: {r['description']}")
        if r["evidence"]:
            print(f"        {r['evidence'][0]}")

    print("\nBy dimension (visible / total):")
    total, seen = Counter(r["dimension"] for r in results), Counter(
        r["dimension"] for r in results if r["status"] == "visible")
    for dim in SIGNALS_BY_DIMENSION:
        if total[dim]:
            print(f"  {dim:<22} {seen[dim]:>2} / {total[dim]}")
    counts = Counter(r["status"] for r in results)
    print(f"\nVisible in profile: {counts['visible']}/{len(results)}   hints: {counts['hint']}   "
          f"not visible: {counts['not visible']}")

    out = PROJECT_ROOT / "data" / "evaluation" / "profile_coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
