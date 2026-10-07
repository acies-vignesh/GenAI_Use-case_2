"""Execution-based scoring: run the rules on the demo DB and check they flag the planted rows.

    cd backend
    .venv/Scripts/python -m evaluation.rule_execution_coverage                     # approved rules
    .venv/Scripts/python -m evaluation.rule_execution_coverage --status any        # all but rejected
    .venv/Scripts/python -m evaluation.rule_execution_coverage --rules x.json --label run1

Only rules that FAIL count (a 5% null-rate limit flags null rows but passes at 3% - no alert).
A planted issue is CAUGHT when the failing rules on its table flag >= 50% of its affected rows (PARTIAL when
fewer, but some). Each rule also gets "unexplained" failures: flagged rows not in any planted issue
of that table - either real problems nobody planted, or a sign that the rule is wrong.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from app.config import PROJECT_ROOT
from app.persistence.repositories import unit_of_work
from app.rules.executor import execute_rule
from app.schemas.rule import Rule
from app.services.pipeline import open_source

MANIFEST = PROJECT_ROOT / "samples" / "source_db" / "planted_issues.json"
CAUGHT_SHARE = 0.5


def evaluate(engine, rules: list[Rule], model, schema=None) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["issues"]
    planted_by_table: dict[str, set] = defaultdict(set)
    for i in manifest:
        planted_by_table[i["table"]] |= set(i["affected_keys"])

    flagged: dict[str, dict[str, set]] = defaultdict(dict)     # table -> rule_id -> keys
    per_rule = []
    for r in rules:
        res = execute_rule(engine, r, model, schema, collect_keys=True)
        table = model.entity(r.target.entity).table
        # a rule only raises an alert when it FAILS overall (e.g. null rate above its limit)
        keys = set(res.failing_keys) if not res.passed else set()
        flagged[table][r.rule_id] = keys
        per_rule.append({"rule": r.name, "entity": r.target.entity, "origin": r.origin.value, "check": r.check.type,
                         "failed": res.failed, "rate": res.failure_rate, "passed": res.passed, "error": res.error,
                         "unexplained": len(keys - planted_by_table[table])})

    names = {r.rule_id: f"[{r.origin.value}] {r.name}" for r in rules}
    issues = []
    for i in manifest:
        affected = set(i["affected_keys"])
        best, by = set(), []
        for rid, keys in flagged[i["table"]].items():
            hit = keys & affected
            if hit:
                by.append((len(hit), names[rid]))
                best |= hit
        share = len(best) / len(affected) if affected else 0
        issues.append({"issue_id": i["issue_id"], "dimension": i["dimension"], "table": i["table"],
                       "description": i["description"], "flagged": len(best), "affected": len(affected),
                       "status": "caught" if share >= CAUGHT_SHARE else "partial" if best else "missed",
                       "by": [n for _, n in sorted(by, reverse=True)[:3]]})
    return {"issues": issues, "rules": per_rule}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=Path, help="JSON list of rules instead of the app DB")
    parser.add_argument("--status", default="approved", choices=["approved", "proposed", "any"])
    parser.add_argument("--source")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    with unit_of_work() as repo:
        source_id = args.source or repo.sources.list()[0].id
        model = repo.semantic_models.latest(source_id)
        if args.rules:
            rules = [Rule.model_validate(r) for r in json.loads(args.rules.read_text(encoding="utf-8"))]
        elif args.status == "any":
            rules = [r for r in repo.rules.list(source_id) if r.status.value != "rejected"]
        else:
            rules = repo.rules.list(source_id, status=args.status)
    config, engine = open_source(source_id)
    result = evaluate(engine, rules, model, config.schema_name)

    marks = {"caught": "[x]", "partial": "[~]", "missed": "[ ]"}
    for i in result["issues"]:
        print(f"{marks[i['status']]} {i['issue_id']} {i['flagged']:>3}/{i['affected']:<3} {i['dimension']:<22} "
              f"{i['table']}: {i['description']}")
    status = Counter(i["status"] for i in result["issues"])
    print(f"\nRules run: {len(rules)}   caught {status['caught']}/{len(result['issues'])}   "
          f"partial {status['partial']}   missed {status['missed']}")
    errors = [r for r in result["rules"] if r["error"]]
    if errors:
        print(f"Rules that could not run: {len(errors)}")
        for r in errors:
            print(f"  ! {r['rule']}: {r['error'][:120]}")
    noisy = sorted((r for r in result["rules"] if r["unexplained"]), key=lambda r: -r["unexplained"])[:8]
    if noisy:
        print("Most unexplained failures (real unplanted problems, or a wrong rule?):")
        for r in noisy:
            print(f"  {r['unexplained']:>6}  [{r['origin']}] {r['entity']}: {r['rule']}")

    out = PROJECT_ROOT / "data" / "evaluation" / f"rule_execution_{args.label or args.status}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
