"""Rule identity for de-duplication.

fingerprint    = WHAT the rule checks, exactly: check type + target + normalised parameters + filter.
                 Name, description, severity, tolerance and rationale are deliberately excluded, so the
                 same rule reworded by the LLM still gets the same fingerprint.
                 Exact match with a rejected rule -> drop the candidate automatically.
similarity_key = WHERE the rule checks: check type + target (+ the other entity / hierarchy involved).
                 Same key, different parameters (e.g. range 0-100 vs 0-120) -> flag for the steward.
"""
import hashlib
import json
import re

from app.rules.expressions import normalize

_EXPRESSION_FIELDS = ("expression", "when", "then", "child_expression", "parent_expression", "sql")
_RELATED_FIELDS = ("ref_entity", "child_entity", "hierarchy")


def _norm_expr(text: str) -> str:
    try:
        return normalize(text)
    except Exception:   # invalid expressions still need a stable identity
        return re.sub(r"\s+", " ", text.strip().lower())


def _sort_values(values: list) -> list:
    return sorted(values, key=lambda v: (type(v).__name__, str(v)))


def canonical(rule) -> dict:
    check = rule.check.model_dump(exclude_none=True)
    if "values" in check:
        values = check["values"]
        if check.get("case_sensitive") is False:
            values = [v.upper() if isinstance(v, str) else v for v in values]
        check["values"] = _sort_values(list(dict.fromkeys(values)))
    for f in _EXPRESSION_FIELDS:
        if f in check:
            check[f] = _norm_expr(check[f])

    columns = [c.lower() for c in rule.target.columns]
    if check["type"] == "foreign_key":       # keep column pairs aligned while sorting
        pairs = sorted(zip(columns, [c.lower() for c in check["ref_columns"]]))
        columns, check["ref_columns"] = [p[0] for p in pairs], [p[1] for p in pairs]
    else:
        columns = sorted(columns)

    return {
        "entity": rule.target.entity,
        "columns": columns,
        "check": check,
        "filter": _norm_expr(rule.filter) if rule.filter else None,
    }


def fingerprint(rule) -> str:
    payload = json.dumps(canonical(rule), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def similarity_key(rule) -> str:
    related = [str(getattr(rule.check, f)) for f in _RELATED_FIELDS if getattr(rule.check, f, None)]
    parts = [rule.check.type, rule.target.entity, ",".join(sorted(c.lower() for c in rule.target.columns))]
    return "|".join(parts + related)
