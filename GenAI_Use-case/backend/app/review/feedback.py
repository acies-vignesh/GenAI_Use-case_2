"""Apply steward decisions (approve/reject/modify), keep the original vs edited version and reason.

`apply_changes` turns a partial edit such as {"check.max_rate": 0.01, "severity": "high"} into a new,
FULLY re-validated rule - an edit can never produce a rule the engine couldn't run.
"""
import json
from typing import Any

from pydantic import ValidationError

from app.rules.validation import validate_rule
from app.schemas.rule import Rule
from app.schemas.semantic_model import SemanticModel

# Identity and lineage can't be edited; status changes only through decisions.
LOCKED_FIELDS = {"rule_id", "source_id", "origin", "generation_run_id", "model", "status", "reviews", "created_at",
                 "fingerprint", "similarity_key"}


class ReviewError(ValueError):
    pass


def parse_value(raw: str) -> Any:
    """CLI/form values: JSON when it parses ('0.01', '["A","B"]', 'null', 'true'), otherwise a string."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def apply_changes(rule: Rule, changes: dict[str, Any], model: SemanticModel) -> Rule:
    if not changes:
        raise ReviewError("no changes given")
    data = rule.model_dump(mode="json", exclude={"fingerprint", "similarity_key"})
    for path, value in changes.items():
        keys = path.split(".")
        if keys[0] in LOCKED_FIELDS:
            raise ReviewError(f"'{keys[0]}' can't be edited")
        node = data
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                raise ReviewError(f"unknown field '{path}'")
            node = node[k]
        if keys[-1] not in node and keys[0] != "check":
            raise ReviewError(f"unknown field '{path}'")
        node[keys[-1]] = value
    try:
        edited = Rule.model_validate(data)
    except ValidationError as e:
        raise ReviewError("; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}"
                                    for err in e.errors(include_url=False))) from e
    errors = validate_rule(edited, model)
    if errors:
        raise ReviewError("; ".join(errors))
    return edited
