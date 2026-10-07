"""Calls the LLM with the built prompt and parses candidate rules into Rule objects.

One call per entity. Each candidate is validated on its own - one bad rule never discards the batch:
  1. structure  (Pydantic Rule: check type, parameters, scope, column counts)
  2. semantics  (validate_rule: entities, columns, hierarchies, expressions exist and parse)
  3. one repair round: invalid candidates go back to the LLM with their exact errors
"""
import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from app.agent.llm_client import LLMClient, LLMError
from app.rules.validation import validate_rule
from app.schemas.rule import Rule
from app.schemas.semantic_model import SemanticModel

log = logging.getLogger(__name__)

# Fields the engine sets itself - never taken from the LLM.
_ENGINE_FIELDS = {"rule_id", "source_id", "origin", "status", "reviews", "created_at", "generation_run_id",
                  "model", "fingerprint", "similarity_key"}


class CandidateList(BaseModel):
    """Lenient envelope: each item is validated separately afterwards."""
    rules: list[dict]


@dataclass
class EntityGeneration:
    entity_id: str
    valid: list[Rule] = field(default_factory=list)
    invalid: list[tuple[dict, list[str]]] = field(default_factory=list)   # (candidate, errors) after repair
    repaired: int = 0
    off_target: int = 0
    filtered_key_checks: int = 0      # generic checks on primary keys, dropped by code
    budget: int | None = None
    prompt: str = ""
    responses: list[str] = field(default_factory=list)
    error: str | None = None


def _to_rule(candidate: dict, source_id: str, run_id: str | None, model_name: str) -> Rule:
    data = {k: v for k, v in candidate.items() if k not in _ENGINE_FIELDS}
    return Rule.model_validate({**data, "source_id": source_id, "origin": "llm",
                                "generation_run_id": run_id, "model": model_name})


def _check(candidate: dict, model: SemanticModel, source_id: str, run_id: str | None,
           model_name: str) -> tuple[Rule | None, list[str]]:
    try:
        rule = _to_rule(candidate, source_id, run_id, model_name)
    except ValidationError as e:
        return None, [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors(include_url=False)]
    errors = validate_rule(rule, model)
    return (None, errors) if errors else (rule, [])


def is_primary_key_check(rule: Rule, model: SemanticModel) -> bool:
    """unique / not_null on primary-key columns: the database enforces these already."""
    if rule.check.type not in ("unique", "not_null", "null_rate_max") or not rule.target.columns:
        return False
    return set(rule.target.columns) <= set(model.entity(rule.target.entity).primary_key)


def generate_for_entity(llm: LLMClient, system: str, user: str, model: SemanticModel, entity_id: str,
                        source_id: str, run_id: str | None = None, repair: bool = True) -> EntityGeneration:
    out = EntityGeneration(entity_id, prompt=user)
    model_name = getattr(llm, "model_name", "unknown")
    try:
        first = llm.complete_structured(system, user, CandidateList)
    except LLMError as e:
        out.error = str(e)
        return out
    out.responses.append(first.model_dump_json())

    failed: list[tuple[dict, list[str]]] = []
    for cand in first.rules:
        rule, errors = _check(cand, model, source_id, run_id, model_name)
        if rule is None:
            failed.append((cand, errors))
        elif rule.target.entity != entity_id:
            out.off_target += 1            # asked for in another entity's call - avoid double proposals
        elif is_primary_key_check(rule, model):
            out.filtered_key_checks += 1
        else:
            out.valid.append(rule)

    if failed and repair:
        problems = [{"candidate": c, "errors": errs} for c, errs in failed]
        repair_msg = ("Some of your rules are invalid. Fix each one using its errors and return ONLY the "
                      "corrected rules as {\"rules\": [...]}; drop any you cannot fix.\n"
                      + json.dumps(problems, indent=1, default=str))
        try:
            fixed = llm.complete_structured(system, repair_msg, CandidateList,
                                            prior=[{"role": "user", "content": user},
                                                   {"role": "assistant", "content": first.model_dump_json()}])
            out.responses.append(fixed.model_dump_json())
            for cand in fixed.rules:
                rule, _ = _check(cand, model, source_id, run_id, model_name)
                if rule is not None and rule.target.entity == entity_id and not is_primary_key_check(rule, model):
                    out.valid.append(rule)
                    out.repaired += 1
        except LLMError as e:
            log.warning("repair round failed for %s: %s", entity_id, e)
    out.invalid = failed        # original failures with their errors; `repaired` of them were recovered
    return out
