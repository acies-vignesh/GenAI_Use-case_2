"""Review use cases (Step 8): what the review screen calls.

    review_queue(source_id)                 proposed rules + similar-rule warnings (+ evidence)
    evidence(source_id, rule_ids)           run rules on the source: failures, rate, masked examples
    approve / reject / modify / bulk_approve / add_steward_rule
"""
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.engine import Engine

from app.persistence.repositories import AddResult, NotFoundError, unit_of_work
from app.persistence.models import RuleRow
from app.review.feedback import ReviewError, apply_changes
from app.rules.executor import ExecutionResult, execute_rule
from app.rules.validation import validate_rule
from app.schemas.rule import ReasonCode, Rule, RuleStatus
from app.services.pipeline import open_source

HIGH_FAILURE_RATE = 0.10     # a rule failing >10% of rows is more often wrong than the data


@dataclass
class ReviewItem:
    rule: Rule
    similar_to: Rule | None = None
    similar_reason: str | None = None        # why the look-alike was rejected, if it was
    execution: ExecutionResult | None = None
    flags: list[str] = field(default_factory=list)


def _flags(item: ReviewItem) -> list[str]:
    flags = []
    if item.rule.check.type == "custom_sql":
        flags.append("custom SQL - check the query carefully")
    if item.similar_to is not None:
        s = item.similar_to
        flags.append(f"similar to {s.status.value} rule '{s.name}'"
                     + (f" (rejected: {item.similar_reason})" if item.similar_reason else ""))
    ex = item.execution
    if ex is not None:
        if ex.error:
            flags.append(f"can't run: {ex.error}")
        elif ex.failure_rate > HIGH_FAILURE_RATE:
            flags.append(f"fails {ex.failure_rate:.0%} of rows - is the rule itself wrong?")
        elif ex.failed and ex.passed:
            limit = item.rule.check.max_rate if item.rule.check.type == "null_rate_max" else item.rule.tolerance
            flags.append(f"flags {ex.failed:,} rows but PASSES (limit {limit:.1%}) - is the threshold right?")
    return flags


def resolve_rule_id(prefix: str, app_engine: Engine | None = None) -> str:
    """CLI convenience: accept the first characters of a rule id."""
    from sqlalchemy import select
    with unit_of_work(app_engine) as repo:
        ids = list(repo.session.scalars(select(RuleRow.id).where(RuleRow.id.like(f"{prefix}%")).limit(2)))
    if len(ids) != 1:
        raise NotFoundError(f"rule id '{prefix}' is {'ambiguous' if ids else 'unknown'}")
    return ids[0]


def evidence(source_id: str, rules: list[Rule], password: str | None = None,
             app_engine: Engine | None = None) -> dict[str, ExecutionResult]:
    with unit_of_work(app_engine) as repo:
        model = repo.semantic_models.latest(source_id)
    config, engine = open_source(source_id, password, app_engine)
    try:
        return {r.rule_id: execute_rule(engine, r, model, config.schema_name) for r in rules}
    finally:
        engine.dispose()


def review_queue(source_id: str, status: str | None = "proposed", entity: str | None = None,
                 origin: str | None = None, with_evidence: bool = False,
                 app_engine: Engine | None = None) -> list[ReviewItem]:
    with unit_of_work(app_engine) as repo:
        rules = repo.rules.list(source_id, status=status, entity=entity, origin=origin)
        items = []
        for r in rules:
            item = ReviewItem(r)
            sim_id = repo.rules.similar_to(r.rule_id)
            if sim_id:
                item.similar_to = repo.rules.get(sim_id)
                if item.similar_to.status == RuleStatus.REJECTED and item.similar_to.reviews:
                    item.similar_reason = item.similar_to.reviews[-1].reason
            items.append(item)
    if with_evidence and items:
        results = evidence(source_id, [i.rule for i in items], app_engine=app_engine)
        for i in items:
            i.execution = results[i.rule.rule_id]
    for i in items:
        i.flags = _flags(i)
    return items


def approve(rule_ids: list[str], reviewer: str, note: str | None = None,
            app_engine: Engine | None = None) -> list[Rule]:
    with unit_of_work(app_engine) as repo:
        return [repo.rules.record_review(rid, "approved", reviewer, reason=note) for rid in rule_ids]


def reject(rule_id: str, reviewer: str, reason_code: ReasonCode | str, reason: str,
           app_engine: Engine | None = None) -> Rule:
    if not reason or not reason.strip():
        raise ReviewError("a rejection needs a reason - it is what the engine learns from")
    with unit_of_work(app_engine) as repo:
        return repo.rules.record_review(rule_id, "rejected", reviewer, reason=reason.strip(),
                                        reason_code=ReasonCode(reason_code))


def modify(rule_id: str, reviewer: str, changes: dict[str, Any], reason: str | None = None,
           reason_code: ReasonCode | str | None = None, app_engine: Engine | None = None) -> Rule:
    with unit_of_work(app_engine) as repo:
        rule = repo.rules.get(rule_id)
        model = repo.semantic_models.latest(rule.source_id)
        edited = apply_changes(rule, changes, model)
        return repo.rules.record_review(rule_id, "modified", reviewer, reason=reason, edited=edited,
                                        reason_code=reason_code)


def bulk_approve(source_id: str, reviewer: str, entity: str | None = None, origin: str | None = None,
                 check_type: str | None = None, note: str | None = None,
                 app_engine: Engine | None = None) -> int:
    with unit_of_work(app_engine) as repo:
        rules = [r for r in repo.rules.list(source_id, status="proposed", entity=entity, origin=origin)
                 if check_type is None or r.check.type == check_type]
        for r in rules:
            repo.rules.record_review(r.rule_id, "approved", reviewer, reason=note or "bulk approval")
    return len(rules)


def add_steward_rule(source_id: str, rule_data: dict, reviewer: str, app_engine: Engine | None = None) -> AddResult:
    """Step 5: a rule or threshold the steward knows applies - stored as approved, origin steward."""
    rule = Rule.model_validate({**rule_data, "source_id": source_id, "origin": "steward", "status": "approved"})
    with unit_of_work(app_engine) as repo:
        errors = validate_rule(rule, repo.semantic_models.latest(source_id))
        if errors:
            raise ReviewError("; ".join(errors))
        return repo.rules.add(rule, created_by=reviewer)
