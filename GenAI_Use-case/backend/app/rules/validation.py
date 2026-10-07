"""Validate a rule against the semantic model: does everything it names actually exist?

Pydantic (schemas/rule.py) already guarantees the rule is well-formed. This adds the checks that need
the model: entities, columns, hierarchies and levels exist, and every expression parses and only uses
columns of the right entity. Used on LLM output (reject hallucinations) and on steward edits alike.
"""
import sqlglot
from sqlglot import exp

from app.rules.expressions import ExpressionError, validate_expression
from app.schemas.rule import (
    AggregateCompare,
    Conditional,
    CustomSql,
    ForeignKey,
    HasChildren,
    HierarchyAcyclic,
    HierarchyAttachLevel,
    HierarchyParentLevel,
    RowCondition,
    Rule,
)
from app.schemas.semantic_model import SemanticModel


def validate_rule(rule: Rule, model: SemanticModel) -> list[str]:
    errors: list[str] = []
    entities = {e.entity_id: e for e in model.entities}
    hierarchies = {h.hierarchy_id: h for h in model.hierarchies}

    target = entities.get(rule.target.entity)
    if target is None:
        return [f"unknown entity '{rule.target.entity}'"]

    def cols_exist(entity_id: str, cols: list[str], label: str) -> None:
        if entity_id not in entities:
            errors.append(f"{label}: unknown entity '{entity_id}'")
            return
        missing = [c for c in cols if c not in entities[entity_id].columns]
        if missing:
            errors.append(f"{label}: unknown column(s) {missing} in '{entity_id}'")

    def expr_ok(text: str, entity_id: str, label: str, aggregate: bool | None = False) -> None:
        if entity_id not in entities:
            return
        try:
            validate_expression(text, entities[entity_id].columns, aggregate=aggregate)
        except ExpressionError as e:
            errors.append(f"{label}: {e}")

    cols_exist(target.entity_id, rule.target.columns, "target")
    if rule.filter:
        expr_ok(rule.filter, target.entity_id, "filter")

    c = rule.check
    if isinstance(c, RowCondition):
        expr_ok(c.expression, target.entity_id, "expression")
    elif isinstance(c, Conditional):
        expr_ok(c.when, target.entity_id, "when")
        expr_ok(c.then, target.entity_id, "then")
    elif isinstance(c, ForeignKey):
        cols_exist(c.ref_entity, c.ref_columns, "ref")
    elif isinstance(c, (HasChildren, AggregateCompare)):
        cols_exist(c.child_entity, c.child_join_columns, "child join")
        cols_exist(target.entity_id, c.parent_join_columns, "parent join")
        if len(c.child_join_columns) != len(c.parent_join_columns):
            errors.append("child_join_columns and parent_join_columns must pair 1:1")
        if isinstance(c, AggregateCompare):
            expr_ok(c.child_expression, c.child_entity, "child_expression", aggregate=True)
            expr_ok(c.parent_expression, target.entity_id, "parent_expression")
    elif isinstance(c, (HierarchyParentLevel, HierarchyAcyclic, HierarchyAttachLevel)):
        h = hierarchies.get(c.hierarchy)
        if h is None:
            errors.append(f"unknown hierarchy '{c.hierarchy}'")
        elif isinstance(c, HierarchyAttachLevel):
            if c.level not in h.levels:
                errors.append(f"level '{c.level}' is not one of {h.levels}")
        elif h.style == "self_referencing" and h.entity != target.entity_id:
            errors.append(f"hierarchy '{h.hierarchy_id}' lives on '{h.entity}', not '{target.entity_id}'")
    elif isinstance(c, CustomSql):
        try:
            if not isinstance(sqlglot.parse_one(c.sql), exp.Select):
                errors.append("custom_sql must be a single SELECT returning failing rows")
        except Exception as e:
            errors.append(f"custom_sql does not parse: {e}")

    return errors
