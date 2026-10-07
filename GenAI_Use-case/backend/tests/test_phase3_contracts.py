import json
import re

import pytest
from pydantic import ValidationError

from app.config import PROJECT_ROOT
from app.rules.expressions import ExpressionError, normalize, to_dialect, validate_expression
from app.rules.validation import validate_rule
from app.schemas.rule import Rule
from app.schemas.semantic_model import SemanticModel

MODEL_FILE = PROJECT_ROOT / "samples" / "semantic_models" / "demo_retail.reference.json"
RULES_FILE = PROJECT_ROOT / "samples" / "rules" / "demo_retail.reference_rules.json"
MANIFEST = PROJECT_ROOT / "samples" / "source_db" / "planted_issues.json"


@pytest.fixture(scope="module")
def model() -> SemanticModel:
    return SemanticModel.model_validate_json(MODEL_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reference_rules() -> list[dict]:
    return json.loads(RULES_FILE.read_text(encoding="utf-8"))["rules"]


def make_rule(**overrides) -> Rule:
    base = dict(source_id="s", name="Order status valid", category="data_quality", dimension="validity",
                scope="column", target={"entity": "order", "columns": ["status"]},
                check={"type": "allowed_values", "values": ["PLACED", "SHIPPED"]}, origin="llm")
    base.update(overrides)
    return Rule.model_validate(base)


# --------------------------------------------------------------------------- semantic model
def test_reference_model_loads(model):
    assert {e.entity_id for e in model.entities} == {"region", "category", "product", "customer", "order",
                                                     "order_item"}
    assert model.hierarchy("geography").levels == ["ZONE", "STATE", "CITY"]
    assert model.entity("customer").attribute("email").pii


def test_model_rejects_dangling_references(model):
    data = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    data["relationships"][0]["to_entity"] = "client"
    data["hierarchies"][0]["attached_entities"][0]["must_attach_at"] = "DISTRICT"
    with pytest.raises(ValidationError) as e:
        SemanticModel.model_validate(data)
    assert "unknown entity 'client'" in str(e.value)
    assert "'DISTRICT' is not one of its levels" in str(e.value)


# --------------------------------------------------------------------------- reference rules
def test_every_reference_rule_is_valid(model, reference_rules):
    for item in reference_rules:
        rule = Rule.model_validate(item["rule"])
        assert validate_rule(rule, model) == [], rule.name


def test_reference_rules_cover_all_planted_issues(reference_rules):
    planted = {i["issue_id"] for i in json.loads(MANIFEST.read_text(encoding="utf-8"))["issues"]}
    covered = {pi for item in reference_rules for pi in item["covers"]}
    assert covered == planted


def test_reference_rules_have_no_duplicates(reference_rules):
    fps = [Rule.model_validate(item["rule"]).fingerprint for item in reference_rules]
    assert len(fps) == len(set(fps))


# --------------------------------------------------------------------------- fingerprints
def test_fingerprint_ignores_wording_spacing_case_and_order():
    a = make_rule(check={"type": "conditional", "when": "status = 'DELIVERED'", "then": "delivery_date IS NOT NULL"},
                  scope="row", target={"entity": "order", "columns": ["status", "delivery_date"]})
    b = make_rule(check={"type": "conditional", "when": "STATUS='DELIVERED'", "then": "Delivery_Date is not null"},
                  scope="row", target={"entity": "order", "columns": ["delivery_date", "status"]},
                  name="Delivered => has delivery date", severity="low", rationale="different words")
    assert a.fingerprint == b.fingerprint


def test_fingerprint_ignores_value_order_but_not_values():
    a = make_rule()
    b = make_rule(check={"type": "allowed_values", "values": ["SHIPPED", "PLACED"]})
    c = make_rule(check={"type": "allowed_values", "values": ["SHIPPED", "PLACED", "DELIVERED"]})
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint
    assert a.similarity_key == c.similarity_key == "allowed_values|order|status"


def test_case_insensitive_values_normalise():
    a = make_rule(check={"type": "allowed_values", "values": ["placed"], "case_sensitive": False})
    b = make_rule(check={"type": "allowed_values", "values": ["PLACED"], "case_sensitive": False})
    assert a.fingerprint == b.fingerprint


def test_rule_json_round_trip_keeps_identity():
    r = make_rule()
    again = Rule.model_validate_json(r.model_dump_json())
    assert (again.rule_id, again.fingerprint) == (r.rule_id, r.fingerprint)


# --------------------------------------------------------------------------- structural validation
@pytest.mark.parametrize("overrides, message", [
    ({"target": {"entity": "order", "columns": ["status", "channel"]}}, "exactly 1 target column"),
    ({"scope": "row"}, "needs scope ['column']"),
    ({"check": {"type": "range", "min": 10, "max": 1}}, "range min > max"),
    ({"check": {"type": "pattern", "regex": "^[A-Z"}}, "unterminated character set"),
    ({"check": {"type": "allowed_values", "values": ["A"], "colour": "red"}}, "Extra inputs are not permitted"),
    ({"check": {"type": "looks_fine"}}, "does not match any of the expected tags"),
    ({"check": {"type": "hierarchy_acyclic", "hierarchy": "geography"}, "scope": "table",
      "target": {"entity": "region"}}, "hierarchy checks must have category 'hierarchy'"),
])
def test_malformed_rules_are_rejected(overrides, message):
    with pytest.raises(ValidationError, match=re.escape(message)):
        make_rule(**overrides)


# --------------------------------------------------------------------------- semantic validation
@pytest.mark.parametrize("overrides, message", [
    ({"target": {"entity": "order", "columns": ["order_status"]}}, "unknown column(s) ['order_status']"),
    ({"scope": "row", "target": {"entity": "order", "columns": ["status"]},
      "check": {"type": "row_condition", "expression": "ship_dt >= order_date"}}, "unknown column(s) ['ship_dt']"),
    ({"category": "hierarchy", "scope": "cross_table", "target": {"entity": "customer", "columns": ["region_id"]},
      "check": {"type": "hierarchy_attach_level", "hierarchy": "geography", "level": "DISTRICT"}},
     "level 'DISTRICT' is not one of"),
    ({"category": "hierarchy", "scope": "table", "target": {"entity": "product"},
      "check": {"type": "hierarchy_acyclic", "hierarchy": "geography"}}, "lives on 'region'"),
    ({"scope": "cross_table", "target": {"entity": "order"},
      "check": {"type": "aggregate_compare", "child_entity": "order_item", "child_join_columns": ["order_id"],
                "parent_join_columns": ["order_id"], "child_expression": "line_amount", "operator": "=",
                "parent_expression": "total_amount"}}, "must be an aggregate"),
])
def test_hallucinations_are_caught(model, overrides, message):
    errors = validate_rule(make_rule(**overrides), model)
    assert any(message in e for e in errors), errors


# --------------------------------------------------------------------------- expressions
@pytest.mark.parametrize("text, message", [
    ("DROP TABLE orders", "plain expression"),
    ("status IN (SELECT status FROM orders)", "plain expression"),
    ("a = 1; DROP TABLE orders", "single expression"),
    ("o.status = 'X'", "bare column names"),
    ("SUM(quantity) > 0", "must not contain aggregates"),
])
def test_unsafe_or_invalid_expressions_rejected(text, message):
    with pytest.raises(ExpressionError, match=message):
        validate_expression(text, {"status", "quantity"})


def test_normalize_and_transpile():
    assert normalize("Ship_Date>=order_date") == normalize("ship_date >= ORDER_DATE")
    expr = "signup_date >= DATE_ADD(date_of_birth, 18, 'YEAR')"
    assert "DATEADD(YEAR, 18, date_of_birth)" in to_dialect(expr, "tsql")
    assert "INTERVAL '18 YEAR'" in to_dialect(expr, "postgres")
    assert "DATE(date_of_birth, '18 YEAR')" in to_dialect(expr, "sqlite")
