import json
import sqlite3

import pytest
from sqlalchemy import text

from app.config import PROJECT_ROOT
from app.connectivity.connectors import ConnectionConfig, create_source_engine
from app.orchestration.context_assembler import build_entity_context
from app.persistence.database import make_engine, upgrade
from app.persistence.models import SourceRow
from app.persistence.repositories import unit_of_work
from app.review.feedback import ReviewError
from app.rules.executor import execute_rule
from app.schemas.rule import Rule
from app.schemas.semantic_model import SemanticModel
from app.services import pipeline, review
from app.services.generation import generate_rules
from tests.conftest import DEMO_DB
from tests.test_phase4_semantic import FakeLLM

REF_MODEL = PROJECT_ROOT / "samples" / "semantic_models" / "demo_retail.reference.json"
REF_RULES = PROJECT_ROOT / "samples" / "rules" / "demo_retail.reference_rules.json"
MANIFEST = PROJECT_ROOT / "samples" / "source_db" / "planted_issues.json"

COD_WRONG = {"name": "Online orders cannot use COD", "category": "business", "dimension": "consistency",
             "scope": "row", "target": {"entity": "order", "columns": ["channel", "payment_method"]},
             "check": {"type": "conditional", "when": "channel = 'ONLINE'", "then": "payment_method <> 'COD'"},
             "severity": "high", "rationale": "context"}
EMAIL_LOOSE = {"name": "Email mostly populated", "category": "data_quality", "dimension": "completeness",
               "scope": "column", "target": {"entity": "customer", "columns": ["email"]},
               "check": {"type": "null_rate_max", "max_rate": 0.05}, "severity": "medium"}
PK_CHECK = {"name": "Order ID present", "category": "data_quality", "dimension": "completeness",
            "scope": "column", "target": {"entity": "order", "columns": ["order_id"]},
            "check": {"type": "not_null"}, "severity": "critical"}


def _llm(*rules):
    return json.dumps({"rules": list(rules)})


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    if not DEMO_DB.exists():
        pytest.skip("Demo DB not built")
    url = f"sqlite:///{(tmp_path_factory.mktemp('rev') / 'app.db').as_posix()}"
    upgrade(url)
    engine = make_engine(url)
    sid = pipeline.connect_source(ConnectionConfig(dialect="sqlite", database=str(DEMO_DB)), app_engine=engine)
    pipeline.build_model(sid, app_engine=engine)
    generate_rules(sid, llm=FakeLLM(_llm(COD_WRONG, PK_CHECK)), entities=["order"], app_engine=engine)
    generate_rules(sid, llm=FakeLLM(_llm(EMAIL_LOOSE)), entities=["customer"], include_baseline=False,
                   app_engine=engine)
    yield engine, sid
    engine.dispose()


def _rule(engine, sid, name) -> Rule:
    with unit_of_work(engine) as repo:
        return next(r for r in repo.rules.list(sid) if r.name == name)


# --------------------------------------------------------------------------- migration on existing data
def test_migration_0002_keeps_existing_data(tmp_path):
    url = f"sqlite:///{(tmp_path / 'old.db').as_posix()}"
    upgrade(url, revision="0001")
    eng = make_engine(url)
    with eng.begin() as c:
        c.execute(text("INSERT INTO sources (id, display_name, dialect, database, store_password, created_at) "
                       "VALUES ('s1', 'x', 'sqlite', 'x.db', 1, '2026-01-01')"))
        c.execute(text("INSERT INTO rules (id, source_id, status, origin, category, dimension, severity, "
                       "target_entity, name, fingerprint, original_fingerprint, similarity_key, payload, "
                       "created_at, updated_at) VALUES ('r1', 's1', 'rejected', 'llm', 'business', 'validity', "
                       "'low', 'order', 'old rule', 'f1', 'f1', 'k', '{}', '2026-01-01', '2026-01-01')"))
        c.execute(text("INSERT INTO rule_reviews (id, rule_id, decision, reviewer, reason, created_at) "
                       "VALUES ('v1', 'r1', 'rejected', 'alice', 'not needed', '2026-01-01')"))
    upgrade(url)
    with eng.connect() as c:
        assert c.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0002"
        assert c.execute(text("SELECT reason, reason_code FROM rule_reviews")).one() == ("not needed", None)
        assert c.execute(text("SELECT name FROM rules")).scalar() == "old rule"
    eng.dispose()


# --------------------------------------------------------------------------- executor
def test_reference_rules_flag_every_planted_issue():
    model = SemanticModel.model_validate_json(REF_MODEL.read_text(encoding="utf-8"))
    engine = create_source_engine(ConnectionConfig(dialect="sqlite", database=str(DEMO_DB)))
    flagged: dict[str, set] = {}
    for item in json.loads(REF_RULES.read_text(encoding="utf-8"))["rules"]:
        res = execute_rule(engine, Rule.model_validate(item["rule"]), model, collect_keys=True)
        assert res.error is None, (item["rule"]["name"], res.error)
        for pi in item["covers"]:
            flagged.setdefault(pi, set()).update(res.failing_keys)
    for issue in json.loads(MANIFEST.read_text(encoding="utf-8"))["issues"]:
        assert set(issue["affected_keys"]) <= flagged[issue["issue_id"]], issue["issue_id"]


@pytest.fixture
def tiny(tmp_path):
    path = tmp_path / "t.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE node (id INTEGER PRIMARY KEY, parent_id INTEGER, lvl TEXT, code TEXT, amount REAL);
        INSERT INTO node VALUES (1, NULL, 'TOP', 'AB', 10), (2, 1, 'MID', 'cd', NULL), (3, 4, 'MID', 'EF', -1),
                                (4, 3, 'MID', NULL, 5);
    """)
    con.close()
    model = SemanticModel.model_validate({
        "source_id": "s", "based_on_metadata": "2026-01-01T00:00:00Z", "domain": {"name": "t"},
        "entities": [{"entity_id": "node", "name": "Node", "table": "node", "entity_type": "hierarchy",
                      "grain": "g", "primary_key": ["id"], "provenance": {"source": "steward"},
                      "attributes": [{"column": c, "name": c, "semantic_type": "other", "role": "attribute",
                                      "provenance": {"source": "steward"}}
                                     for c in ("id", "parent_id", "lvl", "code", "amount")]}],
        "hierarchies": [{"hierarchy_id": "h", "name": "H", "style": "self_referencing", "levels": ["TOP", "MID"],
                         "entity": "node", "parent_attribute": "parent_id", "level_attribute": "lvl",
                         "provenance": {"source": "steward"}}]})
    return create_source_engine(ConnectionConfig(dialect="sqlite", database=str(path))), model


def _tiny_rule(**kw) -> Rule:
    base = {"source_id": "s", "name": "x", "category": "data_quality", "dimension": "validity", "scope": "column",
            "origin": "steward", "target": {"entity": "node", "columns": ["amount"]}}
    return Rule.model_validate({**base, **kw})


def test_null_never_fails_except_null_checks(tiny):
    engine, model = tiny
    assert execute_rule(engine, _tiny_rule(check={"type": "range", "min": 0}), model).failed == 1   # NULL ignored
    assert execute_rule(engine, _tiny_rule(check={"type": "not_null"}), model).failed == 1
    res = execute_rule(engine, _tiny_rule(check={"type": "null_rate_max", "max_rate": 0.5}), model)
    assert (res.failed, res.passed) == (1, True)                                       # 25% <= 50%


def test_pattern_cycle_and_custom_sql(tiny):
    engine, model = tiny
    pat = execute_rule(engine, _tiny_rule(target={"entity": "node", "columns": ["code"]},
                                          check={"type": "pattern", "regex": "^[A-Z]{2}$"}), model, collect_keys=True)
    assert pat.failing_keys == [2]
    cyc = execute_rule(engine, _tiny_rule(category="hierarchy", scope="table", target={"entity": "node"},
                                          check={"type": "hierarchy_acyclic", "hierarchy": "h"}), model,
                       collect_keys=True)
    assert sorted(cyc.failing_keys) == [3, 4]
    broken = execute_rule(engine, _tiny_rule(scope="table", target={"entity": "node"},
                                             check={"type": "custom_sql", "sql": "SELECT * FROM nodes"}), model)
    assert broken.error and "no such table" in broken.error and not broken.passed


# --------------------------------------------------------------------------- review actions
def test_reject_needs_a_reason_and_records_code(demo):
    engine, sid = demo
    rule = _rule(engine, sid, "Online orders cannot use COD")
    with pytest.raises(ReviewError, match="needs a reason"):
        review.reject(rule.rule_id, "alice", "wrong", "  ", app_engine=engine)
    review.reject(rule.rule_id, "alice", "wrong", "COD is only disallowed in stores", app_engine=engine)
    with unit_of_work(engine) as repo:
        rv = repo.rules.reviews(rule.rule_id)[-1]
        assert (rv.decision, rv.reason_code) == ("rejected", "wrong")


def test_modify_is_validated_then_approved(demo):
    engine, sid = demo
    rule = _rule(engine, sid, "Email mostly populated")
    with pytest.raises(ReviewError, match="unknown column"):
        review.modify(rule.rule_id, "alice", {"target.columns": ["e_mail"]}, app_engine=engine)
    with pytest.raises(ReviewError, match="can't be edited"):
        review.modify(rule.rule_id, "alice", {"origin": "steward"}, app_engine=engine)
    with pytest.raises(ReviewError, match="less than or equal to 1"):
        review.modify(rule.rule_id, "alice", {"check.max_rate": 5}, app_engine=engine)
    new = review.modify(rule.rule_id, "alice", {"check.max_rate": 0.01, "severity": "high"},
                        reason="1% is the target", reason_code="too_loose", app_engine=engine)
    assert (new.check.max_rate, new.severity.value, new.status.value) == (0.01, "high", "approved")


def test_pk_checks_are_filtered_from_llm_output(demo):
    engine, sid = demo
    with unit_of_work(engine) as repo:
        assert not [r for r in repo.rules.list(sid) if r.name == "Order ID present"]


def test_bulk_approve_and_change_of_mind(demo):
    engine, sid = demo
    n = review.bulk_approve(sid, "alice", entity="order", origin="heuristic", app_engine=engine)
    assert n > 5
    with unit_of_work(engine) as repo:
        some = next(r for r in repo.rules.list(sid, entity="order")
                    if r.check.type == "unique" and r.target.columns == ["order_number"])
    review.reject(some.rule_id, "alice", "other", "changed my mind", app_engine=engine)
    review.approve([some.rule_id], "alice", app_engine=engine)
    with unit_of_work(engine) as repo:
        assert repo.rules.get(some.rule_id).status.value == "approved"
    with unit_of_work(engine) as repo:
        assert [r.decision for r in repo.rules.reviews(some.rule_id)] == ["approved", "rejected", "approved"]


def test_steward_rule_is_validated_and_approved(demo):
    engine, sid = demo
    rule = {"name": "Phone mostly populated", "category": "data_quality", "dimension": "completeness",
            "scope": "column", "target": {"entity": "customer", "columns": ["phone"]},
            "check": {"type": "null_rate_max", "max_rate": 0.01}}
    added = review.add_steward_rule(sid, rule, "alice", app_engine=engine)
    assert added.outcome in ("stored", "similar") and added.rule.status.value == "approved"
    with pytest.raises(ReviewError):
        review.add_steward_rule(sid, {**rule, "target": {"entity": "customer", "columns": ["mobile"]}}, "alice",
                                app_engine=engine)


def test_queue_flags_wrong_rules_with_evidence(demo):
    engine, sid = demo
    with unit_of_work(engine) as repo:
        # NB: "payment_method != 'COD'" would be normalised to the rejected rule's "<>" and dropped as a duplicate
        added = repo.rules.add(Rule.model_validate({**COD_WRONG, "name": "COD online v2", "source_id": sid,
                                                    "origin": "llm",
                                                    "check": {**COD_WRONG["check"], "when": "channel <> 'STORE'"}}))
        assert added.outcome in ("stored", "similar")
    items = review.review_queue(sid, entity="order", origin="llm", with_evidence=True, app_engine=engine)
    item = next(i for i in items if i.rule.name == "COD online v2")
    assert item.execution.failure_rate > 0.1
    assert any("is the rule itself wrong" in f for f in item.flags)


# --------------------------------------------------------------------------- learning signals in the prompt
def test_patterns_and_budget_reach_the_prompt(demo, monkeypatch):
    from app.services import generation
    monkeypatch.setattr(generation, "WELL_COVERED", 5)
    engine, sid = demo
    with unit_of_work(engine) as repo:
        model, meta = repo.semantic_models.latest(sid), repo.snapshots.latest(sid)
        history = repo.rules.history_for_prompt(sid)
    ctx = build_entity_context(model, meta, "order", [], history, [])
    assert any("conditional: approved 0, rejected" in p and "wrong x1" in p for p in ctx.patterns)
    llm = FakeLLM(_llm())
    result = generate_rules(sid, llm=llm, entities=["order"], store=False, app_engine=engine)
    assert result.per_entity["order"].budget == 5                      # well-covered entity -> small budget
    assert "propose up to 5 NEW rules" in llm.prompts[0][1]["content"]
    assert "COD is only disallowed in stores" in llm.prompts[0][1]["content"]
