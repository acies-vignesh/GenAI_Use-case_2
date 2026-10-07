import json
from datetime import date

import pytest

from app.connectivity.connectors import ConnectionConfig
from app.persistence.database import make_engine, upgrade
from app.persistence.repositories import unit_of_work
from app.rules.baseline import baseline_rules
from app.rules.validation import validate_rule
from app.services import pipeline
from app.services.generation import NoSemanticModelError, generate_rules
from evaluation.rule_coverage import evaluate
from tests.conftest import DEMO_DB
from tests.test_phase4_semantic import FakeLLM


def llm_rules(*rules: dict) -> str:
    return json.dumps({"rules": list(rules)})


GOOD = {"name": "Ship date on/after order date", "category": "business", "dimension": "consistency",
        "scope": "row", "target": {"entity": "order", "columns": ["order_date", "ship_date"]},
        "check": {"type": "row_condition", "expression": "ship_date >= order_date"},
        "severity": "high", "rationale": "30 rows ship before they are ordered"}
BAD_COLUMN = {**GOOD, "name": "Delivery after ship",
              "check": {"type": "row_condition", "expression": "delivered_on >= ship_date"}}
FIXED = {**GOOD, "name": "Delivery after ship", "target": {"entity": "order", "columns": ["ship_date", "delivery_date"]},
         "check": {"type": "row_condition", "expression": "delivery_date >= ship_date"}}
OTHER_ENTITY = {**GOOD, "name": "Wrong target", "target": {"entity": "customer", "columns": ["email"]},
                "scope": "column", "check": {"type": "not_null"}}


@pytest.fixture(scope="module")
def demo_app(tmp_path_factory):
    """App DB with the demo source connected and a heuristic semantic model built (no LLM)."""
    if not DEMO_DB.exists():
        pytest.skip("Demo DB not built")
    url = f"sqlite:///{(tmp_path_factory.mktemp('gen') / 'app.db').as_posix()}"
    upgrade(url)
    engine = make_engine(url)
    sid = pipeline.connect_source(ConnectionConfig(dialect="sqlite", database=str(DEMO_DB)), app_engine=engine)
    pipeline.build_model(sid, app_engine=engine)
    yield engine, sid
    engine.dispose()


def test_baseline_rules_are_valid_and_unique(demo_app):
    engine, sid = demo_app
    with unit_of_work(engine) as repo:
        model, meta = repo.semantic_models.latest(sid), repo.snapshots.latest(sid)
    rules = baseline_rules(model, meta, sid)
    assert len(rules) > 30
    assert all(validate_rule(r, model) == [] for r in rules)
    assert len({r.fingerprint for r in rules}) == len(rules)
    assert {r.origin.value for r in rules} == {"heuristic"}


def test_generation_requires_a_semantic_model(tmp_path):
    url = f"sqlite:///{(tmp_path / 'a.db').as_posix()}"
    upgrade(url)
    with pytest.raises(NoSemanticModelError):
        generate_rules("unknown", app_engine=make_engine(url))


def test_invalid_rules_are_repaired_or_counted(demo_app):
    engine, sid = demo_app
    llm = FakeLLM(llm_rules(GOOD, BAD_COLUMN, OTHER_ENTITY), llm_rules(FIXED))
    result = generate_rules(sid, llm=llm, entities=["order"], include_baseline=False, store=False,
                            app_engine=engine)
    gen = result.per_entity["order"]
    assert {r.name for r in gen.valid} == {"Ship date on/after order date", "Delivery after ship"}
    assert gen.repaired == 1 and gen.off_target == 1 and result.invalid == 0
    assert "delivered_on" in llm.prompts[1][-1]["content"]          # the exact error went back to the LLM
    assert all(r.origin.value == "llm" and r.model == "fake" for r in gen.valid)


def test_store_dedup_and_run_record(demo_app):
    engine, sid = demo_app
    first = generate_rules(sid, llm=FakeLLM(llm_rules(GOOD)), entities=["order"], app_engine=engine)
    assert len(first.stored) > 5 and first.run_id
    # second run: baseline + the same LLM rule again -> everything is a duplicate
    again = generate_rules(sid, llm=FakeLLM(llm_rules({**GOOD, "name": "Reworded", "severity": "low"})),
                           entities=["order"], app_engine=engine)
    assert again.new_rules == [] and len(again.duplicates) == len(first.new_rules)
    with unit_of_work(engine) as repo:
        runs = repo.runs.list(sid)
        assert [r.status for r in runs] == ["succeeded", "succeeded"]
        assert runs[0].dropped_duplicate == len(first.new_rules) and runs[1].generated == len(first.new_rules)


def test_pending_rules_are_in_the_next_prompt(demo_app):
    engine, sid = demo_app
    llm = FakeLLM(llm_rules())
    generate_rules(sid, llm=llm, entities=["order"], include_baseline=False, store=False, app_engine=engine)
    prompt = llm.prompts[0][1]["content"]
    assert "Ship date on/after order date [row_condition on order(order_date,ship_date)" in prompt


def test_rejections_and_reasons_reach_the_prompt(demo_app):
    engine, sid = demo_app
    with unit_of_work(engine) as repo:
        rule = next(r for r in repo.rules.list(sid) if r.name == "Ship date on/after order date")
        repo.rules.record_review(rule.rule_id, "rejected", "alice", reason="ship_date is set by the carrier later")
    llm = FakeLLM(llm_rules())
    generate_rules(sid, llm=llm, entities=["order"], include_baseline=False, store=False, app_engine=engine)
    prompt = llm.prompts[0][1]["content"]
    rejected = prompt.split("REJECTED")[1].split("STEWARD PREFERENCES")[0]
    assert "Ship date on/after order date" in rejected and "set by the carrier later" in rejected


def test_failed_llm_marks_run_failed(demo_app):
    engine, sid = demo_app

    class Boom(FakeLLM):
        def chat(self, messages):
            raise RuntimeError("network down")

    with pytest.raises(RuntimeError):
        generate_rules(sid, llm=Boom(), entities=["order"], app_engine=engine)
    with unit_of_work(engine) as repo:
        latest = repo.runs.list(sid)[0]
        assert latest.status == "failed" and "network down" in latest.error


def test_coverage_matcher(demo_app):
    engine, sid = demo_app
    with unit_of_work(engine) as repo:
        model = repo.semantic_models.latest(sid)
    from app.schemas.rule import Rule

    def rule(**kw):
        return Rule.model_validate({"source_id": sid, "name": "x", "category": "data_quality",
                                    "dimension": "validity", "scope": "column", "origin": "llm", **kw})

    qty = {"entity": "order_item", "columns": ["quantity"]}
    strict = rule(target=qty, check={"type": "range", "min": 1, "max": 100})
    loose = rule(target=qty, check={"type": "range", "min": 0, "max": 10000})
    covered = lambda r: {i["issue_id"] for i in evaluate([r], model)["issues"] if i["covered"]}
    assert {"PI-039", "PI-040"} <= covered(strict)
    assert not {"PI-039", "PI-040"} & covered(loose)           # 0..10000 would not flag 0 or 5000
