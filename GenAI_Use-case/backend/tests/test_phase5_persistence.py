import json

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from app.config import PROJECT_ROOT
from app.connectivity.connectors import ConnectionConfig
from app.connectivity.credentials import CredentialError, decrypt
from app.persistence.database import make_engine, upgrade
from app.persistence.models import RuleReviewRow, SourceRow
from app.persistence.repositories import DuplicateRuleError, structure_hash, unit_of_work
from app.schemas.metadata import ColumnMetadata, GenericType, SourceMetadata, TableMetadata
from app.schemas.rule import Rule, RuleStatus
from app.schemas.semantic_model import SemanticModel
from app.semantic.storage import latest_model, save_model

SRC = "demo_retail"
REF_MODEL = PROJECT_ROOT / "samples" / "semantic_models" / "demo_retail.reference.json"
REF_RULES = PROJECT_ROOT / "samples" / "rules" / "demo_retail.reference_rules.json"


@pytest.fixture
def engine(tmp_path):
    """A brand-new app DB built by the real Alembic migrations, plus one known source."""
    url = f"sqlite:///{(tmp_path / 'app.db').as_posix()}"
    upgrade(url)
    eng = make_engine(url)
    with unit_of_work(eng) as repo:
        repo.session.add(SourceRow(id=SRC, display_name="demo", dialect="sqlite", database="demo.db"))
    yield eng
    eng.dispose()


def make_rule(**overrides) -> Rule:
    base = dict(source_id=SRC, name="Quantity in range", category="data_quality", dimension="validity",
                scope="column", target={"entity": "order_item", "columns": ["quantity"]},
                check={"type": "range", "min": 0, "max": 120}, origin="llm")
    base.update(overrides)
    return Rule.model_validate(base)


# --------------------------------------------------------------------------- schema & integrity
def test_migrations_create_all_tables(engine):
    assert set(inspect(engine).get_table_names()) == {
        "alembic_version", "sources", "metadata_snapshots", "semantic_model_versions", "context_items",
        "generation_runs", "rules", "rule_reviews"}


def test_foreign_keys_are_enforced(engine):
    with pytest.raises(IntegrityError):
        with unit_of_work(engine) as repo:
            repo.session.add(RuleReviewRow(rule_id="no-such-rule", decision="approved", reviewer="x"))


# --------------------------------------------------------------------------- sources & credentials
def test_password_is_encrypted_and_recoverable(engine):
    cfg = ConnectionConfig(dialect="postgresql", host="db.corp", port=5432, database="sales",
                           username="alice", password="s3cret!")
    with unit_of_work(engine) as repo:
        row = repo.sources.upsert(cfg)
        sid = row.id
        assert "s3cret" not in row.password_encrypted
        assert repo.sources.connection_config(sid).password.get_secret_value() == "s3cret!"
    with pytest.raises(CredentialError, match="can't be decrypted"):
        decrypt(row.password_encrypted, key=Fernet.generate_key().decode())


def test_password_not_stored_when_declined(engine):
    cfg = ConnectionConfig(dialect="postgresql", host="h", database="d", username="u", password="pw")
    with unit_of_work(engine) as repo:
        row = repo.sources.upsert(cfg, remember_password=False)
        assert row.password_encrypted is None
        assert repo.sources.connection_config(row.id, password="pw").password.get_secret_value() == "pw"


def test_return_visit_reuses_source_and_shows_approved_rules(engine):
    cfg = ConnectionConfig(dialect="postgresql", host="db.corp", database="sales", username="alice")
    with unit_of_work(engine) as repo:
        sid = repo.sources.upsert(cfg).id
        repo.rules.add(make_rule(source_id=sid, status="approved"), created_by="alice")
    cfg_bob = ConnectionConfig(dialect="postgresql", host="DB.corp", database="Sales", username="bob")
    with unit_of_work(engine) as repo:
        assert repo.sources.upsert(cfg_bob).id == sid
        assert len([s for s in repo.sources.list() if s.id == sid]) == 1
        assert [r.name for r in repo.rules.approved(sid)] == ["Quantity in range"]


# --------------------------------------------------------------------------- snapshots & models
def _meta(columns: list[str]) -> SourceMetadata:
    cols = [ColumnMetadata(name=c, ordinal_position=i, native_type="INTEGER", generic_type=GenericType.INTEGER,
                           nullable=True) for i, c in enumerate(columns, 1)]
    return SourceMetadata(source_id=SRC, dialect="sqlite", database="demo", extracted_at="2026-01-01T00:00:00Z",
                          tables=[TableMetadata(name="t", columns=cols)])


def test_structure_hash_detects_schema_change_only(engine):
    a, b, c = _meta(["x", "y"]), _meta(["x", "y"]), _meta(["x", "y", "z"])
    b.tables[0].row_count = 999   # data changes don't matter
    assert structure_hash(a) == structure_hash(b) != structure_hash(c)
    with unit_of_work(engine) as repo:
        repo.snapshots.save(a)
        assert repo.snapshots.latest(SRC, profiled_only=False).tables[0].name == "t"


def test_semantic_model_versions_are_immutable(engine):
    model = SemanticModel.model_validate_json(REF_MODEL.read_text(encoding="utf-8"))
    v1 = save_model(model, engine=engine)
    edited = v1.model_copy(deep=True)
    edited.entity("product").attribute("brand").name = "Brand (edited)"
    v2 = save_model(edited, created_by="alice", note="renamed brand", engine=engine)
    assert (v1.version, v2.version) == (1, 2)
    with unit_of_work(engine) as repo:
        assert repo.semantic_models.get(SRC, 1).entity("product").attribute("brand").name == "Brand"
        assert [(h.version, h.created_by) for h in repo.semantic_models.history(SRC)] == [(1, "engine"),
                                                                                         (2, "alice")]
    assert latest_model(SRC, engine=engine).entity("product").attribute("brand").name == "Brand (edited)"


# --------------------------------------------------------------------------- rules
def test_duplicates_and_similar_rules(engine):
    with unit_of_work(engine) as repo:
        first = repo.rules.add(make_rule())
        reworded = repo.rules.add(make_rule(name="Qty must be sensible", severity="low"))
        stricter = repo.rules.add(make_rule(check={"type": "range", "min": 1, "max": 100}))
    assert first.outcome == "stored"
    assert reworded.outcome == "duplicate" and reworded.existing.rule_id == first.rule.rule_id
    assert stricter.outcome == "similar" and stricter.existing.rule_id == first.rule.rule_id


def test_modified_review_keeps_audit_and_blocks_original(engine):
    with unit_of_work(engine) as repo:
        rule = repo.rules.add(make_rule()).rule
        edited = make_rule(check={"type": "range", "min": 1, "max": 100}, name="Quantity 1-100")
        new = repo.rules.record_review(rule.rule_id, "modified", reviewer="alice", reason="0 is not a valid qty",
                                       edited=edited)
        assert new.rule_id == rule.rule_id and new.status == RuleStatus.APPROVED
        assert new.fingerprint != rule.fingerprint
        review = repo.rules.reviews(rule.rule_id)[-1]
        assert review.before["check"]["max"] == 120 and review.after["check"]["max"] == 100
        # the LLM re-suggesting the ORIGINAL 0-120 form is still caught as a duplicate
        assert repo.rules.add(make_rule()).outcome == "duplicate"


def test_edit_into_an_existing_rule_is_refused(engine):
    with unit_of_work(engine) as repo:
        a = repo.rules.add(make_rule()).rule
        repo.rules.add(make_rule(check={"type": "range", "min": 1, "max": 100}))
        with pytest.raises(DuplicateRuleError):
            repo.rules.record_review(a.rule_id, "modified", "alice",
                                     edited=make_rule(check={"type": "range", "min": 1, "max": 100}))


def test_history_for_prompt(engine):
    with unit_of_work(engine) as repo:
        ok = repo.rules.add(make_rule(target={"entity": "order", "columns": ["status"]},
                                      check={"type": "not_null"}, name="Status populated")).rule
        bad = repo.rules.add(make_rule(target={"entity": "order", "columns": ["channel"]},
                                       check={"type": "not_null"}, name="Channel populated")).rule
        edit = repo.rules.add(make_rule()).rule
        repo.rules.add(make_rule(target={"entity": "order", "columns": ["currency"]},
                                 check={"type": "not_null"}, name="Still proposed"))
        repo.rules.record_review(ok.rule_id, "approved", "alice")
        repo.rules.record_review(bad.rule_id, "rejected", "alice", reason="channel is optional for B2B")
        repo.rules.record_review(edit.rule_id, "modified", "alice",
                                 edited=make_rule(check={"type": "range", "min": 1, "max": 100}))
        h = repo.rules.history_for_prompt(SRC)
    assert {r.name for r in h.approved} == {"Status populated", "Quantity in range"}
    assert [(r.name, reason) for r, reason in h.rejected] == [("Channel populated", "channel is optional for B2B")]
    assert [(orig.check.max, new.check.max) for orig, new in h.superseded] == [(120, 100)]


def test_reference_rules_round_trip(engine):
    items = json.loads(REF_RULES.read_text(encoding="utf-8"))["rules"]
    with unit_of_work(engine) as repo:
        stored = [repo.rules.add(Rule.model_validate(i["rule"]), created_by="import") for i in items]
    assert {s.outcome for s in stored} == {"stored"}   # 44 distinct rules: no duplicates, no near-matches
    with unit_of_work(engine) as repo:
        loaded = {r.rule_id: r for r in repo.rules.list(SRC)}
        assert repo.rules.counts(SRC) == {"approved": 44}
        for s in stored:
            assert loaded[s.rule.rule_id].model_dump() == s.rule.model_dump()


# --------------------------------------------------------------------------- context & runs
def test_context_items(engine):
    with unit_of_work(engine) as repo:
        a = repo.context.add(SRC, "  COD not offered in stores ", "alice", entity="order")
        repo.context.add(SRC, "Ships within 3 days", "alice")
        repo.context.retire(a.id)
        assert [c.text for c in repo.context.active(SRC)] == ["Ships within 3 days"]


def test_generation_run_lifecycle(engine):
    with unit_of_work(engine) as repo:
        run = repo.runs.start(SRC, model="fake", semantic_model_version=1, prompt="...")
        repo.runs.finish(run.id, "succeeded", generated=12, dropped_duplicate=3, prompt_tokens=900)
        assert [(r.status, r.generated, r.dropped_duplicate) for r in repo.runs.list(SRC)] == [("succeeded", 12, 3)]
