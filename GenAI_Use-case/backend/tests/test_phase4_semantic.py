import json
import sqlite3
from datetime import date

import pytest

from app.agent.llm_client import LLMClient, LLMError, OpenAICompatibleClient
from app.config import Settings
from app.connectivity.connectors import ConnectionConfig, create_source_engine
from app.extraction.profiler import profile_source
from app.extraction.schema_crawler import crawl_schema
from app.schemas.metadata import ValueFrequency
from app.schemas.semantic_model import EntityType, SemanticType as ST
from app.semantic.builder import build_semantic_model
from app.semantic.enrichment import EnrichmentResponse
from app.semantic.heuristics import classify_values, signature_to_regex

AS_OF = date(2026, 1, 1)


class FakeLLM(LLMClient):
    """Returns canned responses in order; records the prompts it was given."""
    model_name = "fake"

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.prompts: list[list[dict]] = []

    def chat(self, messages):
        self.prompts.append(list(messages))
        return self.responses.pop(0)


def vf(*pairs) -> list[ValueFrequency]:
    total = sum(n for _, n in pairs)
    return [ValueFrequency(value=v, count=n, share=n / total) for v, n in pairs]


# --------------------------------------------------------------------------- value classification
def test_classify_status_values():
    valid, anomalies = classify_values(vf(("DELIVERED", 20901), ("CANCELLED", 1622), ("RETURNED", 1216),
                                          ("SHIPPED", 131), ("PLACED", 100), ("Dlvrd", 13), ("unknown", 10),
                                          ("delivered", 9), ("SHIPED", 8)))
    assert valid == ["DELIVERED", "CANCELLED", "RETURNED", "SHIPPED", "PLACED"]
    found = {a.value: a.looks_like for a in anomalies}
    assert found == {"Dlvrd": "DELIVERED", "unknown": None, "delivered": "DELIVERED", "SHIPED": "SHIPPED"}


def test_classify_short_codes_and_aliases():
    valid, anomalies = classify_values(vf(("F", 1253), ("M", 1188), ("O", 46), ("Male", 10), ("X", 6), ("female", 4)))
    assert valid == ["F", "M", "O"]
    assert {a.value: a.looks_like for a in anomalies} == {"Male": "M", "X": None, "female": "F"}
    valid, anomalies = classify_values(vf(("ONLINE", 900), ("MOBILE_APP", 700), ("STORE", 400), ("app", 5)))
    assert anomalies[0].looks_like == "MOBILE_APP"


def test_similar_but_frequent_codes_stay_valid():
    valid, anomalies = classify_values(vf(("CARD", 500), ("CASH", 300), ("UPI", 200)))
    assert valid == ["CARD", "CASH", "UPI"] and anomalies == []


@pytest.mark.parametrize("sig, regex", [
    ("AAA-99999999-999999", r"^[A-Z]{3}\-\d{8}\-\d{6}$"),
    ("+999999999999", r"^\+\d{12}$"),
    ("AAA-999999", r"^[A-Z]{3}\-\d{6}$"),
])
def test_signature_to_regex(sig, regex):
    assert signature_to_regex(sig) == regex


# --------------------------------------------------------------------------- demo build (no LLM)
@pytest.fixture(scope="module")
def demo(request):
    from tests.conftest import DEMO_DB
    if not DEMO_DB.exists():
        pytest.skip("Demo DB not built")
    cfg = ConnectionConfig(dialect="sqlite", database=str(DEMO_DB))
    engine = create_source_engine(cfg)
    meta = profile_source(engine, crawl_schema(engine, cfg), sample_limit=10_000, as_of=AS_OF)
    return engine, meta


@pytest.fixture(scope="module")
def heuristic_model(demo):
    engine, meta = demo
    model, report = build_semantic_model(engine, meta)
    return model, report


def test_heuristic_types(heuristic_model):
    model, _ = heuristic_model
    c, o = model.entity("customer"), model.entity("order")
    assert c.attribute("email").semantic_type == ST.EMAIL and c.attribute("email").pii
    assert c.attribute("phone").expected_pattern == r"^\+\d{12}$"
    assert c.attribute("date_of_birth").semantic_type == ST.BIRTH_DATE
    assert o.attribute("status").semantic_type == ST.STATUS_CODE
    assert o.attribute("order_number").semantic_type == ST.BUSINESS_KEY
    assert model.entity("product").attribute("cost_price").semantic_type == ST.COST
    assert set(o.attribute("status").allowed_values.values) == {"PLACED", "SHIPPED", "DELIVERED", "CANCELLED",
                                                                 "RETURNED"}


def test_entity_types(heuristic_model):
    model, _ = heuristic_model
    types = {e.entity_id: e.entity_type for e in model.entities}
    assert types == {"region": EntityType.HIERARCHY, "category": EntityType.HIERARCHY,
                     "product": EntityType.MASTER, "customer": EntityType.MASTER,
                     "order": EntityType.TRANSACTION, "order_item": EntityType.TRANSACTION_LINE}


def test_hierarchies_and_their_violations(heuristic_model):
    model, _ = heuristic_model
    geo = model.hierarchy("region_hierarchy")
    assert geo.levels == ["ZONE", "STATE", "CITY"]
    assert [(a.entity, a.must_attach_at) for a in geo.attached_entities] == [("customer", "CITY")]
    evidence = " | ".join(geo.evidence)
    assert "1 CITY row(s) have a ZONE parent" in evidence            # PI-001 Gurugram
    assert "1 CITY row(s) point to a missing parent" in evidence     # PI-002 Nagpur
    assert "10 customers row(s) point to a STATE" in evidence        # PI-021
    cat = model.hierarchy("category_hierarchy")
    assert cat.levels == ["DEPARTMENT", "CATEGORY", "SUBCATEGORY"]
    assert any("CATEGORY row(s) have a CATEGORY parent" in e for e in cat.evidence)   # PI-006 cycle


# --------------------------------------------------------------------------- inferred relationships
def test_relationships_inferred_without_declared_fks(tmp_path):
    path = tmp_path / "nofk.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, cust_id INTEGER, amount DECIMAL(10,2));
        CREATE TABLE notes (note_id INTEGER PRIMARY KEY, customer_id INTEGER);
    """)
    con.executemany("INSERT INTO customers VALUES (?, ?)", [(i, f"c{i}") for i in range(1, 51)])
    con.executemany("INSERT INTO orders VALUES (?, ?, ?)", [(i, (i % 50) + 1, 10.0) for i in range(1, 201)])
    # notes.customer_id mostly does NOT match -> must not be inferred
    con.executemany("INSERT INTO notes VALUES (?, ?)", [(i, 1000 + i) for i in range(1, 21)])
    con.commit()
    con.close()
    cfg = ConnectionConfig(dialect="sqlite", database=str(path))
    engine = create_source_engine(cfg)
    meta = profile_source(engine, crawl_schema(engine, cfg), sample_limit=1000, as_of=AS_OF)
    model, report = build_semantic_model(engine, meta)
    rels = {(r.from_entity, r.from_attributes[0], r.to_entity, r.kind) for r in model.relationships}
    assert rels == {("order", "cust_id", "customer", "inferred")}
    assert model.entity("order").attribute("cust_id").semantic_type == ST.FOREIGN_KEY


# --------------------------------------------------------------------------- LLM enrichment & merge
def _enrichment(**brand_attr) -> str:
    return EnrichmentResponse.model_validate({
        "domain": {"name": "Retail", "description": "An Indian omni-channel retailer."},
        "entities": [
            {"entity_id": "product", "name": "Product", "description": "Sellable item.", "grain": "One row per SKU",
             "attributes": [
                 {"column": "brand", "name": "Brand", **brand_attr},
                 {"column": "sku", "name": "SKU", "semantic_type": "label"},       # high confidence: ignored
             ]},
            {"entity_id": "customer", "attributes": [{"column": "gender", "pii": True}],
             "value_verdicts": [{"column": "gender", "value": "X", "verdict": "valid"}]},
            {"entity_id": "ghost", "name": "Hallucinated entity"},
        ],
    }).model_dump_json()


def test_merge_policy(demo):
    engine, meta = demo
    llm = FakeLLM(_enrichment(semantic_type="label", reason="brand names are labels"))
    model, report = build_semantic_model(engine, meta, llm=llm)
    product = model.entity("product")
    assert product.grain == "One row per SKU" and model.domain.description
    assert product.attribute("brand").semantic_type == ST.LABEL                # low confidence: changed
    assert product.attribute("brand").provenance.source == "llm"
    assert product.attribute("sku").semantic_type == ST.BUSINESS_KEY            # high confidence: kept
    assert "X" in model.entity("customer").attribute("gender").allowed_values.values   # verdict applied
    assert any("ghost" in c for c in report.llm_changes)
    prompt = llm.prompts[0][1]["content"]
    assert "[REVIEW]" in prompt
    with engine.connect() as conn:                                              # no personal data sent
        rows = conn.exec_driver_sql("SELECT email, phone FROM customers "
                                    "WHERE email IS NOT NULL AND phone IS NOT NULL").fetchall()
    assert not [v for row in rows for v in row if v in prompt]
    first_name_line = next(ln for ln in prompt.splitlines() if ln.startswith("- first_name"))
    assert "e.g." in first_name_line and "*" in first_name_line.split("e.g.")[1]   # masked examples only


def test_invalid_llm_json_is_retried_then_falls_back(demo):
    engine, meta = demo
    llm = FakeLLM("not json", _enrichment())
    model, report = build_semantic_model(engine, meta, llm=llm)
    assert len(llm.prompts) == 2 and "did not match" in llm.prompts[1][-1]["content"]
    assert model.entity("product").grain == "One row per SKU"

    llm = FakeLLM("{}", "{}")
    model, report = build_semantic_model(engine, meta, llm=llm)
    assert any("skipped" in c for c in report.llm_changes)
    assert model.entity("product").grain == "One row per product"            # heuristic result kept


def test_steward_locks_survive_rebuild_and_drift_is_reported(demo, heuristic_model):
    engine, meta = demo
    previous, _ = heuristic_model
    previous = previous.model_copy(deep=True)
    brand = previous.entity("product").attribute("brand")
    brand.semantic_type, brand.provenance.locked, brand.provenance.source = ST.LABEL, True, "steward"
    previous.entity("product").attributes.append(brand.model_copy(update={"column": "colour"}))

    llm = FakeLLM(_enrichment(semantic_type="organization_name"))
    model, report = build_semantic_model(engine, meta, llm=llm, previous=previous)
    assert model.version == previous.version + 1
    assert model.entity("product").attribute("brand").semantic_type == ST.LABEL
    assert "product.brand" in report.kept_locked
    assert "column removed: products.colour" in report.drift


def test_rebuild_without_llm_keeps_earlier_enrichment(demo):
    engine, meta = demo
    enriched, _ = build_semantic_model(engine, meta, llm=FakeLLM(_enrichment(semantic_type="label",
                                                                               description="Maker of the product")))
    rebuilt, _ = build_semantic_model(engine, meta, llm=None, previous=enriched)
    brand = rebuilt.entity("product").attribute("brand")
    assert (brand.semantic_type, brand.description) == (ST.LABEL, "Maker of the product")
    assert rebuilt.entity("product").grain == "One row per SKU"
    assert rebuilt.model_dump(exclude={"version", "created_at"}) == enriched.model_dump(
        exclude={"version", "created_at"})


# --------------------------------------------------------------------------- client cache
def test_llm_client_caches_responses(tmp_path, monkeypatch):
    client = OpenAICompatibleClient(Settings(llm_api_key="test-key"), cache_dir=tmp_path)
    calls = []

    class _Resp:
        usage = None
        choices = [type("C", (), {"message": type("M", (), {"content": '{"ok": true}'})()})()]

    monkeypatch.setattr(client._client.chat.completions, "create", lambda **kw: calls.append(kw) or _Resp())
    msgs = [{"role": "user", "content": "hi json"}]
    assert client.chat(msgs) == client.chat(msgs) == '{"ok": true}'
    assert len(calls) == 1 and client.usage["cached"] == 1


def test_missing_api_key_fails_clearly():
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        OpenAICompatibleClient(Settings(llm_api_key=""))
