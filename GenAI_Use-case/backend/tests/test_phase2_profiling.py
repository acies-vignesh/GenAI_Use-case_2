import sqlite3
from datetime import date

import pytest

from app.connectivity.connectors import ConnectionConfig, create_source_engine
from app.extraction.profiler import mask, profile_source, signature
from app.extraction.schema_crawler import crawl_schema

AS_OF = date(2026, 1, 1)


# --------------------------------------------------------------------------- pure helpers
@pytest.mark.parametrize("value, exact, collapsed", [
    ("ORD-20240101-000001", "AAA-99999999-999999", "A-9-9"),
    ("+919876543210", "+999999999999", "+9"),
    ("ravi.sharma12@gmail.com", "aaaa.aaaaaa99@aaaaa.aaa", "a.a9@a.a"),
    ("hsr1", "aaa9", "a9"),
    ("D'Souza", "A'Aaaaa", "A'Aa"),
])
def test_signature(value, exact, collapsed):
    assert signature(value) == exact
    assert signature(value, collapse=True) == collapsed


def test_mask_hides_personal_data():
    assert mask("ravi.sharma12@gmail.com") == "r***@g***.com"
    assert mask("+919876543210") == "+9*********10"
    assert mask("Ravi") == "****"


# --------------------------------------------------------------------------- crafted source
@pytest.fixture
def profiled(tmp_path):
    path = tmp_path / "p.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE parent (id INTEGER PRIMARY KEY, code VARCHAR(5));
        CREATE TABLE child (
            id        INTEGER PRIMARY KEY,
            parent_id INTEGER REFERENCES parent(id),
            status    VARCHAR(10),
            phone     VARCHAR(20),
            amount    DECIMAL(10,2),
            event_on  DATE,
            flag      BOOLEAN
        );
        INSERT INTO parent VALUES (1, 'AAA'), (2, 'BBB');
    """)
    rows = []
    for i in range(1, 101):
        rows.append((
            i,
            99 if i <= 3 else 1,                                   # 3 orphans
            "OPEN" if i <= 60 else "CLOSED" if i <= 98 else "opn",  # 2 rare invalid codes
            None if i <= 5 else "12345678" if i <= 7 else f"+9198765{i:05d}",
            -10.0 if i == 1 else 0.0 if i == 2 else float(i),
            "1900-01-01" if i == 1 else "2030-05-05" if i == 2 else "not a date" if i == 3 else "2025-03-01",
            1 if i % 2 else 0,
        ))
    con.executemany("INSERT INTO child VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()

    cfg = ConnectionConfig(dialect="sqlite", database=str(path))
    engine = create_source_engine(cfg)
    return profile_source(engine, crawl_schema(engine, cfg), sample_limit=50, as_of=AS_OF)


def col(meta, table, name):
    return meta.table(table).column(name).profile


def test_sampling_full_vs_random(profiled):
    assert (profiled.table("parent").sample_method, profiled.table("parent").sample_size) == ("full", 2)
    assert (profiled.table("child").sample_method, profiled.table("child").sample_size) == ("random", 50)


def test_exact_counts_ignore_sampling(profiled):
    p = col(profiled, "child", "phone")
    assert (p.row_count, p.null_count, p.null_rate) == (100, 5, 0.05)


def test_top_values_reveal_rare_codes(profiled):
    values = {v.value: v.count for v in col(profiled, "child", "status").top_values}
    assert values == {"OPEN": 60, "CLOSED": 38, "opn": 2}


def test_sensitive_columns_are_masked_and_patterned(profiled):
    p = col(profiled, "child", "phone")
    assert p.sensitive_hint and p.top_values is None
    assert p.detected_format.format == "phone"
    assert p.patterns[0].pattern == "+999999999999"   # the 2 rare 8-digit phones may miss a 50-row sample
    assert all("*" in e for e in p.examples)


def test_numeric_and_date_stats(profiled):
    n = col(profiled, "child", "amount").numeric
    assert (n.min, n.max, n.negative_count, n.zero_count) == (-10.0, 100.0, 1, 1)
    d = col(profiled, "child", "event_on").date
    assert (d.min, d.future_count, d.placeholder_count) == ("1900-01-01", 1, 1)
    assert d.max == "not a date"     # text min/max: exactly why unparseable values must be counted
    assert col(profiled, "child", "flag").boolean.true_count == 50


def test_fk_orphans(profiled):
    fk = profiled.table("child").foreign_keys[0]
    assert (fk.orphan_count, fk.orphan_rate) == (3, 0.03)


# --------------------------------------------------------------------------- demo database
def test_demo_profile_matches_planted_issues(demo_db):
    engine = create_source_engine(demo_db)
    meta = profile_source(engine, crawl_schema(engine, demo_db), sample_limit=10_000, as_of=AS_OF)

    assert col(meta, "customers", "email").null_count == 75
    assert col(meta, "customers", "date_of_birth").date.placeholder_count == 6
    assert col(meta, "orders", "order_date").date.future_count == 5
    assert {"SHIPED", "unknown", "delivered", "Dlvrd"} <= {v.value for v in col(meta, "orders", "status").top_values}
    assert col(meta, "order_items", "quantity").numeric.max == 5000
    orphans = {(t.name, fk.columns[0]): fk.orphan_count for t in meta.tables for fk in t.foreign_keys}
    assert orphans == {
        ("categories", "parent_category_id"): 0, ("customers", "region_id"): 10,
        ("order_items", "product_id"): 15, ("order_items", "order_id"): 15,
        ("orders", "customer_id"): 20, ("products", "category_id"): 3, ("regions", "parent_region_id"): 1,
    }
