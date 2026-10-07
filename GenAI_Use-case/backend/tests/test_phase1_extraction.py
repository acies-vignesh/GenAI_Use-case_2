import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.connectivity.connectors import ConnectionConfig, create_source_engine, check_connection
from app.connectivity.fingerprint import source_fingerprint
from app.extraction.schema_crawler import crawl_schema
from app.schemas.metadata import GenericType, SourceMetadata


def test_sqlite_source_is_read_only(tiny_db):
    engine = create_source_engine(tiny_db)
    with pytest.raises(OperationalError, match="readonly"):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM employee"))


def test_missing_sqlite_file_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_source_engine(ConnectionConfig(dialect="sqlite", database=str(tmp_path / "nope.db")))


def test_crawl_extracts_keys_types_and_counts(tiny_db):
    engine = create_source_engine(tiny_db)
    check_connection(engine)
    meta = crawl_schema(engine, tiny_db)

    assert [t.name for t in meta.tables] == ["dept", "employee"]

    dept = meta.table("dept")
    assert dept.primary_key == ["dept_id"]
    assert dept.row_count == 2
    assert dept.column("dept_id").nullable is False
    assert [uc.columns for uc in dept.unique_constraints] == [["dept_code"]]   # inline UNIQUE is detected
    self_fk = dept.foreign_keys[0]
    assert (self_fk.columns, self_fk.referred_table, self_fk.is_self_referencing) == (["parent_id"], "dept", True)

    emp = meta.table("employee")
    fk = emp.foreign_keys[0]
    assert (fk.columns, fk.referred_table, fk.is_self_referencing) == (["dept_id"], "dept", False)
    assert emp.column("salary").generic_type == GenericType.DECIMAL
    assert (emp.column("salary").precision, emp.column("salary").scale) == (10, 2)
    assert emp.column("hired_on").generic_type == GenericType.DATE
    assert emp.column("is_active").generic_type == GenericType.BOOLEAN


def test_metadata_round_trips_through_json(tiny_db):
    meta = crawl_schema(create_source_engine(tiny_db), tiny_db)
    assert SourceMetadata.model_validate_json(meta.model_dump_json()) == meta


def test_fingerprint_is_stable_and_ignores_credentials():
    a = ConnectionConfig(dialect="postgresql", host="DB.corp", port=5432, database="Sales",
                         username="alice", password="x")
    b = ConnectionConfig(dialect="postgresql", host="db.corp", port=5432, database="sales",
                         username="bob", password="y")
    c = ConnectionConfig(dialect="postgresql", host="db.corp", port=5432, database="sales", schema_name="finance")
    assert source_fingerprint(a) == source_fingerprint(b)
    assert source_fingerprint(a) != source_fingerprint(c)


def test_password_never_serialised():
    cfg = ConnectionConfig(dialect="postgresql", host="h", database="d", password="s3cret")
    assert "s3cret" not in cfg.model_dump_json()
    assert "s3cret" not in repr(cfg)


def test_demo_db_structure(demo_db):
    meta = crawl_schema(create_source_engine(demo_db), demo_db)
    assert {t.name for t in meta.tables} == {"regions", "categories", "products", "customers", "orders",
                                             "order_items"}
    assert meta.table("regions").foreign_keys[0].is_self_referencing
    assert meta.table("categories").foreign_keys[0].is_self_referencing
    assert {fk.referred_table for fk in meta.table("order_items").foreign_keys} == {"orders", "products"}
    assert [uc.columns for uc in meta.table("products").unique_constraints] == [["sku"]]
