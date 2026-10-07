"""The service layer = what the UI will do: connect -> refresh metadata -> build model, by source_id."""
import sqlite3

import pytest

from app.connectivity.connectors import ConnectionConfig
from app.persistence.database import make_engine, upgrade
from app.persistence.repositories import unit_of_work
from app.services import pipeline


@pytest.fixture
def app_engine(tmp_path):
    url = f"sqlite:///{(tmp_path / 'app.db').as_posix()}"
    upgrade(url)
    eng = make_engine(url)
    yield eng
    eng.dispose()


@pytest.fixture
def source_config(tmp_path):
    path = tmp_path / "shop.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, email VARCHAR(100), status VARCHAR(10));
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY,
                             customer_id INTEGER REFERENCES customers(customer_id), total_amount DECIMAL(10,2));
    """)
    con.executemany("INSERT INTO customers VALUES (?, ?, ?)",
                    [(i, f"c{i}@mail.com", "ACTIVE" if i % 5 else "CLOSED") for i in range(1, 101)])
    con.executemany("INSERT INTO orders VALUES (?, ?, ?)", [(i, i % 100 + 1, 10.5 * i) for i in range(1, 301)])
    con.commit()
    con.close()
    return ConnectionConfig(dialect="sqlite", database=str(path))


def test_unreachable_source_is_not_stored(app_engine, tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.connect_source(ConnectionConfig(dialect="sqlite", database=str(tmp_path / "missing.db")),
                                app_engine=app_engine)
    with unit_of_work(app_engine) as repo:
        assert repo.sources.list() == []


def test_full_flow_by_source_id(app_engine, source_config):
    source_id = pipeline.connect_source(source_config, app_engine=app_engine)

    metadata, snapshot_id = pipeline.refresh_metadata(source_id, app_engine=app_engine)
    assert {t.name for t in metadata.tables} == {"customers", "orders"} and metadata.profiled_at

    first = pipeline.build_model(source_id, app_engine=app_engine)
    assert first.created and first.model.version == 1 and first.snapshot_id == snapshot_id
    assert first.model.entity("customer").attribute("email").semantic_type.value == "email"


def test_rebuild_without_changes_does_not_add_a_version(app_engine, source_config):
    source_id = pipeline.connect_source(source_config, app_engine=app_engine)
    pipeline.build_model(source_id, app_engine=app_engine)          # profiles on first use
    again = pipeline.build_model(source_id, app_engine=app_engine)
    assert not again.created and again.model.version == 1
    with unit_of_work(app_engine) as repo:
        assert [v.version for v in repo.semantic_models.history(source_id)] == [1]
