"""Shared fixtures: in-memory app DB, demo source DB, fake LLM client."""
import sqlite3

import pytest

from app.config import PROJECT_ROOT
from app.connectivity.connectors import ConnectionConfig

DEMO_DB = PROJECT_ROOT / "data" / "source.db"


@pytest.fixture
def tiny_db(tmp_path):
    """Small SQLite source with a PK, a FK, a self-referencing FK and a UNIQUE column."""
    path = tmp_path / "tiny.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE dept (
            dept_id   INTEGER PRIMARY KEY,
            dept_code VARCHAR(10) NOT NULL UNIQUE,
            parent_id INTEGER REFERENCES dept(dept_id)
        );
        CREATE TABLE employee (
            emp_id    INTEGER PRIMARY KEY,
            dept_id   INTEGER REFERENCES dept(dept_id),
            salary    DECIMAL(10,2),
            hired_on  DATE,
            is_active BOOLEAN
        );
        INSERT INTO dept VALUES (1, 'HQ', NULL), (2, 'SALES', 1);
        INSERT INTO employee VALUES (1, 2, 50000.00, '2024-01-15', 1);
    """)
    con.close()
    return ConnectionConfig(dialect="sqlite", database=str(path))


@pytest.fixture
def demo_db():
    if not DEMO_DB.exists():
        pytest.skip("Demo DB not built - run samples/source_db/seed_demo_db.py")
    return ConnectionConfig(dialect="sqlite", database=str(DEMO_DB))
