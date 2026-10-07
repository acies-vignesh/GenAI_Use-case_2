"""App DB engine + session factory (SQLite for dev, Postgres later).

This is the engine's OWN database (sources, models, rules, reviews) - not the source being analysed.
Unlike the source, here we want SQLite to enforce foreign keys, and WAL mode so readers (API/UI)
don't block on a writer.

Schema changes go through Alembic migrations (app/persistence/migrations): `upgrade()` brings any
database - new or existing - to the latest structure without losing data.
"""
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.config import settings

BACKEND_DIR = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_engine: Engine | None = None


def make_engine(url: str) -> Engine:
    engine = create_engine(url)
    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys = ON")
            cur.execute("PRAGMA journal_mode = WAL")
            cur.close()
    return engine


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = make_engine(settings.app_db_url)
    return _engine


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """One unit of work: commit if the block succeeds, roll back if it raises."""
    session = Session(engine or get_engine(), expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def upgrade(url: str | None = None, revision: str = "head") -> None:
    """Apply pending migrations up to `revision` (default: the latest)."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url or settings.app_db_url)
    command.upgrade(cfg, revision)
