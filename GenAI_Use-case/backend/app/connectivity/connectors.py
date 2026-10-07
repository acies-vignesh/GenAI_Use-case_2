"""Build SQLAlchemy engines for supported source DBs (Postgres, MySQL, SQL Server, SQLite...). Read-only usage.

SQLAlchemy gives one API over many databases: the rest of the code talks to an `Engine` and never
cares which database is behind it. Only this module knows about dialects and drivers.

Read-only:
- SQLite is opened with `mode=ro`, so any write fails at the driver level.
- For server databases, read-only must be enforced by connecting as a read-only DB user
  (a privilege the DB enforces - safer than anything the app could promise).
"""
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr
from sqlalchemy import URL, create_engine, make_url, text
from sqlalchemy.engine import Engine

Dialect = Literal["sqlite", "postgresql", "mysql", "mssql", "oracle"]

# SQLAlchemy driver per dialect. Server drivers are installed when that DB is first needed
# (e.g. `pip install psycopg2-binary` for Postgres).
DRIVERS: dict[str, str] = {
    "postgresql": "postgresql+psycopg2",
    "mysql": "mysql+pymysql",
    "mssql": "mssql+pyodbc",
    "oracle": "oracle+oracledb",
}


class ConnectionConfig(BaseModel):
    """What the user enters in the connection form. `database` is a file path for SQLite."""
    dialect: Dialect
    database: str
    host: str | None = None
    port: int | None = None
    username: str | None = None
    password: SecretStr | None = None     # SecretStr prints as '**********' - never leaks into logs/JSON
    schema_name: str | None = None

    @classmethod
    def from_url(cls, url: str, schema_name: str | None = None) -> "ConnectionConfig":
        u = make_url(url)
        return cls(
            dialect=u.get_backend_name(),
            database=u.database or "",
            host=u.host,
            port=u.port,
            username=u.username,
            password=u.password,
            schema_name=schema_name,
        )

    @property
    def display_name(self) -> str:
        """Human-friendly name without credentials."""
        if self.dialect == "sqlite":
            return Path(self.database).stem
        return self.database


def create_source_engine(config: ConnectionConfig) -> Engine:
    if config.dialect == "sqlite":
        path = Path(config.database).resolve()
        if not path.exists():
            raise FileNotFoundError(f"SQLite database not found: {path}")
        uri = f"{path.as_uri()}?mode=ro"
        return create_engine(
            "sqlite://",
            creator=lambda: sqlite3.connect(uri, uri=True, check_same_thread=False),
        )

    url = URL.create(
        DRIVERS[config.dialect],
        username=config.username,
        password=config.password.get_secret_value() if config.password else None,
        host=config.host,
        port=config.port,
        database=config.database,
    )
    return create_engine(url, pool_pre_ping=True, pool_size=2, max_overflow=0)


def check_connection(engine: Engine) -> None:
    """Raise ConnectionError with a readable message if the source can't be reached."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # driver errors vary per database
        raise ConnectionError(f"Could not connect to source: {exc}") from exc
