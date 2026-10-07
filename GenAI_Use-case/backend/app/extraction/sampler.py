"""Pull bounded row samples per table (row limits, random sampling where the dialect supports it).

Small tables are read in full. Large ones get a random sample, so profiling cost stays flat no
matter how big the source is. Note: random sampling makes sample-based stats vary slightly between
runs; the exact (SQL) stats never do.
"""
import pandas as pd
from sqlalchemy import column, func, select, table
from sqlalchemy.engine import Engine

# Each database spells "random row order" differently.
_RANDOM = {
    "sqlite": lambda: func.random(),
    "postgresql": lambda: func.random(),
    "mysql": lambda: func.rand(),
    "mariadb": lambda: func.rand(),
    "mssql": lambda: func.newid(),
    "oracle": lambda: func.dbms_random.value(),
}


def sample_table(engine: Engine, table_name: str, columns: list[str], schema: str | None,
                 row_count: int | None, limit: int) -> tuple[pd.DataFrame, str]:
    """Return (sample dataframe, method) where method is 'full', 'random' or 'first_n'."""
    t = table(table_name, *[column(c) for c in columns], schema=schema)
    stmt = select(*t.c)

    if row_count is not None and row_count <= limit:
        method = "full"
    elif engine.dialect.name in _RANDOM:
        stmt = stmt.order_by(_RANDOM[engine.dialect.name]()).limit(limit)
        method = "random"
    else:
        stmt = stmt.limit(limit)
        method = "first_n"

    with engine.connect() as conn:
        return pd.read_sql(stmt, conn), method
