"""Reflect tables, columns, types, keys, constraints, FK relationships via SQLAlchemy inspector.

"Reflection" = asking the database to describe itself. Every database keeps a catalog
(information_schema, sqlite_master, ...); SQLAlchemy's Inspector reads it through one API.
"""
from datetime import datetime, timezone

from sqlalchemy import func, inspect, select, table
from sqlalchemy import types as sat
from sqlalchemy.engine import Engine, Inspector

from app.connectivity.connectors import ConnectionConfig
from app.connectivity.fingerprint import source_fingerprint
from app.schemas.metadata import (
    ColumnMetadata,
    ForeignKeyMetadata,
    GenericType,
    IndexMetadata,
    SourceMetadata,
    TableMetadata,
    UniqueConstraintMetadata,
)

# Order matters: subclasses before their parents (Float is a Numeric, Text is a String, ...).
_TYPE_MAP: list[tuple[type, GenericType]] = [
    (sat.Boolean, GenericType.BOOLEAN),
    (sat.Integer, GenericType.INTEGER),
    (sat.Float, GenericType.FLOAT),
    (sat.Numeric, GenericType.DECIMAL),
    (sat.DateTime, GenericType.DATETIME),
    (sat.Date, GenericType.DATE),
    (sat.Time, GenericType.TIME),
    (sat.JSON, GenericType.JSON),
    (sat.LargeBinary, GenericType.BINARY),
    (sat.String, GenericType.STRING),
]


def generic_type(sa_type: sat.TypeEngine) -> GenericType:
    for cls, generic in _TYPE_MAP:
        if isinstance(sa_type, cls):
            return generic
    return GenericType.OTHER


def crawl_schema(engine: Engine, config: ConnectionConfig, include_row_counts: bool = True) -> SourceMetadata:
    insp = inspect(engine)
    schema = config.schema_name
    tables = [
        _crawl_table(engine, insp, name, schema, include_row_counts)
        for name in sorted(insp.get_table_names(schema=schema))
    ]
    return SourceMetadata(
        source_id=source_fingerprint(config),
        dialect=engine.dialect.name,
        database=config.display_name,
        schema_name=schema,
        extracted_at=datetime.now(timezone.utc),
        tables=tables,
    )


def _crawl_table(engine: Engine, insp: Inspector, name: str, schema: str | None,
                 include_row_counts: bool) -> TableMetadata:
    pk_cols = (insp.get_pk_constraint(name, schema=schema) or {}).get("constrained_columns") or []

    columns = []
    for position, col in enumerate(insp.get_columns(name, schema=schema), start=1):
        sa_type = col["type"]
        columns.append(ColumnMetadata(
            name=col["name"],
            ordinal_position=position,
            native_type=_native_type(sa_type, engine),
            generic_type=generic_type(sa_type),
            nullable=bool(col.get("nullable", True)) and col["name"] not in pk_cols,
            is_primary_key=col["name"] in pk_cols,
            max_length=getattr(sa_type, "length", None),
            precision=getattr(sa_type, "precision", None),
            scale=getattr(sa_type, "scale", None),
            default=str(col["default"]) if col.get("default") is not None else None,
            comment=col.get("comment"),
        ))

    foreign_keys = [
        ForeignKeyMetadata(
            name=fk.get("name"),
            columns=fk["constrained_columns"],
            referred_schema=fk.get("referred_schema"),
            referred_table=fk["referred_table"],
            referred_columns=fk["referred_columns"],
            is_self_referencing=fk["referred_table"] == name and fk.get("referred_schema") in (None, schema),
        )
        for fk in insp.get_foreign_keys(name, schema=schema)
    ]

    unique_constraints = [
        UniqueConstraintMetadata(name=uc.get("name"), columns=uc["column_names"])
        for uc in insp.get_unique_constraints(name, schema=schema)
    ]
    indexes = []
    is_sqlite = engine.dialect.name == "sqlite"
    raw_indexes = (insp.get_indexes(name, schema=schema, include_auto_indexes=True) if is_sqlite
                   else insp.get_indexes(name, schema=schema))
    for ix in raw_indexes:
        cols = [c for c in ix["column_names"] if c]
        if is_sqlite and (ix.get("name") or "").startswith("sqlite_autoindex_"):
            # SQLite's parser misses inline `col TYPE(n) UNIQUE`; the auto-index SQLite builds for it is
            # the reliable signal. Skip ones that just back the primary key or a constraint already found.
            known = [uc.columns for uc in unique_constraints] + [pk_cols]
            if ix["unique"] and cols not in known:
                unique_constraints.append(UniqueConstraintMetadata(name=None, columns=cols))
            continue
        indexes.append(IndexMetadata(name=ix.get("name"), columns=cols, unique=bool(ix["unique"])))

    return TableMetadata(
        name=name,
        schema_name=schema,
        row_count=_row_count(engine, name, schema) if include_row_counts else None,
        comment=_table_comment(insp, name, schema),
        columns=columns,
        primary_key=pk_cols,
        foreign_keys=foreign_keys,
        unique_constraints=unique_constraints,
        indexes=indexes,
    )


def _native_type(sa_type: sat.TypeEngine, engine: Engine) -> str:
    try:
        return str(sa_type.compile(dialect=engine.dialect))
    except Exception:  # some reflected types can't be rendered back
        return repr(sa_type)


def _table_comment(insp: Inspector, name: str, schema: str | None) -> str | None:
    try:
        return insp.get_table_comment(name, schema=schema).get("text")
    except NotImplementedError:  # SQLite has no table comments
        return None


def _row_count(engine: Engine, name: str, schema: str | None) -> int:
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(table(name, schema=schema))).scalar_one()
