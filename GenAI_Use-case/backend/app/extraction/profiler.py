"""Column statistics: null rate, distinct count, min/max, top values, length stats, regex pattern detection.

Hybrid strategy:
- EXACT stats come from SQL aggregates over the full table (one query per table): counts, nulls,
  distincts, min/max, negatives/zeros, future & placeholder dates, top values, FK orphans.
  Rare problems (5 bad rows in 50k) are never missed this way.
- SAMPLE stats come from pandas on a bounded sample: patterns, formats, lengths, percentiles, outliers.
  They describe the *shape* of the data, which a sample captures well.

Privacy: columns whose name or content looks like personal data (names, email, phone, DOB...) are
flagged `sensitive_hint`; their raw values are never stored - only patterns and masked examples.
"""
import re
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy import String, and_, case, cast, column, distinct, exists, func, select, table
from sqlalchemy.engine import Engine

from app.extraction.sampler import sample_table
from app.schemas.metadata import (
    BooleanStats,
    ColumnMetadata,
    ColumnProfile,
    DateStats,
    DetectedFormat,
    ForeignKeyMetadata,
    GenericType,
    NumericStats,
    PatternFrequency,
    SourceMetadata,
    TableMetadata,
    TextStats,
    ValueFrequency,
)

TOP_VALUES_MAX_DISTINCT = 50     # store exact value frequencies only for low-cardinality columns
TOP_PATTERNS = 10
EXACT_PATTERN_MIN_COVERAGE = 0.5  # below this, exact patterns are too varied -> collapse runs
FORMAT_MIN_MATCH = 0.8            # a format is "detected" when at least 80% of values match it
PLACEHOLDER_DATES = ["1900-01-01", "1899-12-31", "1970-01-01", "9999-12-31", "2099-12-31"]

SENSITIVE_NAME = re.compile(
    r"(first|last|middle|full|given|sur|customer|contact|person|user)_?name"
    r"|e_?mail|phone|mobile|address|birth|\bdob\b|ssn|aadhaar|passport|pan_?(no|number)|salary",
    re.IGNORECASE,
)
SENSITIVE_FORMATS = {"email", "phone"}

# Order matters on ties: the more specific format wins.
FORMATS = {
    "email": re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$"),
    "uuid": re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
    "url": re.compile(r"^https?://\S+$"),
    "iso_datetime": re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}"),
    "iso_date": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "phone": re.compile(r"^\+?[0-9][0-9 ()-]{7,16}$"),
    "decimal_text": re.compile(r"^-?\d+\.\d+$"),
    "integer_text": re.compile(r"^-?\d+$"),
    "code": re.compile(r"^[A-Z0-9]+([-_][A-Z0-9]+)*$"),
}

_TRUE = {True, 1, "1", "true", "t", "y", "yes"}
_FALSE = {False, 0, "0", "false", "f", "n", "no"}


# --------------------------------------------------------------------------- small pure helpers
def signature(value: str, collapse: bool = False) -> str:
    """Shape of a value: 'ORD-2024' -> 'AAA-9999'; collapsed: 'AAA-9999' -> 'A-9'."""
    out: list[str] = []
    for ch in value:
        k = "A" if ch.isupper() else "a" if ch.islower() else "9" if ch.isdigit() else ch
        if collapse and out and out[-1] == k and k in "Aa9":
            continue
        out.append(k)
    return "".join(out)


def mask(value: str) -> str:
    """'ravi.s12@gmail.com' -> 'r***@g***.com'; '+919876543210' -> '+9*********10'."""
    if "@" in value:
        local, _, domain = value.partition("@")
        name, dot, tld = domain.rpartition(".")
        return f"{local[:1]}***@{name[:1]}***{dot}{tld}" if dot else f"{local[:1]}***@***"
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "".join("*" if c.isalnum() else c for c in value[2:-2]) + value[-2:]


def _num(v) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


# --------------------------------------------------------------------------- entry point
def profile_source(engine: Engine, metadata: SourceMetadata, sample_limit: int,
                   as_of: date | None = None) -> SourceMetadata:
    """Attach profiles to every table/column/FK of `metadata` (in place) and return it."""
    as_of = as_of or date.today()
    for t in metadata.tables:
        profile_table(engine, t, metadata.schema_name, sample_limit, as_of)
    metadata.profiled_at = datetime.now(timezone.utc)
    return metadata


def profile_table(engine: Engine, t: TableMetadata, schema: str | None, sample_limit: int, as_of: date) -> None:
    exact = _exact_stats(engine, t, schema, as_of)
    t.row_count = exact["__rows"]

    df, method = sample_table(engine, t.name, [c.name for c in t.columns], schema, t.row_count, sample_limit)
    t.sample_size, t.sample_method = len(df), method

    for i, col in enumerate(t.columns):
        col.profile = _profile_column(engine, t, schema, col, exact, i, df[col.name])

    for fk in t.foreign_keys:
        fk.orphan_count, checked = _fk_orphans(engine, t.name, fk, schema)
        fk.orphan_rate = round(fk.orphan_count / checked, 6) if checked else 0.0


# --------------------------------------------------------------------------- exact stats (SQL)
def _exact_stats(engine: Engine, t: TableMetadata, schema: str | None, as_of: date) -> dict:
    """One aggregate query per table. Labels are positional (c0_nn, c0_d...) to avoid name clashes."""
    sa_t = table(t.name, *[column(c.name) for c in t.columns], schema=schema)
    exprs = [func.count().label("__rows")]
    for i, c in enumerate(t.columns):
        col = sa_t.c[c.name]
        exprs += [func.count(col).label(f"c{i}_nn"), func.count(distinct(col)).label(f"c{i}_d")]
        if c.generic_type in (GenericType.INTEGER, GenericType.DECIMAL, GenericType.FLOAT):
            exprs += [
                func.min(col).label(f"c{i}_min"), func.max(col).label(f"c{i}_max"),
                func.sum(case((col == 0, 1), else_=0)).label(f"c{i}_zero"),
                func.sum(case((col < 0, 1), else_=0)).label(f"c{i}_neg"),
            ]
        elif c.generic_type in (GenericType.DATE, GenericType.DATETIME):
            # Where dates are stored as text (SQLite), 'not a date' > '2026-01-01' is true - so a value
            # only counts as future if it starts with a 4-digit year.
            looks_dated = func.substr(cast(col, String), 1, 4).between("0000", "9999")
            exprs += [
                func.min(col).label(f"c{i}_min"), func.max(col).label(f"c{i}_max"),
                func.sum(case((and_(col > as_of, looks_dated), 1), else_=0)).label(f"c{i}_future"),
                func.sum(case((func.substr(cast(col, String), 1, 10).in_(PLACEHOLDER_DATES), 1), else_=0)).label(f"c{i}_ph"),
            ]
    with engine.connect() as conn:
        return dict(conn.execute(select(*exprs).select_from(sa_t)).mappings().one())


def _top_values(engine: Engine, table_name: str, schema: str | None, col_name: str, rows: int):
    sa_t = table(table_name, column(col_name), schema=schema)
    col = sa_t.c[col_name]
    n = func.count().label("n")
    stmt = select(col, n).group_by(col).order_by(n.desc()).limit(TOP_VALUES_MAX_DISTINCT)
    with engine.connect() as conn:
        return [ValueFrequency(value=v, count=k, share=round(k / rows, 6)) for v, k in conn.execute(stmt)]


def _fk_orphans(engine: Engine, child: str, fk: ForeignKeyMetadata, schema: str | None) -> tuple[int, int]:
    """Count non-null child keys with no parent. Aliases make self-referencing FKs work too."""
    c = table(child, *[column(x) for x in fk.columns], schema=schema).alias("c")
    p = table(fk.referred_table, *[column(x) for x in fk.referred_columns],
              schema=fk.referred_schema or schema).alias("p")
    match = and_(*[p.c[pc] == c.c[cc] for cc, pc in zip(fk.columns, fk.referred_columns)])
    not_null = and_(*[c.c[cc].is_not(None) for cc in fk.columns])
    with engine.connect() as conn:
        checked = conn.execute(select(func.count()).select_from(c).where(not_null)).scalar_one()
        orphans = conn.execute(
            select(func.count()).select_from(c).where(not_null, ~exists().where(match))
        ).scalar_one()
    return orphans, checked


# --------------------------------------------------------------------------- per-column
def _profile_column(engine: Engine, t: TableMetadata, schema: str | None, col: ColumnMetadata,
                    exact: dict, i: int, sample: pd.Series) -> ColumnProfile:
    rows = exact["__rows"]
    non_null = exact[f"c{i}_nn"]
    distinct_count = exact[f"c{i}_d"]
    values = sample.dropna()
    is_text = col.generic_type == GenericType.STRING
    str_values = values.astype(str) if is_text else None

    detected = _detect_format(str_values) if is_text and len(values) else None
    sensitive = bool(SENSITIVE_NAME.search(col.name)) or (detected is not None and detected.format in SENSITIVE_FORMATS)

    profile = ColumnProfile(
        row_count=rows,
        null_count=rows - non_null,
        null_rate=round((rows - non_null) / rows, 6) if rows else 0.0,
        distinct_count=distinct_count,
        uniqueness=round(distinct_count / non_null, 6) if non_null else None,
        is_constant=distinct_count == 1,
        sensitive_hint=sensitive,
        detected_format=detected,
    )

    if distinct_count <= TOP_VALUES_MAX_DISTINCT and not sensitive and rows:
        profile.top_values = _top_values(engine, t.name, schema, col.name, rows)

    if is_text and len(str_values):
        profile.patterns, profile.pattern_mode = _patterns(str_values)
        profile.text = _text_stats(str_values)
        profile.examples = _examples(str_values, profile.pattern_mode, sensitive)
    elif col.generic_type in (GenericType.INTEGER, GenericType.DECIMAL, GenericType.FLOAT):
        profile.numeric = _numeric_stats(values, exact, i)
    elif col.generic_type in (GenericType.DATE, GenericType.DATETIME):
        profile.date = DateStats(
            min=None if exact[f"c{i}_min"] is None else str(exact[f"c{i}_min"]),
            max=None if exact[f"c{i}_max"] is None else str(exact[f"c{i}_max"]),
            future_count=exact[f"c{i}_future"] or 0,
            placeholder_count=exact[f"c{i}_ph"] or 0,
            unparseable_count=int(pd.to_datetime(values, errors="coerce", format="ISO8601").isna().sum()),
        )
    elif col.generic_type == GenericType.BOOLEAN and profile.top_values:
        profile.boolean = _boolean_stats(profile.top_values)

    return profile


def _detect_format(values: pd.Series) -> DetectedFormat | None:
    best = None
    for name, rx in FORMATS.items():
        rate = values.str.match(rx).mean()
        if rate >= FORMAT_MIN_MATCH and (best is None or rate > best.match_rate):
            best = DetectedFormat(format=name, match_rate=round(float(rate), 6))
    return best


def _patterns(values: pd.Series) -> tuple[list[PatternFrequency], str]:
    total = len(values)
    mode = "exact"
    counts = values.map(signature).value_counts()
    if counts.head(TOP_PATTERNS).sum() / total < EXACT_PATTERN_MIN_COVERAGE:
        mode = "collapsed"
        counts = values.map(lambda v: signature(v, collapse=True)).value_counts()
    top = counts.head(TOP_PATTERNS)
    return [PatternFrequency(pattern=p, count=int(n), share=round(n / total, 6)) for p, n in top.items()], mode


def _text_stats(values: pd.Series) -> TextStats:
    lengths = values.str.len()
    has_letters = values[values.str.contains(r"[A-Za-z]", regex=True)]
    upper = int(has_letters.str.isupper().sum())
    lower = int(has_letters.str.islower().sum())
    return TextStats(
        min_length=int(lengths.min()),
        avg_length=round(float(lengths.mean()), 2),
        max_length=int(lengths.max()),
        blank_count=int((values.str.strip() == "").sum()),
        untrimmed_count=int((values != values.str.strip()).sum()),
        upper_count=upper,
        lower_count=lower,
        mixed_case_count=len(has_letters) - upper - lower,
    )


def _examples(values: pd.Series, mode: str, sensitive: bool) -> list[str]:
    """One example for each of the 3 most common patterns, so the variety is visible."""
    sig = values.map(lambda v: signature(v, collapse=(mode == "collapsed")))
    picks = [values[sig == p].iloc[0] for p in sig.value_counts().index[:3]]
    return [mask(v) if sensitive else v for v in picks]


def _numeric_stats(values: pd.Series, exact: dict, i: int) -> NumericStats:
    s = pd.to_numeric(values, errors="coerce").dropna()
    stats = NumericStats(
        min=_num(exact[f"c{i}_min"]), max=_num(exact[f"c{i}_max"]),
        zero_count=exact[f"c{i}_zero"] or 0, negative_count=exact[f"c{i}_neg"] or 0,
    )
    if s.empty:
        return stats
    q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    stats.mean, stats.std = round(float(s.mean()), 4), round(float(s.std()), 4) if len(s) > 1 else None
    stats.p01, stats.p05, stats.median, stats.p95, stats.p99 = (round(float(q[k]), 4)
                                                                for k in (0.01, 0.05, 0.5, 0.95, 0.99))
    iqr = q[0.75] - q[0.25]
    if iqr > 0:   # with IQR 0 (e.g. quantity is almost always 1) fences are meaningless
        lo, hi = q[0.25] - 3 * iqr, q[0.75] + 3 * iqr
        stats.outlier_bounds = [round(float(lo), 4), round(float(hi), 4)]
        stats.outlier_count = int(((s < lo) | (s > hi)).sum())
    return stats


def _boolean_stats(top: list[ValueFrequency]) -> BooleanStats:
    def norm(v):
        return v.strip().lower() if isinstance(v, str) else v
    true = sum(f.count for f in top if f.value is not None and norm(f.value) in _TRUE)
    false = sum(f.count for f in top if f.value is not None and norm(f.value) in _FALSE)
    invalid = [f.value for f in top if f.value is not None and norm(f.value) not in _TRUE | _FALSE]
    return BooleanStats(true_count=true, false_count=false, invalid_values=invalid)
