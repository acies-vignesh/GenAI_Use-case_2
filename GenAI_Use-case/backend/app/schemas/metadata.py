"""Raw extracted metadata: tables, columns, types, PK/FK, constraints, profiling stats. (Output of Layer 3)

Two kinds of facts live here, both free of interpretation:
- what the database *declares* (Phase 1: schema_crawler.py)
- what the data *actually contains* (Phase 2: profiler.py -> the optional `profile` fields)
Business meaning ("this column is a customer email") is added later by the semantic layer.
"""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

ScalarValue = str | int | float | bool | None


class GenericType(str, Enum):
    """Dialect-independent type bucket, so later layers never deal with VARCHAR2 vs NVARCHAR vs TEXT."""
    INTEGER = "integer"
    DECIMAL = "decimal"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    JSON = "json"
    BINARY = "binary"
    OTHER = "other"


# --------------------------------------------------------------------------- profiling (Phase 2)
# Field naming convention: counts computed with SQL over the FULL table are exact; anything under
# `text`, the percentiles/mean/outliers in `numeric`, `patterns` and `detected_format` come from the sample.

class ValueFrequency(BaseModel):
    value: ScalarValue
    count: int
    share: float


class PatternFrequency(BaseModel):
    pattern: str = Field(description="Shape of the value: A=upper, a=lower, 9=digit, other chars kept")
    count: int
    share: float


class DetectedFormat(BaseModel):
    format: str
    match_rate: float


class TextStats(BaseModel):
    min_length: int
    avg_length: float
    max_length: int
    blank_count: int = Field(description="Empty or whitespace-only strings")
    untrimmed_count: int = Field(description="Values with leading/trailing whitespace")
    upper_count: int
    lower_count: int
    mixed_case_count: int


class NumericStats(BaseModel):
    min: float | None = None              # exact
    max: float | None = None              # exact
    zero_count: int = 0                   # exact
    negative_count: int = 0               # exact
    mean: float | None = None
    std: float | None = None
    p01: float | None = None
    p05: float | None = None
    median: float | None = None
    p95: float | None = None
    p99: float | None = None
    outlier_bounds: list[float] | None = Field(None, description="Tukey far-out fences: Q1-3*IQR, Q3+3*IQR")
    outlier_count: int | None = None      # within the sample


class DateStats(BaseModel):
    min: str | None = None                # exact
    max: str | None = None                # exact
    future_count: int = 0                 # exact, relative to profiled_at
    placeholder_count: int = 0            # exact (1900-01-01, 9999-12-31, ...)
    unparseable_count: int = 0            # within the sample


class BooleanStats(BaseModel):
    true_count: int
    false_count: int
    invalid_values: list[ScalarValue] = []


class ColumnProfile(BaseModel):
    row_count: int
    null_count: int
    null_rate: float
    distinct_count: int
    uniqueness: float | None = Field(None, description="distinct / non-null values (1.0 = all unique)")
    is_constant: bool
    sensitive_hint: bool = Field(False, description="Looks like personal data: raw values are withheld")
    top_values: list[ValueFrequency] | None = Field(
        None, description="Exact, only for low-cardinality non-sensitive columns")
    patterns: list[PatternFrequency] | None = None
    pattern_mode: str | None = Field(None, description="'exact' or 'collapsed' (runs of the same class merged)")
    detected_format: DetectedFormat | None = None
    examples: list[str] = Field([], description="Up to 3 values, masked when sensitive_hint")
    text: TextStats | None = None
    numeric: NumericStats | None = None
    date: DateStats | None = None
    boolean: BooleanStats | None = None


# --------------------------------------------------------------------------- structure (Phase 1)

class ColumnMetadata(BaseModel):
    name: str
    ordinal_position: int
    native_type: str = Field(description="Type exactly as the database reports it, e.g. VARCHAR(10)")
    generic_type: GenericType
    nullable: bool
    is_primary_key: bool = False
    max_length: int | None = None
    precision: int | None = None
    scale: int | None = None
    default: str | None = None
    comment: str | None = None
    profile: ColumnProfile | None = None


class ForeignKeyMetadata(BaseModel):
    name: str | None = None
    columns: list[str]
    referred_schema: str | None = None
    referred_table: str
    referred_columns: list[str]
    is_self_referencing: bool = Field(description="Points at its own table - usually a parent/child hierarchy")
    orphan_count: int | None = Field(None, description="Non-null child values with no matching parent row")
    orphan_rate: float | None = None


class UniqueConstraintMetadata(BaseModel):
    name: str | None = None
    columns: list[str]


class IndexMetadata(BaseModel):
    name: str | None = None
    columns: list[str]
    unique: bool


class TableMetadata(BaseModel):
    name: str
    schema_name: str | None = None
    row_count: int | None = None
    comment: str | None = None
    sample_size: int | None = None
    sample_method: str | None = Field(None, description="'full', 'random' or 'first_n'")
    columns: list[ColumnMetadata]
    primary_key: list[str] = []
    foreign_keys: list[ForeignKeyMetadata] = []
    unique_constraints: list[UniqueConstraintMetadata] = []
    indexes: list[IndexMetadata] = []

    def column(self, name: str) -> ColumnMetadata:
        return next(c for c in self.columns if c.name == name)


class SourceMetadata(BaseModel):
    metadata_version: str = "1.0"
    source_id: str = Field(description="Stable fingerprint of the source (see connectivity/fingerprint.py)")
    dialect: str
    database: str
    schema_name: str | None = None
    extracted_at: datetime
    profiled_at: datetime | None = None
    tables: list[TableMetadata]

    def table(self, name: str) -> TableMetadata:
        return next(t for t in self.tables if t.name == name)
