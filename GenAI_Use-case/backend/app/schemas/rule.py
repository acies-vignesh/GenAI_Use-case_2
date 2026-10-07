"""Structured DQ rule contract: category (data quality / hierarchy / business), dimension (completeness, validity, ...), target entity/attribute, check type, parameters, severity, rationale, fingerprint.

A rule is *declarative*: it says WHAT to check (check type + parameters), never HOW (no SQL, except
the flagged `custom_sql` escape hatch). SQL is generated from it later, per database dialect.

Semantics shared by all checks:
- A row FAILS only when the check evaluates to FALSE. NULL never fails a check - except
  `not_null` / `null_rate_max`, whose whole job is NULLs. (Same convention as SQL CHECK constraints.)
- `filter` restricts a rule to matching rows, e.g. pattern '^[A-Z]{3}$' only WHERE region_level = 'CITY'.
- `tolerance` is the failure rate the business accepts (0 = every row must pass).

Expressions (filter, row_condition, conditional, aggregate_compare) are small ANSI-SQL boolean/scalar
expressions over the target entity's columns, validated and transpiled by sqlglot (app/rules/expressions.py).
"""
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from app.schemas.metadata import ScalarValue


class RuleCategory(str, Enum):
    DATA_QUALITY = "data_quality"
    HIERARCHY = "hierarchy"
    BUSINESS = "business"


class Dimension(str, Enum):
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    CONSISTENCY = "consistency"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"


class Scope(str, Enum):
    COLUMN = "column"            # one column, row by row
    ROW = "row"                  # several columns of the same row
    TABLE = "table"              # needs the whole table (uniqueness, tree checks)
    CROSS_TABLE = "cross_table"  # needs another table (FKs, aggregates, hierarchy attachment)


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RuleOrigin(str, Enum):
    LLM = "llm"
    STEWARD = "steward"
    HEURISTIC = "heuristic"


class RuleStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"


Operator = Literal["=", "<>", "<", "<=", ">", ">="]


# --------------------------------------------------------------------------- check catalogue
# Each check is a flat object discriminated by `type`. Pydantic picks the right class from `type`
# and validates its parameters - an LLM answer with a wrong/missing parameter is rejected here.

class _Check(BaseModel):
    model_config = {"extra": "forbid"}


# column checks
class NotNull(_Check):
    type: Literal["not_null"] = "not_null"


class NullRateMax(_Check):
    type: Literal["null_rate_max"] = "null_rate_max"
    max_rate: float = Field(ge=0, le=1)


class Unique(_Check):
    """Target columns together must be unique (composite key when several)."""
    type: Literal["unique"] = "unique"


class AllowedValues(_Check):
    type: Literal["allowed_values"] = "allowed_values"
    values: list[ScalarValue] = Field(min_length=1)
    case_sensitive: bool = True


class Pattern(_Check):
    type: Literal["pattern"] = "pattern"
    regex: str

    @field_validator("regex")
    @classmethod
    def _compiles(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as e:   # pydantic only turns ValueError into a validation error
            raise ValueError(f"invalid regex: {e}") from e
        return v


class Range(_Check):
    type: Literal["range"] = "range"
    min: float | None = None
    max: float | None = None
    inclusive: bool = True

    @model_validator(mode="after")
    def _bounds(self):
        if self.min is None and self.max is None:
            raise ValueError("range needs min and/or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("range min > max")
        return self


class LengthRange(_Check):
    type: Literal["length_range"] = "length_range"
    min_length: int | None = Field(None, ge=0)
    max_length: int | None = Field(None, ge=0)


class DateNotFuture(_Check):
    type: Literal["date_not_future"] = "date_not_future"


class NotPlaceholder(_Check):
    type: Literal["not_placeholder"] = "not_placeholder"
    values: list[ScalarValue] = Field(min_length=1)


# row checks
class RowCondition(_Check):
    type: Literal["row_condition"] = "row_condition"
    expression: str = Field(description="Boolean SQL expression every row must satisfy")


class Conditional(_Check):
    type: Literal["conditional"] = "conditional"
    when: str = Field(description="Boolean SQL expression selecting the rows the rule applies to")
    then: str = Field(description="Boolean SQL expression those rows must satisfy")


# cross-table checks
class ForeignKey(_Check):
    type: Literal["foreign_key"] = "foreign_key"
    ref_entity: str
    ref_columns: list[str] = Field(min_length=1)


class HasChildren(_Check):
    """Every target row must have at least `min_children` rows in child_entity."""
    type: Literal["has_children"] = "has_children"
    child_entity: str
    child_join_columns: list[str] = Field(min_length=1)
    parent_join_columns: list[str] = Field(min_length=1)
    min_children: int = Field(1, ge=1)


class AggregateCompare(_Check):
    """parent_expression <operator> child aggregate, per target row (+/- tolerance_abs for '=')."""
    type: Literal["aggregate_compare"] = "aggregate_compare"
    child_entity: str
    child_join_columns: list[str] = Field(min_length=1)
    parent_join_columns: list[str] = Field(min_length=1)
    child_expression: str = Field(description="Aggregate over child columns, e.g. SUM(line_amount)")
    operator: Operator
    parent_expression: str = Field(description="Expression over target columns, e.g. total_amount + discount_amount")
    tolerance_abs: float = Field(0.0, ge=0)


# hierarchy checks
class HierarchyParentLevel(_Check):
    """Each node's parent must sit exactly one level above it; top-level nodes have no parent."""
    type: Literal["hierarchy_parent_level"] = "hierarchy_parent_level"
    hierarchy: str


class HierarchyAcyclic(_Check):
    type: Literal["hierarchy_acyclic"] = "hierarchy_acyclic"
    hierarchy: str


class HierarchyAttachLevel(_Check):
    """The target column must reference a hierarchy node at `level`."""
    type: Literal["hierarchy_attach_level"] = "hierarchy_attach_level"
    hierarchy: str
    level: str


# escape hatch
class CustomSql(_Check):
    """SELECT returning the FAILING rows. Always needs extra steward scrutiny."""
    type: Literal["custom_sql"] = "custom_sql"
    sql: str


Check = Annotated[
    NotNull | NullRateMax | Unique | AllowedValues | Pattern | Range | LengthRange | DateNotFuture
    | NotPlaceholder | RowCondition | Conditional | ForeignKey | HasChildren | AggregateCompare
    | HierarchyParentLevel | HierarchyAcyclic | HierarchyAttachLevel | CustomSql,
    Field(discriminator="type"),
]

# check type -> (allowed scopes, min target columns, max target columns or None)
CHECK_SHAPE: dict[str, tuple[set[Scope], int, int | None]] = {
    "not_null": ({Scope.COLUMN}, 1, 1),
    "null_rate_max": ({Scope.COLUMN}, 1, 1),
    "unique": ({Scope.COLUMN, Scope.TABLE}, 1, None),
    "allowed_values": ({Scope.COLUMN}, 1, 1),
    "pattern": ({Scope.COLUMN}, 1, 1),
    "range": ({Scope.COLUMN}, 1, 1),
    "length_range": ({Scope.COLUMN}, 1, 1),
    "date_not_future": ({Scope.COLUMN}, 1, 1),
    "not_placeholder": ({Scope.COLUMN}, 1, 1),
    "row_condition": ({Scope.ROW}, 1, None),
    "conditional": ({Scope.ROW}, 1, None),
    "foreign_key": ({Scope.CROSS_TABLE}, 1, None),
    "has_children": ({Scope.CROSS_TABLE}, 0, None),
    "aggregate_compare": ({Scope.CROSS_TABLE}, 0, None),
    "hierarchy_parent_level": ({Scope.TABLE}, 0, None),
    "hierarchy_acyclic": ({Scope.TABLE}, 0, None),
    "hierarchy_attach_level": ({Scope.CROSS_TABLE}, 1, 1),
    "custom_sql": (set(Scope), 0, None),
}
CHECK_TYPES = list(CHECK_SHAPE)


# --------------------------------------------------------------------------- the rule
class Target(BaseModel):
    entity: str
    columns: list[str] = []


class ReasonCode(str, Enum):
    """Why a steward rejected (or edited) a rule - a structured signal the generator learns from."""
    WRONG = "wrong"                        # contradicts the business
    NOT_RELEVANT = "not_relevant"          # correct, but nobody cares
    TOO_STRICT = "too_strict"              # right idea, threshold too tight
    TOO_LOOSE = "too_loose"                # right idea, threshold too lax
    ALREADY_ENFORCED = "already_enforced"  # the database / another system guarantees it
    DUPLICATE = "duplicate"                # covered by another rule
    OTHER = "other"


class Review(BaseModel):
    decision: Literal["approved", "rejected", "modified"]
    reviewer: str
    reason_code: ReasonCode | None = None
    reason: str | None = None
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    previous: dict | None = Field(None, description="Rule fields before a 'modified' edit")


class Rule(BaseModel):
    rule_id: str = Field(default_factory=lambda: uuid4().hex)
    source_id: str
    name: str
    description: str | None = None
    category: RuleCategory
    dimension: Dimension
    scope: Scope
    target: Target
    check: Check
    filter: str | None = Field(None, description="Boolean SQL expression: rule applies only to these rows")
    severity: Severity = Severity.MEDIUM
    tolerance: float = Field(0.0, ge=0, le=1)
    rationale: str | None = None
    evidence: list[str] = []
    origin: RuleOrigin
    generation_run_id: str | None = None
    model: str | None = None
    status: RuleStatus = RuleStatus.PROPOSED
    reviews: list[Review] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _shape(self):
        scopes, lo, hi = CHECK_SHAPE[self.check.type]
        if self.scope not in scopes:
            raise ValueError(f"check '{self.check.type}' needs scope {sorted(s.value for s in scopes)}, "
                             f"got '{self.scope.value}'")
        n = len(self.target.columns)
        if n < lo or (hi is not None and n > hi):
            want = f"exactly {lo}" if lo == hi else f"at least {lo}" if hi is None else f"{lo}-{hi}"
            raise ValueError(f"check '{self.check.type}' needs {want} target column(s), got {n}")
        if isinstance(self.check, ForeignKey) and len(self.check.ref_columns) != n:
            raise ValueError("foreign_key: ref_columns must pair 1:1 with target columns")
        if self.check.type.startswith("hierarchy_") and self.category != RuleCategory.HIERARCHY:
            raise ValueError("hierarchy checks must have category 'hierarchy'")
        return self

    # Imported lazily: fingerprinting needs sqlglot normalisation from app.rules
    @computed_field
    @property
    def fingerprint(self) -> str:
        from app.rules.fingerprint import fingerprint
        return fingerprint(self)

    @computed_field
    @property
    def similarity_key(self) -> str:
        from app.rules.fingerprint import similarity_key
        return similarity_key(self)
