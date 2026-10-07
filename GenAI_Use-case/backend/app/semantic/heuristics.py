"""Deterministic first pass: name/pattern/profile based semantic typing (email, id, date, currency, code...).

Evidence scoring: every signal adds points to a candidate semantic type; the winner's share of all
points is its confidence. Signals:
  - keys        : primary key / FK / self-referencing FK (strongest)
  - name        : words in the column name (one match per group, most specific first)
  - format      : detected_format from profiling (email, phone, code...)
  - statistics  : "this is a code / a number / a date / text" - credited to the most specific
                  type the name already suggests within that family, so name + stats reinforce
                  each other instead of competing (status: name says status_code, stats say "some code").
Incompatible types are removed (a DATE column can't be an email).
"""
import re
from collections import defaultdict
from dataclasses import dataclass, field

from app.schemas.metadata import ColumnMetadata, GenericType as G, TableMetadata, ValueFrequency
from app.schemas.semantic_model import AttributeRole, SemanticType as ST, ValueAnomaly

REVIEW_BELOW = 0.9                 # attributes below this confidence are sent to the LLM for review
CANONICAL_MIN_SHARE = 0.01         # a value this common is valid unless it is a case variant
STYLE_MIN_SHARE = 0.003            # a rarer value is valid only if it also matches the style of valid ones
DOMINANT_PATTERN_SHARE = 0.95

NUMERIC_G = {G.INTEGER, G.DECIMAL, G.FLOAT}
DATE_TYPES = {ST.BIRTH_DATE, ST.EVENT_DATE, ST.EVENT_TIMESTAMP}
NUMERIC_TYPES = {ST.AMOUNT, ST.PRICE, ST.COST, ST.QUANTITY, ST.PERCENTAGE, ST.RATE, ST.COUNT}
KEY_TYPES = {ST.IDENTIFIER, ST.BUSINESS_KEY, ST.FOREIGN_KEY, ST.PARENT_KEY}
CODE_TYPES = {ST.STATUS_CODE, ST.CATEGORY_CODE, ST.HIERARCHY_LEVEL, ST.CURRENCY_CODE, ST.COUNTRY_CODE}
TEXT_TYPES = {ST.LABEL, ST.PERSON_NAME, ST.ORGANIZATION_NAME, ST.DESCRIPTION, ST.FREE_TEXT}
PII_TYPES = {ST.PERSON_NAME, ST.EMAIL, ST.PHONE, ST.ADDRESS, ST.BIRTH_DATE, ST.POSTAL_CODE}
PII_NAME = re.compile(r"gender|(^|_)sex($|_)|ethnic|religion|nationality|marital")

# (group, regex over the snake_case name, type, weight). Within a group only the FIRST match counts,
# so order = most specific first ("cost_price" is a cost, not a price; "first_name" is a person, not a label).
NAME_HINTS: list[tuple[str, re.Pattern, ST, float]] = [(g, re.compile(rx), t, w) for g, rx, t, w in [
    ("contact", r"(^|_)e_?mail(_|$)", ST.EMAIL, 3),
    ("contact", r"(^|_)(phone|mobile|tel|telephone|cell)(_|$)", ST.PHONE, 3),
    ("contact", r"(^|_)(address|street|addr)(_|$)", ST.ADDRESS, 3),
    ("contact", r"(^|_)(zip|pin|postal|postcode|pincode)(_|$)", ST.POSTAL_CODE, 3),
    ("contact", r"(^|_)(url|website|link)(_|$)", ST.URL, 3),
    ("name", r"(^|_)(first|last|middle|given|sur|full)_?name$|^(customer|contact|person|employee)_name$",
     ST.PERSON_NAME, 3),
    ("name", r"(^|_)(company|org|organization|organisation|supplier|vendor)_name$", ST.ORGANIZATION_NAME, 3),
    ("name", r"(desc|description|remarks?|notes?|comments?)$", ST.DESCRIPTION, 2.5),
    ("name", r"(_name|_title)$|^(name|title)$", ST.LABEL, 2),
    ("date", r"(^|_)(dob|birth_?date|date_of_birth|birthday)(_|$)", ST.BIRTH_DATE, 3),
    ("date", r"(_at|_ts|_time|_timestamp|_datetime)$", ST.EVENT_TIMESTAMP, 2),
    ("date", r"(_date|_dt|_on|_day)$|^date_", ST.EVENT_DATE, 2),
    ("number", r"(_pct|_percent|_percentage|_perc)$|^(pct|percent)_", ST.PERCENTAGE, 3),
    ("number", r"(^|_)cost(_|$)", ST.COST, 3),
    ("number", r"(^|_)(price|mrp)(_|$)", ST.PRICE, 3),
    ("number", r"(^|_)(qty|quantity|units)(_|$)", ST.QUANTITY, 3),
    ("number", r"(^|_)(amount|amt|total|revenue|sales|balance|fee|charge|tax|value)(_|$)", ST.AMOUNT, 2.5),
    ("number", r"(_rate|_ratio)$", ST.RATE, 2),
    ("number", r"(_count|_cnt)$|^(num|no_of|count)_", ST.COUNT, 2),
    ("code", r"(^|_)(currency|ccy)(_code)?$", ST.CURRENCY_CODE, 3),
    ("code", r"(^|_)country(_code)?$", ST.COUNTRY_CODE, 3),
    ("code", r"(^|_)status$", ST.STATUS_CODE, 3),
    ("code", r"(^|_)(level|tier|depth|rank)$", ST.HIERARCHY_LEVEL, 3),
    ("code", r"(type|category|segment|class|group|channel|method|mode|gender|kind)$", ST.CATEGORY_CODE, 2.5),
    ("flag", r"^(is|has|can|should|was)_|(_flag|_yn|_ind)$", ST.FLAG, 3),
    ("key", r"(^|_)(sku|code|number|no|ref|reference)$", ST.BUSINESS_KEY, 1.5),
]]

FORMAT_HINTS = {"email": ST.EMAIL, "phone": ST.PHONE, "url": ST.URL, "uuid": ST.IDENTIFIER}


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


@dataclass
class ColumnContext:
    table: TableMetadata
    column: ColumnMetadata
    fk_target: str | None = None        # referenced table (declared or inferred FK)
    is_self_fk: bool = False
    table_is_self_referencing: bool = False


@dataclass
class TypeGuess:
    semantic_type: ST
    confidence: float
    reasons: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- semantic type
def _compatible(generic: G, fmt: str | None) -> set[ST]:
    every = set(ST)
    if generic in (G.DATE, G.DATETIME):
        return DATE_TYPES | {ST.OTHER}
    if generic == G.BOOLEAN:
        return {ST.FLAG, ST.OTHER}
    if generic in NUMERIC_G:
        return NUMERIC_TYPES | KEY_TYPES | CODE_TYPES | {ST.FLAG, ST.POSTAL_CODE, ST.PHONE, ST.OTHER}
    allowed = every - NUMERIC_TYPES - DATE_TYPES
    if fmt in ("integer_text", "decimal_text"):
        allowed |= NUMERIC_TYPES
    if fmt in ("iso_date", "iso_datetime"):
        allowed |= DATE_TYPES
    return allowed


def guess_semantic_type(ctx: ColumnContext) -> TypeGuess:
    col, p = ctx.column, ctx.column.profile
    name = snake(col.name)
    scores: dict[ST, float] = defaultdict(float)
    reasons: dict[ST, list[str]] = defaultdict(list)

    def add(t: ST, w: float, why: str) -> None:
        scores[t] += w
        reasons[t].append(why)

    # keys
    if col.is_primary_key:
        add(ST.IDENTIFIER, 5, "primary key")
    if ctx.is_self_fk:
        add(ST.PARENT_KEY, 5, "references its own table")
    elif ctx.fk_target:
        add(ST.FOREIGN_KEY, 5, f"references {ctx.fk_target}")

    # name
    seen_groups: set[str] = set()
    for group, rx, t, w in NAME_HINTS:
        if group not in seen_groups and rx.search(name):
            seen_groups.add(group)
            add(t, w, f"name '{col.name}'")
    if scores[ST.HIERARCHY_LEVEL] and ctx.table_is_self_referencing:
        add(ST.HIERARCHY_LEVEL, 2, "level column in a self-referencing table")

    # format
    fmt = p.detected_format.format if p and p.detected_format else None
    if fmt in FORMAT_HINTS:
        add(FORMAT_HINTS[fmt], 3, f"{p.detected_format.match_rate:.0%} of values look like {fmt}")

    # statistics - credited to the best name-suggested type of the family
    def credit(family: set[ST], default: ST, w: float, why: str) -> None:
        hinted = [t for t in family if scores[t] > 0]
        add(max(hinted, key=lambda t: scores[t]) if hinted else default, w, why)

    is_key = col.is_primary_key or ctx.fk_target is not None
    if col.generic_type == G.BOOLEAN:
        add(ST.FLAG, 4, "boolean type")
    elif col.generic_type in (G.DATE, G.DATETIME):
        credit(DATE_TYPES, ST.EVENT_TIMESTAMP if col.generic_type == G.DATETIME else ST.EVENT_DATE, 3, "date type")
    elif p and not is_key:
        values = {v.value for v in (p.top_values or []) if v.value is not None}
        if col.generic_type in NUMERIC_G and p.distinct_count == 2 and values <= {0, 1}:
            add(ST.FLAG, 2, "only 0/1 values")
        elif col.generic_type in NUMERIC_G:
            credit(NUMERIC_TYPES, ST.AMOUNT if col.generic_type != G.INTEGER else ST.COUNT, 2, "numeric")
        elif col.generic_type == G.STRING and fmt not in FORMAT_HINTS:
            code_like = fmt == "code"
            uniq = p.uniqueness or 0
            text_named = any(scores[t] for t in TEXT_TYPES)
            if code_like and uniq >= 0.95 and p.null_rate < 0.05:
                add(ST.BUSINESS_KEY, 3, f"near-unique code ({uniq:.1%} distinct)")
            elif p.distinct_count <= 50 and uniq < 0.5 and not text_named:
                credit(CODE_TYPES, ST.CATEGORY_CODE, 2.5 if code_like else 1.5,
                       f"{p.distinct_count} distinct values")
            else:
                long_text = p.text is not None and p.text.avg_length > 40
                credit(TEXT_TYPES, ST.FREE_TEXT if long_text else ST.LABEL, 1, "varied text")

    add(ST.OTHER, 0.5, "baseline")
    allowed = _compatible(col.generic_type, fmt)
    scores = {t: s for t, s in scores.items() if t in allowed}
    best = max(scores, key=scores.get)
    return TypeGuess(best, round(scores[best] / sum(scores.values()), 2), reasons[best])


def guess_role(semantic_type: ST) -> AttributeRole:
    if semantic_type in (ST.IDENTIFIER, ST.BUSINESS_KEY):
        return AttributeRole.KEY
    if semantic_type in (ST.FOREIGN_KEY, ST.PARENT_KEY):
        return AttributeRole.FOREIGN_KEY
    if semantic_type in NUMERIC_TYPES:
        return AttributeRole.MEASURE
    if semantic_type in CODE_TYPES | {ST.FLAG}:
        return AttributeRole.DIMENSION
    return AttributeRole.ATTRIBUTE


def is_pii(semantic_type: ST, column: ColumnMetadata) -> bool:
    sensitive = column.profile.sensitive_hint if column.profile else False
    return semantic_type in PII_TYPES or sensitive or bool(PII_NAME.search(snake(column.name)))


# --------------------------------------------------------------------------- patterns
def signature_to_regex(signature: str) -> str:
    """'AAA-99999999' -> '^[A-Z]{3}-\\d{8}$' (inverse of profiler.signature)."""
    classes = {"A": "[A-Z]", "a": "[a-z]", "9": r"\d"}
    out = []
    for run in re.finditer(r"(.)\1*", signature):
        ch, n = run.group(1), len(run.group(0))
        token = classes.get(ch, re.escape(ch))
        out.append(token if n == 1 else f"{token}{{{n}}}")
    return "^" + "".join(out) + "$"


def expected_pattern(semantic_type: ST, column: ColumnMetadata) -> str | None:
    p = column.profile
    if semantic_type == ST.EMAIL:
        return r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$"
    if semantic_type not in (ST.BUSINESS_KEY, ST.PHONE, ST.POSTAL_CODE) or not p or not p.patterns:
        return None
    top = p.patterns[0]
    if p.pattern_mode == "exact" and top.share >= DOMINANT_PATTERN_SHARE:
        return signature_to_regex(top.pattern)
    return None


# --------------------------------------------------------------------------- allowed values
def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def _variant_of(value: str, canonical: list[str], rare: bool) -> tuple[str, str] | None:
    v = value.casefold()
    for c in canonical:
        cf = c.casefold()
        if v == cf:
            return c, f"case variant of '{c}'"
    if not rare:
        return None
    for c in canonical:
        cf = c.casefold()
        if len(v) >= 4 and _levenshtein(v, cf) <= (1 if len(cf) < 8 else 2):
            return c, f"misspelling of '{c}'"
        if 2 <= len(v) < len(cf) and v[0] == cf[0] and _is_subsequence(v, cf):
            return c, f"abbreviation of '{c}'"
        if len(cf) <= 3 and v.startswith(cf) and len(v) > len(cf):
            return c, f"spelled-out form of '{c}'"
        if v in re.split(r"[_\-\s]+", cf) and len(v) >= 3:
            return c, f"short form of '{c}'"
    return None


def _style(value: str) -> tuple:
    letters = [ch for ch in value if ch.isalpha()]
    case = "upper" if letters and all(ch.isupper() for ch in letters) else \
        "lower" if letters and all(ch.islower() for ch in letters) else "mixed" if letters else "none"
    return case, any(ch.isdigit() for ch in value), bool(re.search(r"[^\w]", value))


def classify_values(top_values: list[ValueFrequency]) -> tuple[list, list[ValueAnomaly]]:
    """Split observed values into (canonical values, anomalies). Values arrive most-frequent first."""
    vals = [v for v in top_values if v.value is not None]
    total = sum(v.count for v in vals) or 1
    canonical: list[ValueFrequency] = []
    anomalies: list[ValueAnomaly] = []
    for v in vals:
        s, share = str(v.value), v.count / total
        if not canonical:
            canonical.append(v)
            continue
        canon_strs = [str(c.value) for c in canonical]
        variant = _variant_of(s, canon_strs, rare=share < CANONICAL_MIN_SHARE)
        if variant:
            anomalies.append(ValueAnomaly(value=v.value, count=v.count, looks_like=variant[0], reason=variant[1]))
            continue
        styles = {_style(c) for c in canon_strs}
        if share >= CANONICAL_MIN_SHARE or (share >= STYLE_MIN_SHARE and _style(s) in styles):
            canonical.append(v)
        else:
            anomalies.append(ValueAnomaly(value=v.value, count=v.count,
                                          reason=f"rare ({share:.2%}) and unlike any valid value"))
    return [c.value for c in canonical], anomalies
