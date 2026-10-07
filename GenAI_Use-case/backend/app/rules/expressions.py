"""Parse, validate, normalise and transpile the small SQL expressions used inside rules.

Expressions are written in plain ANSI-style SQL over unqualified column names of one entity:
    ship_date >= order_date
    status = 'DELIVERED'
    ABS(line_amount - ROUND(quantity * unit_price * (1 - discount_pct / 100.0), 2)) <= 0.01
    signup_date >= DATE_ADD(date_of_birth, 18, 'YEAR')        <- portable date arithmetic form
    SUM(line_amount)                                           <- aggregates only where asked for

sqlglot turns the text into a syntax tree, which lets us (a) reject anything that is not a plain
expression (no SELECT/DROP/subqueries), (b) check every column exists, (c) compare two expressions
regardless of spacing/case, and (d) generate the right SQL for each database.
"""
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


class ExpressionError(ValueError):
    pass


def parse_expression(text: str) -> exp.Expression:
    try:
        trees = sqlglot.parse(text)
    except ParseError as e:
        raise ExpressionError(f"cannot parse '{text}': {e.errors[0]['description'] if e.errors else e}") from e
    if len(trees) != 1 or trees[0] is None:
        raise ExpressionError(f"'{text}' must be a single expression")
    tree = trees[0]
    if not isinstance(tree, exp.Condition) or tree.find(exp.Query, exp.Subquery):
        raise ExpressionError(f"'{text}' must be a plain expression (no SELECT, subqueries or statements)")
    return tree


def column_names(tree: exp.Expression) -> set[str]:
    return {c.name.lower() for c in tree.find_all(exp.Column)}


def validate_expression(text: str, allowed_columns: set[str], aggregate: bool | None = False) -> exp.Expression:
    """aggregate=False: row expression (no SUM/MIN...); True: must aggregate; None: either."""
    tree = parse_expression(text)
    qualified = [c.sql() for c in tree.find_all(exp.Column) if c.table]
    if qualified:
        raise ExpressionError(f"'{text}': use bare column names, not {qualified}")
    unknown = sorted(column_names(tree) - {c.lower() for c in allowed_columns})
    if unknown:
        raise ExpressionError(f"'{text}': unknown column(s) {unknown}")
    has_agg = tree.find(exp.AggFunc) is not None
    if aggregate is True and not has_agg:
        raise ExpressionError(f"'{text}' must be an aggregate (SUM, MIN, MAX, COUNT, AVG)")
    if aggregate is False and has_agg:
        raise ExpressionError(f"'{text}' must not contain aggregates")
    return tree


def normalize(text: str) -> str:
    """Canonical text: 'Ship_Date>=order_date' and 'ship_date >= ORDER_DATE' give the same result."""
    return sqlglot.parse_one(text).sql(normalize=True)


def to_dialect(text: str, dialect: str) -> str:
    """Render an expression for a target database: 'sqlite', 'postgres', 'tsql', 'mysql', 'oracle'..."""
    return sqlglot.parse_one(text).sql(dialect=dialect)
