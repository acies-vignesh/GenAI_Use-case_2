"""Run a rule against the source (read-only) and report what it flags.

    result = execute_rule(engine, rule, model, schema)
    result.failed, result.failure_rate, result.passed, result.examples

Each check type becomes a query for its FAILING rows, built with SQLAlchemy Core so LIMIT/TOP,
quoting and parameters are right for every database. Rule expressions are parsed by sqlglot,
qualified with the table alias, and rendered in the source's SQL dialect.

Two checks run in Python because SQL can't express them portably:
  - pattern           (regex support differs per database) -> evaluated on the distinct values
  - hierarchy_acyclic (recursive SQL differs per database) -> evaluated on the (id, parent) pairs

Semantics (same as schemas/rule.py): a row fails only when the check is FALSE - NULL never fails,
except for not_null / null_rate_max. `filter` limits the rows in scope; for `conditional` the rows
in scope are those matching `when`.
"""
import re
from dataclasses import dataclass, field
from datetime import date

import sqlglot
from sqlglot import exp
from sqlalchemy import and_, case, column, exists, func, literal, literal_column, not_, or_, select, table, text, true
from sqlalchemy.engine import Engine

from app.extraction.profiler import mask
from app.schemas.rule import Rule
from app.schemas.semantic_model import Entity, SemanticModel

MAX_KEYS = 100_000           # failing keys collected when collect_keys=True
MAX_DISTINCT_FOR_PATTERN = 500_000

SQLGLOT_DIALECT = {"sqlite": "sqlite", "postgresql": "postgres", "mysql": "mysql", "mariadb": "mysql",
                   "mssql": "tsql", "oracle": "oracle"}


class ExecutionError(RuntimeError):
    pass


@dataclass
class ExecutionResult:
    rule_id: str
    total: int = 0                   # rows in scope
    failed: int = 0
    failure_rate: float = 0.0
    passed: bool = True              # failure_rate within tolerance (null_rate_max: within max_rate)
    examples: list[dict] = field(default_factory=list)
    failing_keys: list = field(default_factory=list)   # only with collect_keys=True
    error: str | None = None


# --------------------------------------------------------------------------- expression rendering
def render(expr: str, alias: str, dialect: str) -> str:
    """'ship_date >= order_date' -> 't.ship_date >= t.order_date' in the target SQL dialect."""
    tree = sqlglot.parse_one(expr)
    tree = tree.transform(lambda n: exp.column(n.name, table=alias) if isinstance(n, exp.Column) else n)
    return tree.sql(dialect=SQLGLOT_DIALECT.get(dialect, dialect))


class _Ctx:
    """Tables and helpers for one rule execution."""

    def __init__(self, engine: Engine, rule: Rule, model: SemanticModel, schema: str | None):
        self.engine, self.rule, self.model, self.schema = engine, rule, model, schema
        self.dialect = engine.dialect.name
        self.entity: Entity = model.entity(rule.target.entity)
        self.t = self.tbl(self.entity, "t")
        self.pk = self.entity.primary_key

    def tbl(self, entity: Entity, alias: str):
        return table(entity.table, *[column(a.column) for a in entity.attributes], schema=self.schema).alias(alias)

    def expr(self, text: str, alias: str = "t"):
        return literal_column(f"({render(text, alias, self.dialect)})")

    def scope(self):
        conds = [self.expr(self.rule.filter)] if self.rule.filter else []
        if self.rule.check.type == "conditional":
            conds.append(self.expr(self.rule.check.when))
        return and_(true(), *conds)


# --------------------------------------------------------------------------- per check type
def _failing(ctx: _Ctx):
    """(from_clause, failure condition) for SQL-expressible checks."""
    c, t, r = ctx.rule.check, ctx.t, ctx.rule
    col = t.c[r.target.columns[0]] if r.target.columns else None
    typ = c.type

    if typ in ("not_null", "null_rate_max"):
        return t, col.is_(None)
    if typ == "allowed_values":
        values = [str(v) for v in c.values]
        if c.case_sensitive:
            return t, and_(col.is_not(None), col.not_in(values))
        return t, and_(col.is_not(None), func.upper(col).not_in([v.upper() for v in values]))
    if typ == "range":
        parts = []
        if c.min is not None:
            parts.append(col < c.min if c.inclusive else col <= c.min)
        if c.max is not None:
            parts.append(col > c.max if c.inclusive else col >= c.max)
        return t, and_(col.is_not(None), or_(*parts))
    if typ == "length_range":
        length = literal_column(render(f"LENGTH({r.target.columns[0]})", "t", ctx.dialect))
        parts = ([length < c.min_length] if c.min_length is not None else []) + \
                ([length > c.max_length] if c.max_length is not None else [])
        return t, and_(col.is_not(None), or_(*parts))
    if typ == "date_not_future":
        return t, and_(col.is_not(None), col > date.today())
    if typ == "not_placeholder":
        return t, col.in_([str(v) for v in c.values])
    if typ == "unique":
        cols = r.target.columns
        dupes = (select(*[t.c[x] for x in cols]).where(and_(*[t.c[x].is_not(None) for x in cols]))
                 .group_by(*[t.c[x] for x in cols]).having(func.count() > 1).subquery("d"))
        joined = t.join(dupes, and_(*[t.c[x] == dupes.c[x] for x in cols]))
        return joined, true()
    if typ == "row_condition":
        return t, not_(ctx.expr(c.expression))
    if typ == "conditional":
        return t, not_(ctx.expr(c.then))
    if typ == "foreign_key":
        p = ctx.tbl(ctx.model.entity(c.ref_entity), "p")
        match = and_(*[p.c[pc] == t.c[cc] for cc, pc in zip(r.target.columns, c.ref_columns)])
        not_null = and_(*[t.c[cc].is_not(None) for cc in r.target.columns])
        return t, and_(not_null, ~exists(select(literal(1)).select_from(p).where(match)))
    if typ in ("has_children", "aggregate_compare"):
        # Aggregate the child table ONCE (GROUP BY) and join it - a correlated subquery would re-scan the
        # child table for every parent row (24k orders x 49k lines without an index = minutes).
        ch = ctx.tbl(ctx.model.entity(c.child_entity), "ch")
        keys = [ch.c[cj].label(f"k{i}") for i, cj in enumerate(c.child_join_columns)]
        value = func.count() if typ == "has_children" else \
            literal_column(render(c.child_expression, "ch", ctx.dialect))
        agg = select(*keys, value.label("agg")).select_from(ch).group_by(*[ch.c[cj] for cj in c.child_join_columns]) \
            .subquery("a")
        joined = t.outerjoin(agg, and_(*[agg.c[f"k{i}"] == t.c[pj] for i, pj in enumerate(c.parent_join_columns)]))
        if typ == "has_children":
            return joined, func.coalesce(agg.c.agg, 0) < c.min_children
        parent = ctx.expr(c.parent_expression)
        if c.operator == "=":
            return joined, func.abs(parent - agg.c.agg) > c.tolerance_abs
        return joined, not_(parent.op(c.operator)(agg.c.agg))
    if typ == "hierarchy_parent_level":
        h = ctx.model.hierarchy(c.hierarchy)
        p = ctx.tbl(ctx.entity, "p")
        lvl, parent, pk = h.level_attribute, h.parent_attribute, ctx.pk[0]
        expected = case({lv: h.levels[i - 1] for i, lv in enumerate(h.levels) if i > 0}, value=t.c[lvl])
        top_wrong = and_(t.c[lvl] == h.levels[0], t.c[parent].is_not(None))
        child_wrong = and_(t.c[lvl].in_(h.levels[1:]),
                           or_(t.c[parent].is_(None), p.c[lvl].is_(None), p.c[lvl] != expected))
        unknown_level = or_(t.c[lvl].is_(None), t.c[lvl].not_in(h.levels))
        return t.outerjoin(p, p.c[pk] == t.c[parent]), or_(top_wrong, child_wrong, unknown_level)
    if typ == "hierarchy_attach_level":
        h = ctx.model.hierarchy(c.hierarchy)
        node_entity = ctx.model.entity(h.entity)
        n = ctx.tbl(node_entity, "n")
        return t.join(n, n.c[node_entity.primary_key[0]] == col), n.c[h.level_attribute] != c.level
    raise ExecutionError(f"no SQL translation for '{typ}'")


def _example(row, ctx: _Ctx) -> dict:
    out = {}
    for k, v in row._mapping.items():
        attr = next((a for a in ctx.entity.attributes if a.column == k), None)
        out[k] = mask(str(v)) if (attr is not None and attr.pii and v is not None) else v
    return out


def _run_sql(ctx: _Ctx, collect_keys: bool, max_examples: int, res: ExecutionResult) -> None:
    from_clause, fail = _failing(ctx)
    t = ctx.t
    scope = ctx.scope()
    shown = list(dict.fromkeys(ctx.pk + ctx.rule.target.columns))
    with ctx.engine.connect() as conn:
        res.total = conn.execute(select(func.count()).select_from(t).where(scope)).scalar_one()
        res.failed = conn.execute(select(func.count()).select_from(from_clause).where(scope, fail)).scalar_one()
        if res.failed:
            rows = conn.execute(select(*[t.c[x] for x in shown]).select_from(from_clause)
                                .where(scope, fail).limit(max_examples))
            res.examples = [_example(r, ctx) for r in rows]
            if collect_keys and len(ctx.pk) == 1:
                res.failing_keys = list(conn.execute(select(t.c[ctx.pk[0]]).select_from(from_clause)
                                                     .where(scope, fail).limit(MAX_KEYS)).scalars())


def _run_pattern(ctx: _Ctx, collect_keys: bool, max_examples: int, res: ExecutionResult) -> None:
    t, colname = ctx.t, ctx.rule.target.columns[0]
    col = t.c[colname]
    rx = re.compile(ctx.rule.check.regex)
    scope = and_(ctx.scope(), col.is_not(None))
    with ctx.engine.connect() as conn:
        res.total = conn.execute(select(func.count()).select_from(t).where(ctx.scope())).scalar_one()
        counts = conn.execute(select(col, func.count()).where(scope).group_by(col)
                              .limit(MAX_DISTINCT_FOR_PATTERN)).all()
        bad = [v for v, _ in counts if not rx.search(str(v))]
        res.failed = sum(n for v, n in counts if not rx.search(str(v)))
        if bad:
            shown = list(dict.fromkeys(ctx.pk + [colname]))
            rows = conn.execute(select(*[t.c[x] for x in shown]).where(scope, col.in_(bad[:1000]))
                                .limit(max_examples))
            res.examples = [_example(r, ctx) for r in rows]
            if collect_keys and len(ctx.pk) == 1:
                res.failing_keys = [k for i in range(0, len(bad), 900) for k in conn.execute(
                    select(t.c[ctx.pk[0]]).where(scope, col.in_(bad[i:i + 900]))).scalars()][:MAX_KEYS]


def _run_acyclic(ctx: _Ctx, collect_keys: bool, max_examples: int, res: ExecutionResult) -> None:
    h = ctx.model.hierarchy(ctx.rule.check.hierarchy)
    t, pk = ctx.t, ctx.pk[0]
    with ctx.engine.connect() as conn:
        parent_of = dict(conn.execute(select(t.c[pk], t.c[h.parent_attribute])).all())
    in_cycle: set = set()
    for start in parent_of:
        seen, node = [], start
        while node is not None and node in parent_of and node not in seen:
            seen.append(node)
            node = parent_of[node]
        if node is not None and node in seen:
            in_cycle |= set(seen[seen.index(node):])
    res.total, res.failed = len(parent_of), len(in_cycle)
    res.examples = [{pk: k, h.parent_attribute: parent_of[k]} for k in sorted(in_cycle)[:max_examples]]
    if collect_keys:
        res.failing_keys = sorted(in_cycle)


def _run_custom(ctx: _Ctx, collect_keys: bool, max_examples: int, res: ExecutionResult) -> None:
    """Steward-scrutinised escape hatch: a SELECT returning failing rows (validated as a single SELECT)."""
    sql = sqlglot.transpile(ctx.rule.check.sql, write=SQLGLOT_DIALECT.get(ctx.dialect, ctx.dialect))[0]
    with ctx.engine.connect() as conn:
        res.total = conn.execute(select(func.count()).select_from(ctx.t)).scalar_one()
        res.failed = conn.execute(text(f"SELECT COUNT(*) FROM ({sql}) x")).scalar_one()
        res.examples = [dict(r._mapping) for r in conn.execute(text(sql)).fetchmany(max_examples)]


def execute_rule(engine: Engine, rule: Rule, model: SemanticModel, schema: str | None = None,
                 collect_keys: bool = False, max_examples: int = 5) -> ExecutionResult:
    res = ExecutionResult(rule.rule_id)
    try:
        ctx = _Ctx(engine, rule, model, schema)
        runner = {"pattern": _run_pattern, "hierarchy_acyclic": _run_acyclic,
                  "custom_sql": _run_custom}.get(rule.check.type, _run_sql)
        runner(ctx, collect_keys, max_examples, res)
    except Exception as e:   # a broken rule must not break a review session
        res.error = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
        res.passed = False
        return res
    res.failure_rate = round(res.failed / res.total, 6) if res.total else 0.0
    if rule.check.type == "null_rate_max":
        res.passed = res.failure_rate <= rule.check.max_rate
    else:
        res.passed = res.failure_rate <= rule.tolerance
    return res
