"""Declared + inferred relationships (FK, name matching, value overlap) and parent-child hierarchies.

Unlike the profiler, these need to look at how tables relate to each other, so they run a few light
SQL queries against the source:
  - inferred FK      : what share of a column's distinct values exist in the candidate parent key?
  - hierarchy levels : for each child level, which level is its parent?   (ZONE->STATE->CITY)
  - attach level     : at which level do rows of another table point into the hierarchy?
"""
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from sqlalchemy import case, column, distinct, func, select, table
from sqlalchemy.engine import Engine

from app.schemas.metadata import GenericType, SourceMetadata, TableMetadata

INFER_MIN_CONTAINMENT = 0.9      # >= 90% of child values must exist in the parent key
ATTACH_MIN_SHARE = 0.9           # dominant level must cover >= 90% of pointers
LINK_MIN_SHARE = 0.5             # a level is "child of" another when >= 50% of its rows say so
_ROOT, _MISSING = "__root__", "__missing__"


@dataclass
class Link:
    child_table: str
    child_columns: list[str]
    parent_table: str
    parent_columns: list[str]
    kind: str                         # declared | inferred
    confidence: float = 1.0
    evidence: str = ""


@dataclass
class LevelStructure:
    levels: list[str]
    evidence: list[str] = field(default_factory=list)


def singular(name: str) -> str:
    n = name.lower()
    if n.endswith("ies"):
        return n[:-3] + "y"
    if n.endswith(("ses", "xes", "ches", "shes")):
        return n[:-2]
    if n.endswith("s") and not n.endswith("ss"):
        return n[:-1]
    return n


def entity_id_for(table_name: str) -> str:
    eid = re.sub(r"[^a-z0-9_]", "_", singular(table_name))
    return eid if eid[:1].isalpha() else f"t_{eid}"


# --------------------------------------------------------------------------- relationships
def declared_links(metadata: SourceMetadata) -> list[Link]:
    return [
        Link(t.name, fk.columns, fk.referred_table, fk.referred_columns, "declared",
             evidence=f"declared foreign key; {fk.orphan_count or 0} orphan rows")
        for t in metadata.tables for fk in t.foreign_keys
    ]


def _name_candidates(child: TableMetadata, metadata: SourceMetadata, taken: set[tuple[str, str]]):
    """Yield (child column, parent table, parent pk column) pairs whose names suggest a link."""
    for parent in metadata.tables:
        if len(parent.primary_key) != 1:
            continue
        pk = parent.primary_key[0]
        stem = singular(parent.name)
        names = {pk.lower(), f"{stem}_{pk}".lower(), f"{stem}_id"}
        prefix = stem[:3] if len(stem) >= 3 else None
        for col in child.columns:
            n = col.name.lower()
            if col.is_primary_key or (child.name, col.name) in taken or parent.name == child.name and n == pk:
                continue
            matches = n in names or (prefix and n.endswith("_id") and n != "id"
                                     and stem.startswith(n[:-3]) and n[:-3].startswith(prefix))
            if matches and _compatible(col.generic_type, parent.column(pk).generic_type):
                yield col.name, parent, pk


def _compatible(a: GenericType, b: GenericType) -> bool:
    numeric = {GenericType.INTEGER, GenericType.DECIMAL}
    return a == b or (a in numeric and b in numeric)


def infer_links(engine: Engine, metadata: SourceMetadata, known: list[Link]) -> list[Link]:
    taken = {(lk.child_table, c) for lk in known for c in lk.child_columns}
    schema = metadata.schema_name
    found = []
    for child in metadata.tables:
        for col, parent, pk in _name_candidates(child, metadata, taken):
            c = table(child.name, column(col), schema=schema)
            p = table(parent.name, column(pk), schema=schema)
            with engine.connect() as conn:
                total = conn.execute(select(func.count(distinct(c.c[col])))).scalar_one()
                matched = conn.execute(
                    select(func.count(distinct(c.c[col]))).where(c.c[col].in_(select(p.c[pk])))
                ).scalar_one()
            if total < 2:
                continue
            containment = matched / total
            if containment >= INFER_MIN_CONTAINMENT:
                found.append(Link(child.name, [col], parent.name, [pk], "inferred",
                                  confidence=round(0.9 * containment, 2),
                                  evidence=f"name match; {containment:.1%} of {total:,} distinct values "
                                           f"exist in {parent.name}.{pk}"))
                taken.add((child.name, col))
    return found


# --------------------------------------------------------------------------- hierarchies
def hierarchy_levels(engine: Engine, t: TableMetadata, schema: str | None, parent_col: str,
                     level_col: str) -> LevelStructure | None:
    pk = t.primary_key[0]
    c = table(t.name, column(pk), column(parent_col), column(level_col), schema=schema).alias("c")
    p = table(t.name, column(pk), column(level_col), schema=schema).alias("p")
    parent_level = case((c.c[parent_col].is_(None), _ROOT), else_=func.coalesce(p.c[level_col], _MISSING))
    stmt = (select(c.c[level_col], parent_level, func.count())
            .select_from(c.outerjoin(p, c.c[parent_col] == p.c[pk]))
            .group_by(c.c[level_col], parent_level))
    with engine.connect() as conn:
        counts = {(str(lv), str(pl)): n for lv, pl, n in conn.execute(stmt) if lv is not None}

    totals: Counter = Counter()
    by_parent: dict[str, Counter] = defaultdict(Counter)
    for (lv, pl), n in counts.items():
        totals[lv] += n
        by_parent[lv][pl] += n

    roots = [lv for lv in totals if by_parent[lv][_ROOT] / totals[lv] >= LINK_MIN_SHARE]
    if len(roots) != 1:
        return None
    chain = roots
    while True:
        nxt = [lv for lv in totals if lv not in chain and by_parent[lv][chain[-1]] / totals[lv] >= LINK_MIN_SHARE]
        if len(nxt) != 1:
            break
        chain.append(nxt[0])
    if len(chain) < 2:
        return None

    evidence = []
    for i, lv in enumerate(chain):
        expected = _ROOT if i == 0 else chain[i - 1]
        for pl, n in by_parent[lv].items():
            if pl == expected:
                continue
            what = ("have no parent" if pl == _ROOT else "point to a missing parent" if pl == _MISSING
                    else f"have a {pl} parent (expected {expected if i else 'none'})")
            evidence.append(f"{n} {lv} row(s) {what}")
    stray = [lv for lv in totals if lv not in chain]
    if stray:
        evidence.append(f"level value(s) outside the hierarchy: {stray}")
    return LevelStructure(chain, evidence)


def attach_level(engine: Engine, child: str, child_col: str, node: TableMetadata, level_col: str,
                 schema: str | None) -> tuple[str, list[str]] | None:
    pk = node.primary_key[0]
    c = table(child, column(child_col), schema=schema)
    n = table(node.name, column(pk), column(level_col), schema=schema)
    stmt = (select(n.c[level_col], func.count()).select_from(c.join(n, c.c[child_col] == n.c[pk]))
            .group_by(n.c[level_col]))
    with engine.connect() as conn:
        counts = {str(lv): k for lv, k in conn.execute(stmt)}
    total = sum(counts.values())
    if not total:
        return None
    level, top = max(counts.items(), key=lambda kv: kv[1])
    if top / total < ATTACH_MIN_SHARE:
        return None
    evidence = [f"{k} {child} row(s) point to a {lv} instead of a {level}" for lv, k in counts.items() if lv != level]
    return level, evidence
