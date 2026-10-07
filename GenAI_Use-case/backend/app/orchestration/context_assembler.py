"""Gather the 4 inputs: semantic layer + business context + user rules + approved/rejected history.

Produces an EntityContext per target entity: plain-text blocks the prompt template lays out.
Privacy: only aggregated statistics, valid codes, anomalies of code columns, value SHAPES and masked
examples - never raw rows or raw values of personal-data columns.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.persistence.models import ContextItemRow
from app.persistence.repositories import PromptHistory
from app.schemas.metadata import ColumnMetadata, SourceMetadata
from app.schemas.rule import Rule
from app.schemas.semantic_model import Attribute, Entity, SemanticModel


@dataclass
class EntityContext:
    entity_id: str
    entity_block: str
    related_block: str
    hierarchy_block: str
    business_context: list[str] = field(default_factory=list)
    covered: list[str] = field(default_factory=list)       # baseline + approved: don't repeat
    rejected: list[str] = field(default_factory=list)      # with reasons: don't repeat or vary
    preferences: list[str] = field(default_factory=list)   # steward edits: suggested X -> approved Y
    patterns: list[str] = field(default_factory=list)      # what KINDS of rules the steward accepts/rejects


def _share(x: float) -> str:
    return f"{x:.1%}" if x >= 0.001 or x == 0 else f"{x:.2%}"


def attribute_facts(a: Attribute, col: ColumnMetadata) -> str:
    p = col.profile
    head = f"- {a.column} ({col.generic_type.value}, {a.semantic_type.value}, {a.role.value}"
    head += ", PII)" if a.pii else ")"
    parts = [head + (f": {a.description}" if a.description else "")]
    if p:
        stats = f"nulls {_share(p.null_rate)}, distinct {p.distinct_count:,}"
        if p.uniqueness is not None and p.uniqueness >= 0.95:
            stats += f", uniqueness {p.uniqueness:.4f}"
        parts.append(stats)
        if a.allowed_values:
            parts.append("valid values: " + ", ".join(map(str, a.allowed_values.values)))
        if a.value_anomalies:
            parts.append("anomalies: " + ", ".join(
                f"{x.value!r} x{x.count}" + (f" (looks like {x.looks_like!r})" if x.looks_like is not None else "")
                for x in a.value_anomalies))
        if p.numeric and p.numeric.min is not None:
            n = p.numeric
            parts.append(f"range {n.min:g}..{n.max:g}, p01 {n.p01}, median {n.median}, p99 {n.p99}, "
                         f"{n.negative_count} negative, {n.zero_count} zero")
        if p.date and p.date.min:
            d = p.date
            parts.append(f"dates {d.min}..{d.max}, {d.future_count} future, {d.placeholder_count} placeholder")
        if p.patterns and not a.allowed_values:
            parts.append("shapes " + ", ".join(f"{pt.pattern} {_share(pt.share)}" for pt in p.patterns[:3]))
        if p.examples and not a.allowed_values:
            parts.append("e.g. " + ", ".join(p.examples[:2]))   # masked for personal data
    if a.expected_pattern:
        parts.append(f"expected pattern {a.expected_pattern}")
    return " | ".join(parts)


def rule_line(r: Rule) -> str:
    c = r.check.model_dump(exclude={"type"}, exclude_none=True)
    params = ", ".join(f"{k}={v}" for k, v in c.items())
    cols = ",".join(r.target.columns) or "-"
    flt = f" WHERE {r.filter}" if r.filter else ""
    return f"{r.name} [{r.check.type} on {r.target.entity}({cols}){(': ' + params) if params else ''}{flt}]"


PATTERN_MIN_DECISIONS = 2


def decision_patterns(history: PromptHistory, entity_id: str | None) -> list[str]:
    """Summarise decisions per check type, e.g. 'not_null: rejected 5 of 6 (already_enforced x5)'.

    entity_id=None summarises the whole source (only clear signals: 3+ rejections of one kind).
    """
    approved: Counter = Counter()
    rejected: Counter = Counter()
    codes: dict[str, Counter] = defaultdict(Counter)
    example: dict[str, str] = {}
    for r in history.approved:
        if entity_id is None or r.target.entity == entity_id:
            approved[r.check.type] += 1
    for r, reason in history.rejected:
        if entity_id is None or r.target.entity == entity_id:
            rejected[r.check.type] += 1
            codes[r.check.type][history.reason_codes.get(r.rule_id) or "no code"] += 1
            if reason:
                example[r.check.type] = reason
    lines = []
    for typ in sorted(set(approved) | set(rejected)):
        a, rj = approved[typ], rejected[typ]
        if entity_id is None and rj < 3:
            continue
        if a + rj < PATTERN_MIN_DECISIONS and not rj:
            continue
        line = f"{typ}: approved {a}, rejected {rj}"
        if rj:
            line += " (" + ", ".join(f"{c} x{n}" for c, n in codes[typ].most_common()) + ")"
            if typ in example:
                line += f" - e.g. \"{example[typ]}\""
        lines.append(line)
    return lines


def _unique(rules: list[Rule]) -> list[Rule]:
    seen: set[str] = set()
    return [r for r in rules if not (r.fingerprint in seen or seen.add(r.fingerprint))]


def build_entity_context(model: SemanticModel, metadata: SourceMetadata, entity_id: str,
                         context_items: list[ContextItemRow], history: PromptHistory,
                         baseline: list[Rule]) -> EntityContext:
    e: Entity = model.entity(entity_id)
    t = metadata.table(e.table)

    lines = [f"ENTITY {e.entity_id} - {e.name} (table {e.table}, {t.row_count:,} rows, {e.entity_type.value})",
             f"Grain: {e.grain}"]
    if e.description:
        lines.append(f"Description: {e.description}")
    lines.append(f"Primary key: {', '.join(e.primary_key)}; business key: {', '.join(e.business_key) or '-'}")
    lines.append("Attributes:")
    lines += [attribute_facts(a, t.column(a.column)) for a in e.attributes]

    fk_orphans = {fk.columns[0]: fk.orphan_count for fk in t.foreign_keys if len(fk.columns) == 1}
    rels = []
    for r in model.relationships:
        if r.from_entity == entity_id:
            orphans = fk_orphans.get(r.from_attributes[0])
            rels.append(f"- {entity_id}.{','.join(r.from_attributes)} -> {r.to_entity}.{','.join(r.to_attributes)}"
                        f" ({'mandatory' if r.mandatory else 'optional'}"
                        + (f", {orphans} orphan rows" if orphans is not None else "") + f"): {r.description}")
        elif r.to_entity == entity_id:
            rels.append(f"- {r.from_entity}.{','.join(r.from_attributes)} -> {entity_id} (child rows): {r.description}")
    if rels:
        lines.append("Relationships:")
        lines += rels

    related_ids = {r.to_entity for r in model.relationships if r.from_entity == entity_id} | \
                  {r.from_entity for r in model.relationships if r.to_entity == entity_id}
    related_ids.discard(entity_id)
    related = [f"- {o.entity_id} ({o.table}, {o.entity_type.value}): "
               + ", ".join(f"{a.column}:{a.semantic_type.value}" for a in o.attributes)
               for o in model.entities if o.entity_id in related_ids]

    hierarchies = []
    for h in model.hierarchies:
        attached = [a for a in h.attached_entities if a.entity == entity_id]
        if h.entity == entity_id or attached:
            hierarchies.append(f"- {h.hierarchy_id} ({h.name}): {' > '.join(h.levels)} on {h.entity}"
                               f" via {h.parent_attribute}, level column {h.level_attribute}")
            hierarchies += [f"  {a.entity}.{a.attribute} must point to a {a.must_attach_at}" for a in h.attached_entities]
            hierarchies += [f"  observed: {ev}" for ev in h.evidence]

    def relevant(r: Rule) -> bool:
        return r.target.entity == entity_id

    ctx = [f"- [{'.'.join(x for x in (c.entity, c.attribute) if x) or 'whole source'}] {c.text}"
           for c in context_items if c.entity in (None, entity_id)]

    return EntityContext(
        entity_id=entity_id,
        entity_block="\n".join(lines),
        related_block="\n".join(related) or "(none)",
        hierarchy_block="\n".join(hierarchies) or "(none)",
        business_context=ctx,
        covered=[rule_line(r) for r in _unique(baseline + history.approved + history.pending) if relevant(r)],
        rejected=[rule_line(r) + (f" - steward's reason: {reason}" if reason else "")
                  for r, reason in history.rejected if relevant(r)],
        preferences=[f"suggested {rule_line(orig)} -> steward approved {rule_line(new)}"
                     for orig, new in history.superseded if relevant(new)],
        patterns=[f"[{entity_id}] {ln}" for ln in decision_patterns(history, entity_id)]
                 + [f"[whole source] {ln}" for ln in decision_patterns(history, None)],
    )
