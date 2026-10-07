"""Baseline rules derived by code from the semantic model - no LLM needed.

These follow directly from facts the model already holds (keys, relationships, allowed values,
patterns, hierarchies, semantic types). Generating them deterministically is cheaper and more
reliable than asking the LLM, and frees the LLM to focus on business, cross-column and aggregate
rules that code can't infer. Every rationale cites the profile evidence behind it.
"""
from app.schemas.metadata import SourceMetadata
from app.schemas.rule import Rule
from app.schemas.semantic_model import Attribute, Entity, EntityType, SemanticModel, SemanticType as ST

PLACEHOLDER_DATES = ["1900-01-01", "1899-12-31", "1970-01-01", "9999-12-31"]
NUMERIC_FLOORS = {  # semantic type -> (min, inclusive) that holds by definition
    ST.PRICE: (0, False),
    ST.COST: (0, True),
    ST.QUANTITY: (0, False),
}


def baseline_rules(model: SemanticModel, metadata: SourceMetadata, source_id: str) -> list[Rule]:
    rules: list[Rule] = []
    entities = {e.entity_id: e for e in model.entities}

    def add(entity: Entity, name: str, category: str, dimension: str, scope: str, columns: list[str],
            check: dict, severity: str = "medium", rationale: str = "", evidence: list[str] | None = None,
            filter_: str | None = None) -> None:
        rules.append(Rule.model_validate(dict(
            source_id=source_id, name=name, category=category, dimension=dimension, scope=scope,
            target={"entity": entity.entity_id, "columns": columns}, check=check, severity=severity,
            rationale=rationale, evidence=evidence or [], filter=filter_, origin="heuristic")))

    for e in model.entities:
        t = metadata.table(e.table)
        fks = {fk.columns[0]: fk for fk in t.foreign_keys if len(fk.columns) == 1}

        for key in e.business_key:
            p = t.column(key).profile
            add(e, f"{_label(e, key)} is unique", "data_quality", "uniqueness", "column", [key],
                {"type": "unique"}, "high", f"{key} is the business key of {e.name}; "
                f"{p.distinct_count:,} distinct of {p.row_count - p.null_count:,} values.",
                [f"profile.{e.table}.{key}.uniqueness"])

        for rel in model.relationships:
            if rel.from_entity != e.entity_id:
                continue
            col = rel.from_attributes[0]
            fk = fks.get(col)
            orphans = f"{fk.orphan_count} orphan rows found" if fk and fk.orphan_count is not None else \
                "relationship inferred from names and values"
            target = entities[rel.to_entity]
            add(e, f"{_label(e, col)} must exist in {target.name}", "data_quality",
                "referential_integrity", "cross_table", rel.from_attributes,
                {"type": "foreign_key", "ref_entity": rel.to_entity, "ref_columns": rel.to_attributes},
                "high", f"{rel.description}; {orphans}.", [f"relationship.{rel.relationship_id}"])
            if rel.mandatory:
                p = t.column(col).profile
                add(e, f"{_label(e, col)} is mandatory", "data_quality", "completeness", "column",
                    [col], {"type": "not_null"}, "high",
                    f"Every {e.name.lower()} must have a {target.name.lower()}; {p.null_count} rows are empty.",
                    [f"profile.{e.table}.{col}.null_count"])

        for a in e.attributes:
            _attribute_rules(e, a, t.column(a.column).profile, add)

    for hy in model.hierarchies:
        node = entities[hy.entity]
        evidence = [f"hierarchy.{hy.hierarchy_id}.evidence"]
        add(node, f"{hy.name}: each node's parent is one level up", "hierarchy", "consistency", "table", [],
            {"type": "hierarchy_parent_level", "hierarchy": hy.hierarchy_id}, "high",
            f"Levels {' > '.join(hy.levels)}. Observed: " + ("; ".join(hy.evidence) or "no violations"), evidence)
        add(node, f"{hy.name} has no cycles", "hierarchy", "consistency", "table", [],
            {"type": "hierarchy_acyclic", "hierarchy": hy.hierarchy_id}, "critical",
            "A node that is its own ancestor breaks every roll-up through the hierarchy.", evidence)
        for att in hy.attached_entities:
            ent = entities[att.entity]
            add(ent, f"{ent.name} is assigned at {att.must_attach_at} level of {hy.name}", "hierarchy",
                "consistency", "cross_table", [att.attribute],
                {"type": "hierarchy_attach_level", "hierarchy": hy.hierarchy_id, "level": att.must_attach_at},
                "high", f"{ent.name} rows should point to {att.must_attach_at} nodes "
                        f"(the dominant level in the data).", evidence)
    return rules


def _label(e: Entity, column: str) -> str:
    return e.attribute(column).name or column


def _attribute_rules(e: Entity, a: Attribute, p, add) -> None:
    label = a.name or a.column
    ev = [f"profile.{e.table}.{a.column}"]

    if a.allowed_values and a.semantic_type in (ST.STATUS_CODE, ST.CATEGORY_CODE, ST.CURRENCY_CODE,
                                                 ST.COUNTRY_CODE, ST.HIERARCHY_LEVEL):
        bad = sum(x.count for x in a.value_anomalies)
        found = (f" {bad} rows hold other values: " + ", ".join(repr(x.value) for x in a.value_anomalies[:6])
                 if a.value_anomalies else "")
        add(e, f"{label} is a valid code", "data_quality", "validity", "column", [a.column],
            {"type": "allowed_values", "values": a.allowed_values.values},
            "high" if a.semantic_type == ST.STATUS_CODE else "medium",
            f"{label} is a {a.semantic_type.value.replace('_', ' ')}.{found}", ev)

    if a.expected_pattern and a.semantic_type in (ST.BUSINESS_KEY, ST.EMAIL, ST.PHONE, ST.POSTAL_CODE):
        share = f"{p.patterns[0].share:.1%} of values follow the dominant shape" if p and p.patterns else ""
        add(e, f"{label} is well-formed", "data_quality", "validity", "column", [a.column],
            {"type": "pattern", "regex": a.expected_pattern}, "medium", share, ev)

    if a.semantic_type == ST.BIRTH_DATE:
        n = p.date.placeholder_count if p and p.date else 0
        add(e, f"{label} is not a placeholder date", "data_quality", "accuracy", "column", [a.column],
            {"type": "not_placeholder", "values": PLACEHOLDER_DATES}, "medium",
            f"Default dates like 1900-01-01 hide missing values; {n} found.", ev)

    if a.semantic_type == ST.EVENT_DATE and e.entity_type in (EntityType.TRANSACTION, EntityType.TRANSACTION_LINE):
        n = p.date.future_count if p and p.date else 0
        add(e, f"{label} is not in the future", "data_quality", "timeliness", "column", [a.column],
            {"type": "date_not_future"}, "medium", f"A recorded event can't be in the future; {n} found.", ev)

    if a.semantic_type in NUMERIC_FLOORS:
        lo, inclusive = NUMERIC_FLOORS[a.semantic_type]
        neg = (p.numeric.negative_count + (0 if inclusive else p.numeric.zero_count)) if p and p.numeric else 0
        add(e, f"{label} is {'non-negative' if inclusive else 'positive'}", "data_quality", "validity", "column",
            [a.column], {"type": "range", "min": lo, "inclusive": inclusive}, "high",
            f"A {a.semantic_type.value} below {'zero' if inclusive else 'or equal to zero'} is meaningless; "
            f"{neg} found.", ev)

    if a.semantic_type == ST.PERCENTAGE:
        add(e, f"{label} is between 0 and 100", "data_quality", "validity", "column", [a.column],
            {"type": "range", "min": 0, "max": 100}, "medium", "Percentages outside 0-100 are invalid.", ev)
