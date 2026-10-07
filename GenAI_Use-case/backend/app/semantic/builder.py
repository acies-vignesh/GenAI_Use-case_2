"""Orchestrates heuristics + relationships + enrichment into a SemanticModel and writes the JSON.

    metadata ─► links (declared + inferred) ─► attributes (heuristics) ─► hierarchies ─► entity types
             ─► [LLM enrichment] ─► merge with previous version (steward locks win) ─► validate
"""
from dataclasses import dataclass, field

from sqlalchemy.engine import Engine

from app.agent.llm_client import LLMClient
from app.schemas.metadata import GenericType, SourceMetadata, TableMetadata
from app.schemas.semantic_model import (
    AllowedValues,
    AttachedEntity,
    Attribute,
    AttributeRole,
    Domain,
    Entity,
    EntityType,
    Hierarchy,
    Provenance,
    Relationship,
    SemanticModel,
    SemanticType as ST,
    ValueRange,
)
from app.semantic import heuristics as h
from app.semantic.enrichment import enrich
from app.semantic.relationship_mapper import (
    Link,
    attach_level,
    declared_links,
    entity_id_for,
    hierarchy_levels,
    infer_links,
)

MANDATORY_MAX_NULL_RATE = 0.02   # an FK with <= 2% nulls is treated as mandatory (the nulls are errors)
REFERENCE_MAX_ROWS = 1000


@dataclass
class BuildReport:
    llm_used: bool = False
    llm_changes: list[str] = field(default_factory=list)
    inferred_links: list[str] = field(default_factory=list)
    needs_review: list[str] = field(default_factory=list)
    drift: list[str] = field(default_factory=list)
    kept_locked: list[str] = field(default_factory=list)


def build_semantic_model(engine: Engine, metadata: SourceMetadata, llm: LLMClient | None = None,
                         previous: SemanticModel | None = None) -> tuple[SemanticModel, BuildReport]:
    if metadata.profiled_at is None:
        raise ValueError("metadata must be profiled first (python -m app.cli profile)")
    report = BuildReport()
    ids = {t.name: entity_id_for(t.name) for t in metadata.tables}

    links = declared_links(metadata)
    inferred = infer_links(engine, metadata, links)
    links += inferred
    report.inferred_links = [f"{lk.child_table}.{lk.child_columns[0]} -> {lk.parent_table} ({lk.evidence})"
                             for lk in inferred]
    fk_by_col = {(lk.child_table, lk.child_columns[0]): lk for lk in links if len(lk.child_columns) == 1}

    entities = [_build_entity(t, ids[t.name], fk_by_col) for t in metadata.tables]
    relationships = [_relationship(lk, ids, metadata) for lk in links]
    hierarchies = _build_hierarchies(engine, metadata, entities, links, ids)
    _assign_entity_types(entities, links, hierarchies, metadata, ids)

    model = SemanticModel(
        source_id=metadata.source_id,
        based_on_metadata=metadata.extracted_at,
        domain=Domain(name=metadata.database),
        entities=entities, relationships=relationships, hierarchies=hierarchies,
    )

    if llm is not None:
        report.llm_used = True
        enrich(model, metadata, llm, report.llm_changes)
    if previous is not None:
        model = _merge_previous(model, previous, report)

    report.needs_review = [f"{e.entity_id}.{a.column} = {a.semantic_type.value} ({a.provenance.confidence:.2f})"
                           for e in model.entities for a in e.attributes
                           if a.provenance.confidence < h.REVIEW_BELOW and not a.provenance.locked]
    return SemanticModel.model_validate(model.model_dump()), report   # re-run all validators


# --------------------------------------------------------------------------- entities & attributes
def _build_entity(t: TableMetadata, entity_id: str, fk_by_col: dict[tuple[str, str], Link]) -> Entity:
    self_ref = any(fk.is_self_referencing for fk in t.foreign_keys) or any(
        lk.parent_table == t.name for (tbl, _), lk in fk_by_col.items() if tbl == t.name)
    attributes = []
    for col in t.columns:
        link = fk_by_col.get((t.name, col.name))
        ctx = h.ColumnContext(t, col, fk_target=link.parent_table if link else None,
                              is_self_fk=bool(link and link.parent_table == t.name),
                              table_is_self_referencing=self_ref)
        guess = h.guess_semantic_type(ctx)
        attributes.append(_attribute(col, guess))
    name = entity_id.replace("_", " ").capitalize()
    business_key = [a.column for a in attributes if a.semantic_type == ST.BUSINESS_KEY][:1]
    return Entity(entity_id=entity_id, name=name, table=t.name, entity_type=EntityType.MASTER,
                  grain=f"One row per {name.lower()}", primary_key=t.primary_key, business_key=business_key,
                  attributes=attributes, provenance=Provenance(source="heuristic", confidence=0.7))


def _attribute(col, guess: h.TypeGuess) -> Attribute:
    st, p = guess.semantic_type, col.profile
    allowed, anomalies = None, []
    if st in h.CODE_TYPES and p and p.top_values:
        values, anomalies = h.classify_values(p.top_values)
        allowed = AllowedValues(values=values, source="observed") if values else None
    vrange = None
    if st in h.NUMERIC_TYPES and p and p.numeric and p.numeric.p01 is not None:
        vrange = ValueRange(min=p.numeric.p01, max=p.numeric.p99, source="observed")
    return Attribute(
        column=col.name,
        name=col.name.replace("_", " ").capitalize(),
        semantic_type=st,
        role=h.guess_role(st),
        pii=h.is_pii(st, col),
        expected_pattern=h.expected_pattern(st, col),
        allowed_values=allowed,
        value_anomalies=anomalies,
        valid_range=vrange,
        provenance=Provenance(source="heuristic", confidence=guess.confidence, note="; ".join(guess.reasons)),
    )


# --------------------------------------------------------------------------- relationships
def _relationship(lk: Link, ids: dict[str, str], metadata: SourceMetadata) -> Relationship:
    child, parent = ids[lk.child_table], ids[lk.parent_table]
    col = metadata.table(lk.child_table).column(lk.child_columns[0])
    null_rate = col.profile.null_rate if col.profile else 0.0
    self_ref = child == parent
    child_name, parent_name = child.replace("_", " "), parent.replace("_", " ")
    return Relationship(
        relationship_id=f"{child}_parent" if self_ref else f"{child}_{parent}",
        from_entity=child, from_attributes=lk.child_columns, to_entity=parent, to_attributes=lk.parent_columns,
        cardinality="many_to_one",
        mandatory=not self_ref and null_rate <= MANDATORY_MAX_NULL_RATE,
        kind=lk.kind,
        description=(f"A {child_name}'s parent {parent_name}" if self_ref
                     else f"Each {child_name} belongs to one {parent_name}"),
        provenance=Provenance(source="heuristic", confidence=lk.confidence, note=lk.evidence),
    )


# --------------------------------------------------------------------------- hierarchies
def _build_hierarchies(engine: Engine, metadata: SourceMetadata, entities: list[Entity], links: list[Link],
                       ids: dict[str, str]) -> list[Hierarchy]:
    by_table = {e.table: e for e in entities}
    result = []
    for lk in links:
        if lk.child_table != lk.parent_table or len(lk.child_columns) != 1:
            continue
        t = metadata.table(lk.child_table)
        entity = by_table[t.name]
        level_attr = next((a for a in entity.attributes if a.semantic_type == ST.HIERARCHY_LEVEL), None)
        if level_attr is None or len(t.primary_key) != 1:
            continue
        structure = hierarchy_levels(engine, t, metadata.schema_name, lk.child_columns[0], level_attr.column)
        if structure is None:
            continue
        evidence = list(structure.evidence)
        attached = []
        for other in links:
            if other.parent_table != t.name or other.child_table == t.name or len(other.child_columns) != 1:
                continue
            found = attach_level(engine, other.child_table, other.child_columns[0], t, level_attr.column,
                                 metadata.schema_name)
            if found:
                level, ev = found
                attached.append(AttachedEntity(entity=ids[other.child_table], attribute=other.child_columns[0],
                                               must_attach_at=level))
                evidence += ev
        # the level column's allowed values are exactly the hierarchy levels
        level_attr.allowed_values = AllowedValues(values=structure.levels, source="observed")
        result.append(Hierarchy(
            hierarchy_id=f"{entity.entity_id}_hierarchy", name=f"{entity.name} hierarchy", style="self_referencing",
            levels=structure.levels, entity=entity.entity_id, parent_attribute=lk.child_columns[0],
            level_attribute=level_attr.column, attached_entities=attached, evidence=evidence,
            provenance=Provenance(source="heuristic", confidence=0.9,
                                  note="levels derived from parent/child level counts"),
        ))
    return result


def _assign_entity_types(entities: list[Entity], links: list[Link], hierarchies: list[Hierarchy],
                         metadata: SourceMetadata, ids: dict[str, str]) -> None:
    hierarchy_entities = {hy.entity for hy in hierarchies}
    outgoing = {e.entity_id: {ids[lk.parent_table] for lk in links
                              if ids[lk.child_table] == e.entity_id and lk.parent_table != e.table}
                for e in entities}

    def has(e: Entity, *types: ST) -> bool:
        return any(a.semantic_type in types for a in e.attributes)

    transactions = {e.entity_id for e in entities if e.entity_id not in hierarchy_entities and outgoing[e.entity_id]
                    and has(e, ST.EVENT_DATE, ST.EVENT_TIMESTAMP) and has(e, ST.AMOUNT)}
    for e in entities:
        rows = metadata.table(e.table).row_count or 0
        if e.entity_id in hierarchy_entities:
            e.entity_type = EntityType.HIERARCHY
        elif outgoing[e.entity_id] & transactions and has(e, ST.QUANTITY, ST.AMOUNT):
            e.entity_type = EntityType.TRANSACTION_LINE
        elif e.entity_id in transactions:
            e.entity_type = EntityType.TRANSACTION
        elif not outgoing[e.entity_id] and rows <= REFERENCE_MAX_ROWS and all(
                a.semantic_type in h.CODE_TYPES | h.KEY_TYPES | {ST.LABEL, ST.DESCRIPTION} for a in e.attributes):
            e.entity_type = EntityType.REFERENCE
        if e.entity_type == EntityType.TRANSACTION:   # the first event date is what you analyse by
            first_date = next((a for a in e.attributes if a.semantic_type == ST.EVENT_DATE), None)
            if first_date:
                first_date.role = AttributeRole.DIMENSION


# --------------------------------------------------------------------------- versions
def _carry_forward_attribute(new: Attribute, old: Attribute) -> None:
    """Keep earlier enrichment that this build didn't redo (e.g. a rebuild with --no-llm)."""
    if new.description is None:            # this build wrote no text of its own -> keep the earlier text
        new.name, new.description = old.name, old.description

    def observed(a: Attribute) -> set:
        return set(a.allowed_values.values if a.allowed_values else []) | {x.value for x in a.value_anomalies}
    if old.allowed_values and observed(new) == observed(old):
        # same values in the data -> earlier verdicts (LLM/steward) on which ones are valid still hold
        new.allowed_values, new.value_anomalies = old.allowed_values, old.value_anomalies
    if new.provenance.source == "heuristic" and new.provenance.confidence < h.REVIEW_BELOW:
        if old.provenance.source == "llm":
            # the LLM settled an uncertain type before; the heuristics are still uncertain -> keep its call
            new.semantic_type, new.role, new.provenance = old.semantic_type, old.role, old.provenance
        elif old.semantic_type == new.semantic_type and old.provenance.confidence > new.provenance.confidence:
            new.provenance = old.provenance          # e.g. "confirmed by LLM"
    new.pii = new.pii or old.pii                     # personal-data flags are never silently dropped


def _merge_previous(model: SemanticModel, prev: SemanticModel, report: BuildReport) -> SemanticModel:
    model.version = prev.version + 1
    if prev.domain.description:
        model.domain = prev.domain
    prev_entities = {e.table: e for e in prev.entities}
    new_tables = {e.table for e in model.entities}
    report.drift += [f"table removed: {t}" for t in prev_entities if t not in new_tables]

    for e in model.entities:
        old = prev_entities.get(e.table)
        if old is None:
            report.drift.append(f"table added: {e.table}")
            continue
        old_attrs = {a.column: a for a in old.attributes}
        new_cols = {a.column for a in e.attributes}
        report.drift += [f"column removed: {e.table}.{c}" for c in old_attrs if c not in new_cols]
        report.drift += [f"column added: {e.table}.{c}" for c in new_cols if c not in old_attrs]
        for i, a in enumerate(e.attributes):
            prev_a = old_attrs.get(a.column)
            if prev_a is None:
                continue
            if prev_a.provenance.locked:
                e.attributes[i] = prev_a
                report.kept_locked.append(f"{e.entity_id}.{a.column}")
                continue
            _carry_forward_attribute(a, prev_a)
        if old.provenance.locked:
            e.entity_id, e.name, e.description, e.grain = old.entity_id, old.name, old.description, old.grain
            e.entity_type, e.business_key, e.provenance = old.entity_type, old.business_key, old.provenance
            report.kept_locked.append(e.entity_id)
        elif e.provenance.source == "heuristic" and old.provenance.source != "heuristic":
            # a rebuild without the LLM must not erase what an earlier LLM/steward pass wrote
            e.name, e.description, e.grain, e.provenance = old.name, old.description, old.grain, old.provenance

    old_hierarchies = {hy.hierarchy_id: hy for hy in prev.hierarchies}
    for hy in model.hierarchies:
        if hy.hierarchy_id in old_hierarchies:
            hy.name = old_hierarchies[hy.hierarchy_id].name

    entity_ids = {e.entity_id for e in model.entities}
    for kind, new_items, old_items, key in (
            ("relationship", model.relationships, prev.relationships, "relationship_id"),
            ("hierarchy", model.hierarchies, prev.hierarchies, "hierarchy_id")):
        for old in old_items:
            if not old.provenance.locked:
                continue
            refs = {old.from_entity, old.to_entity} if kind == "relationship" else {old.entity}
            if not refs <= entity_ids:
                report.drift.append(f"locked {kind} {getattr(old, key)} dropped: its entities no longer exist")
                continue
            new_items[:] = [x for x in new_items if getattr(x, key) != getattr(old, key)] + [old]
            report.kept_locked.append(getattr(old, key))
    return model
