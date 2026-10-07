"""LLM pass that adds business meaning/descriptions to entities and attributes.

The LLM sees a compact DIGEST of the heuristic model + profile (never raw rows, never raw values of
sensitive columns), and returns JSON validated against EnrichmentResponse. The merge policy decides
what it may change:
  - names / descriptions / grain / domain : yes, unless steward-locked
  - semantic_type                          : only where heuristic confidence < REVIEW_BELOW
  - pii                                    : may only turn ON (never silently un-flag personal data)
  - value verdicts                         : may move observed values between allowed / anomalies
"""
import logging
from typing import Literal

from pydantic import BaseModel

from app.agent.llm_client import LLMClient, LLMError
from app.schemas.metadata import ScalarValue, SourceMetadata, TableMetadata
from app.schemas.semantic_model import (
    AllowedValues,
    Attribute,
    Domain,
    Entity,
    EntityType,
    Provenance,
    SemanticModel,
    SemanticType,
    ValueAnomaly,
)
from app.semantic.heuristics import PII_TYPES, REVIEW_BELOW, guess_role

log = logging.getLogger(__name__)
TABLES_PER_BATCH = 10


# --------------------------------------------------------------------------- response contract
class AttributeEnrichment(BaseModel):
    column: str
    name: str | None = None
    description: str | None = None
    semantic_type: SemanticType | None = None
    reason: str | None = None
    pii: bool | None = None


class ValueVerdict(BaseModel):
    column: str
    value: ScalarValue
    verdict: Literal["valid", "variant", "invalid"]
    looks_like: ScalarValue = None


class EntityEnrichment(BaseModel):
    entity_id: str
    name: str | None = None
    description: str | None = None
    grain: str | None = None
    entity_type: EntityType | None = None
    attributes: list[AttributeEnrichment] = []
    value_verdicts: list[ValueVerdict] = []


class HierarchyEnrichment(BaseModel):
    hierarchy_id: str
    name: str


class EnrichmentResponse(BaseModel):
    domain: Domain | None = None
    entities: list[EntityEnrichment]
    hierarchies: list[HierarchyEnrichment] = []


# --------------------------------------------------------------------------- prompt
SYSTEM_PROMPT = f"""You are a senior data steward documenting a database for a data quality team.
You receive a machine-generated draft of its semantic model (built from column names and data
profiling statistics). Improve it and answer with ONE JSON object only.

Tasks:
1. domain: name and 1-2 sentence description of the business domain.
2. For EVERY entity: business name, one-sentence description, grain ("One row per ...").
   Set entity_type only if the draft is wrong.
3. For EVERY attribute: business name and a short description of its business meaning.
4. For attributes marked [REVIEW]: confirm or correct semantic_type and give a short reason.
   Do not change semantic_type of attributes that are not marked [REVIEW].
5. Set pii=true for any attribute that is personal data but is not flagged PII.
6. For code columns, the draft splits observed values into "valid" and "anomalies". Give a
   value_verdict for every anomaly and for any valid value you believe is actually wrong:
   verdict "valid" (a legitimate code), "variant" (a misspelling/alias of a valid code - set
   looks_like), or "invalid" (garbage / placeholder).
7. hierarchies: a business name for each.

Only use entity_ids, columns and hierarchy_ids that appear in the draft.
semantic_type must be one of: {", ".join(t.value for t in SemanticType)}
entity_type must be one of: {", ".join(t.value for t in EntityType)}

JSON shape:
{{"domain": {{"name": "...", "description": "..."}},
  "entities": [{{"entity_id": "...", "name": "...", "description": "...", "grain": "One row per ...",
                 "attributes": [{{"column": "...", "name": "...", "description": "...",
                                  "semantic_type": "... (only for [REVIEW])", "reason": "...", "pii": true}}],
                 "value_verdicts": [{{"column": "...", "value": "...", "verdict": "variant", "looks_like": "..."}}]}}],
  "hierarchies": [{{"hierarchy_id": "...", "name": "..."}}]}}"""


def _fmt_share(x: float) -> str:
    return f"{x:.1%}" if x >= 0.001 else f"{x:.2%}"


def describe_attribute(attr: Attribute, col_meta) -> str:
    p = col_meta.profile
    flags = [f"{attr.semantic_type.value}({attr.provenance.confidence:.2f})"]
    if attr.provenance.confidence < REVIEW_BELOW and not attr.provenance.locked:
        flags.append("[REVIEW]")
    if attr.pii:
        flags.append("PII")
    line = f"- {attr.column} {col_meta.generic_type.value} {' '.join(flags)}"
    if p:
        line += f" | nulls {_fmt_share(p.null_rate)} distinct {p.distinct_count:,}"
        if attr.allowed_values:
            shares = {v.value: v.share for v in (p.top_values or [])}
            line += " | valid: " + ", ".join(f"{v} {_fmt_share(shares.get(v, 0))}" for v in attr.allowed_values.values)
        if attr.value_anomalies:
            line += " | anomalies: " + ", ".join(
                f"{a.value}({a.count})" + (f"->{a.looks_like}?" if a.looks_like is not None else "->?")
                for a in attr.value_anomalies)
        if p.numeric and p.numeric.min is not None:
            line += f" | range {p.numeric.min:g}..{p.numeric.max:g} (p01 {p.numeric.p01}, p99 {p.numeric.p99})"
        if p.date and p.date.min:
            line += f" | {p.date.min}..{p.date.max}"
        if p.patterns and not attr.allowed_values:
            line += " | patterns " + ", ".join(f"{pt.pattern} {_fmt_share(pt.share)}" for pt in p.patterns[:2])
        if p.examples and not attr.allowed_values:
            line += f" | e.g. {', '.join(p.examples[:2])}"   # already masked for sensitive columns
    return line


def build_digest(model: SemanticModel, metadata: SourceMetadata, entity_ids: list[str]) -> str:
    lines = []
    for eid in entity_ids:
        e = model.entity(eid)
        t: TableMetadata = metadata.table(e.table)
        lines.append(f"\nENTITY {e.entity_id} (table {e.table}, {t.row_count:,} rows) "
                     f"draft entity_type={e.entity_type.value} grain='{e.grain}'")
        lines += [describe_attribute(a, t.column(a.column)) for a in e.attributes]
    rels = [r for r in model.relationships if r.from_entity in entity_ids or r.to_entity in entity_ids]
    if rels:
        lines.append("\nRELATIONSHIPS")
        lines += [f"- {r.from_entity}.{','.join(r.from_attributes)} -> {r.to_entity} ({r.kind})" for r in rels]
    if model.hierarchies:
        lines.append("\nHIERARCHIES")
        lines += [f"- {h.hierarchy_id}: {' > '.join(h.levels)} on {h.entity}" for h in model.hierarchies]
    return "\n".join(lines)


# --------------------------------------------------------------------------- apply
def enrich(model: SemanticModel, metadata: SourceMetadata, llm: LLMClient, changes: list[str]) -> SemanticModel:
    ids = [e.entity_id for e in model.entities]
    for i in range(0, len(ids), TABLES_PER_BATCH):
        batch = ids[i:i + TABLES_PER_BATCH]
        user = f"Draft semantic model for database '{metadata.database}':\n{build_digest(model, metadata, batch)}"
        try:
            response = llm.complete_structured(SYSTEM_PROMPT, user, EnrichmentResponse)
        except LLMError as e:
            changes.append(f"LLM batch {batch} skipped: {e}")
            continue
        apply_enrichment(model, metadata, response, changes)
    return model


def apply_enrichment(model: SemanticModel, metadata: SourceMetadata, response: EnrichmentResponse,
                     changes: list[str]) -> None:
    if response.domain and response.domain.description:
        model.domain = response.domain
    entities = {e.entity_id: e for e in model.entities}
    for ee in response.entities:
        entity = entities.get(ee.entity_id)
        if entity is None:
            changes.append(f"ignored unknown entity '{ee.entity_id}' from LLM")
            continue
        _apply_entity(entity, metadata.table(entity.table), ee, changes)
    hierarchies = {h.hierarchy_id: h for h in model.hierarchies}
    for he in response.hierarchies:
        h = hierarchies.get(he.hierarchy_id)
        if h and not h.provenance.locked:
            h.name = he.name


def _apply_entity(entity: Entity, t: TableMetadata, ee: EntityEnrichment, changes: list[str]) -> None:
    if not entity.provenance.locked:
        entity.name = ee.name or entity.name
        entity.description = ee.description or entity.description
        entity.grain = ee.grain or entity.grain
        if ee.entity_type and ee.entity_type != entity.entity_type:
            changes.append(f"{entity.entity_id}: entity_type {entity.entity_type.value} -> {ee.entity_type.value}")
            entity.entity_type = ee.entity_type
        entity.provenance = Provenance(source="llm", confidence=entity.provenance.confidence,
                                       note="named and described by LLM")

    attrs = {a.column: a for a in entity.attributes}
    for ae in ee.attributes:
        attr = attrs.get(ae.column)
        if attr is None:
            changes.append(f"ignored unknown column '{entity.entity_id}.{ae.column}' from LLM")
        elif not attr.provenance.locked:
            _apply_attribute(entity.entity_id, attr, ae, changes)

    for vv in ee.value_verdicts:
        attr = attrs.get(vv.column)
        if attr is not None and not attr.provenance.locked and attr.allowed_values:
            counts = {v.value: v.count for v in (t.column(attr.column).profile.top_values or [])}
            _apply_verdict(entity.entity_id, attr, vv, counts.get(vv.value, 0), changes)


def _apply_attribute(eid: str, attr: Attribute, ae: AttributeEnrichment, changes: list[str]) -> None:
    attr.name = ae.name or attr.name
    attr.description = ae.description or attr.description
    prov = attr.provenance
    if ae.semantic_type and ae.semantic_type != attr.semantic_type:
        if prov.confidence < REVIEW_BELOW:
            changes.append(f"{eid}.{attr.column}: semantic_type {attr.semantic_type.value} -> "
                           f"{ae.semantic_type.value} ({ae.reason or 'no reason given'})")
            note = f"heuristic said {attr.semantic_type.value} ({prov.confidence:.2f}); LLM: {ae.reason or ''}"
            attr.semantic_type = ae.semantic_type
            attr.role = guess_role(ae.semantic_type)
            attr.pii = attr.pii or ae.semantic_type in PII_TYPES
            attr.provenance = Provenance(source="llm", confidence=0.8, note=note)
        else:
            changes.append(f"{eid}.{attr.column}: LLM suggested {ae.semantic_type.value}, kept "
                           f"{attr.semantic_type.value} (confidence {prov.confidence:.2f})")
    elif ae.semantic_type == attr.semantic_type and prov.confidence < REVIEW_BELOW:
        attr.provenance = Provenance(source=prov.source, confidence=REVIEW_BELOW, note="confirmed by LLM")
    if ae.pii and not attr.pii:
        attr.pii = True
        changes.append(f"{eid}.{attr.column}: flagged as PII by LLM")


def _apply_verdict(eid: str, attr: Attribute, vv: ValueVerdict, count: int, changes: list[str]) -> None:
    allowed = list(attr.allowed_values.values)
    anomaly = next((a for a in attr.value_anomalies if a.value == vv.value), None)
    if vv.verdict == "valid" and anomaly is not None:
        attr.value_anomalies.remove(anomaly)
        allowed.append(vv.value)
        changes.append(f"{eid}.{attr.column}: '{vv.value}' accepted as a valid value")
    elif vv.verdict in ("variant", "invalid") and vv.value in allowed:
        allowed.remove(vv.value)
        attr.value_anomalies.append(ValueAnomaly(
            value=vv.value, count=count, looks_like=vv.looks_like,
            reason=f"LLM: {'variant of ' + repr(vv.looks_like) if vv.verdict == 'variant' else 'invalid value'}"))
        changes.append(f"{eid}.{attr.column}: '{vv.value}' moved to anomalies ({vv.verdict})")
    elif anomaly is not None and vv.verdict == "variant" and vv.looks_like is not None and anomaly.looks_like is None:
        anomaly.looks_like = vv.looks_like
        anomaly.reason = f"LLM: variant of {vv.looks_like!r}"
    if allowed:
        attr.allowed_values = AllowedValues(values=allowed, source=attr.allowed_values.source)
