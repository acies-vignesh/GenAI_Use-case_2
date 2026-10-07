"""Semantic layer JSON contract: entities, attributes (business meaning, semantic type, PII flag), relationships, hierarchies, domain context. (Output of Layer 4, editable by the steward)

The metadata says *what is stored* ("region_id INTEGER, 10 orphans"); this model says *what it means*
("a customer lives in a City of the Geography hierarchy"). Rule generation reasons over this model.

Every entity/attribute/relationship/hierarchy carries `provenance`, so a rebuild (heuristics + LLM)
never overwrites what a steward has corrected and locked.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.metadata import ScalarValue


class SemanticType(str, Enum):
    """Controlled vocabulary - rule generation keys off these, so keep it small and stable."""
    # keys
    IDENTIFIER = "identifier"            # surrogate / technical key
    BUSINESS_KEY = "business_key"        # natural key people use (SKU, order number)
    FOREIGN_KEY = "foreign_key"
    PARENT_KEY = "parent_key"            # self-reference to a parent row in a hierarchy
    # names & text
    LABEL = "label"                      # name of a thing (product name, region name)
    PERSON_NAME = "person_name"
    ORGANIZATION_NAME = "organization_name"
    DESCRIPTION = "description"
    FREE_TEXT = "free_text"
    # contact & location
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    POSTAL_CODE = "postal_code"
    COUNTRY_CODE = "country_code"
    URL = "url"
    # time
    BIRTH_DATE = "birth_date"
    EVENT_DATE = "event_date"
    EVENT_TIMESTAMP = "event_timestamp"
    # numbers
    AMOUNT = "amount"
    PRICE = "price"
    COST = "cost"
    QUANTITY = "quantity"
    PERCENTAGE = "percentage"
    RATE = "rate"
    COUNT = "count"
    # codes
    CURRENCY_CODE = "currency_code"
    STATUS_CODE = "status_code"          # lifecycle state (PLACED > SHIPPED > DELIVERED)
    CATEGORY_CODE = "category_code"      # classification (segment, channel, gender)
    HIERARCHY_LEVEL = "hierarchy_level"  # which level of a hierarchy a row sits at
    FLAG = "flag"
    OTHER = "other"


class AttributeRole(str, Enum):
    KEY = "key"
    FOREIGN_KEY = "foreign_key"
    MEASURE = "measure"          # numbers you aggregate
    DIMENSION = "dimension"      # things you group/filter by
    ATTRIBUTE = "attribute"      # descriptive
    AUDIT = "audit"              # created_at, updated_by...


class EntityType(str, Enum):
    MASTER = "master"                      # customers, products
    TRANSACTION = "transaction"            # orders
    TRANSACTION_LINE = "transaction_line"  # order lines
    REFERENCE = "reference"                # code lists, currencies
    HIERARCHY = "hierarchy"                # self-referencing trees (regions, categories)


class Provenance(BaseModel):
    source: Literal["heuristic", "llm", "steward"]
    confidence: float = Field(1.0, ge=0, le=1)
    locked: bool = Field(False, description="Steward-confirmed: rebuilds must not change it")
    note: str | None = None


class AllowedValues(BaseModel):
    values: list[ScalarValue] = Field(min_length=1)
    source: Literal["observed", "steward"] = Field(
        description="'observed' = seen in the data (may contain errors); 'steward' = declared truth")


class ValueAnomaly(BaseModel):
    """A value seen in the data that is NOT considered valid - evidence for rule generation."""
    value: ScalarValue
    count: int
    looks_like: ScalarValue = Field(None, description="The valid value it most likely means, if any")
    reason: str


class ValueRange(BaseModel):
    min: float | None = None
    max: float | None = None
    source: Literal["observed", "steward"] = "observed"


class Attribute(BaseModel):
    column: str
    name: str
    description: str | None = None
    semantic_type: SemanticType
    role: AttributeRole
    pii: bool = False
    unit: str | None = Field(None, description="e.g. INR, units, %, days")
    expected_pattern: str | None = Field(None, description="Regex every value should match")
    allowed_values: AllowedValues | None = None
    value_anomalies: list[ValueAnomaly] = []
    valid_range: ValueRange | None = None
    provenance: Provenance


class Entity(BaseModel):
    entity_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str
    table: str
    entity_type: EntityType
    grain: str = Field(description="What one row represents")
    description: str | None = None
    primary_key: list[str]
    business_key: list[str] = []
    attributes: list[Attribute]
    provenance: Provenance

    def attribute(self, column: str) -> Attribute:
        return next(a for a in self.attributes if a.column == column)

    @property
    def columns(self) -> set[str]:
        return {a.column for a in self.attributes}


class Relationship(BaseModel):
    relationship_id: str
    from_entity: str
    from_attributes: list[str] = Field(min_length=1)
    to_entity: str
    to_attributes: list[str] = Field(min_length=1)
    cardinality: Literal["many_to_one", "one_to_one"]
    mandatory: bool = Field(description="Business rule: every child row must have a parent")
    kind: Literal["declared", "inferred"]
    description: str
    provenance: Provenance


class AttachedEntity(BaseModel):
    entity: str
    attribute: str
    must_attach_at: str = Field(description="Hierarchy level the entity must point to (usually the leaf)")


class Hierarchy(BaseModel):
    hierarchy_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str
    style: Literal["self_referencing", "multi_table"]
    levels: list[str] = Field(min_length=2, description="Top -> leaf. Level values (self_referencing) "
                                                         "or entity_ids (multi_table)")
    entity: str | None = Field(None, description="self_referencing only: the tree's entity")
    parent_attribute: str | None = None
    level_attribute: str | None = None
    attached_entities: list[AttachedEntity] = []
    evidence: list[str] = Field([], description="Observed violations of the structure, e.g. level skips")
    provenance: Provenance

    @model_validator(mode="after")
    def _style_fields(self):
        if self.style == "self_referencing" and not (self.entity and self.parent_attribute and self.level_attribute):
            raise ValueError("self_referencing hierarchy needs entity, parent_attribute and level_attribute")
        return self


class Domain(BaseModel):
    name: str
    description: str | None = None


class SemanticModel(BaseModel):
    model_version: str = "1.0"
    source_id: str
    version: int = Field(1, ge=1)
    status: Literal["draft", "reviewed"] = "draft"
    based_on_metadata: datetime = Field(description="extracted_at of the metadata snapshot this was built from")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    domain: Domain
    entities: list[Entity]
    relationships: list[Relationship] = []
    hierarchies: list[Hierarchy] = []

    def entity(self, entity_id: str) -> Entity:
        return next(e for e in self.entities if e.entity_id == entity_id)

    def entity_by_table(self, table: str) -> Entity:
        return next(e for e in self.entities if e.table == table)

    def hierarchy(self, hierarchy_id: str) -> Hierarchy:
        return next(h for h in self.hierarchies if h.hierarchy_id == hierarchy_id)

    @model_validator(mode="after")
    def _references_resolve(self):
        """Every name used anywhere must point at something that exists."""
        errors: list[str] = []
        ents = {e.entity_id: e for e in self.entities}
        if len(ents) != len(self.entities):
            errors.append("duplicate entity_id")

        def check_attrs(entity_id: str, cols: list[str], where: str):
            if entity_id not in ents:
                errors.append(f"{where}: unknown entity '{entity_id}'")
            else:
                errors.extend(f"{where}: unknown attribute '{entity_id}.{c}'"
                              for c in cols if c not in ents[entity_id].columns)

        for e in self.entities:
            check_attrs(e.entity_id, e.primary_key + e.business_key, f"entity {e.entity_id}")
        for r in self.relationships:
            check_attrs(r.from_entity, r.from_attributes, f"relationship {r.relationship_id}")
            check_attrs(r.to_entity, r.to_attributes, f"relationship {r.relationship_id}")
        for h in self.hierarchies:
            where = f"hierarchy {h.hierarchy_id}"
            if h.style == "self_referencing":
                check_attrs(h.entity, [h.parent_attribute, h.level_attribute], where)
            else:
                errors.extend(f"{where}: unknown level entity '{lv}'" for lv in h.levels if lv not in ents)
            for a in h.attached_entities:
                check_attrs(a.entity, [a.attribute], where)
                if a.must_attach_at not in h.levels:
                    errors.append(f"{where}: '{a.must_attach_at}' is not one of its levels {h.levels}")
        if errors:
            raise ValueError("; ".join(errors))
        return self
