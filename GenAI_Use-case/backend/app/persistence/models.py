"""ORM tables: sources, metadata_snapshots, semantic_model_versions, context_items, generation_runs, rules, rule_reviews.

Storage approach:
- documents read/written whole (metadata snapshot, semantic model)  -> one JSON column
- rules                                                              -> JSON + indexed columns for filtering
- small flat records (sources, reviews, runs, context)              -> plain columns
"""
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, MetaData, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    # Predictable constraint names - Alembic needs names to alter/drop constraints later (esp. on SQLite).
    metadata = MetaData(naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    })


class SourceRow(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, comment="source fingerprint")
    display_name: Mapped[str] = mapped_column(String(200))
    dialect: Mapped[str] = mapped_column(String(20))
    host: Mapped[str | None] = mapped_column(String(255))
    port: Mapped[int | None] = mapped_column(Integer)
    database: Mapped[str] = mapped_column(String(500))
    schema_name: Mapped[str | None] = mapped_column(String(128))
    username: Mapped[str | None] = mapped_column(String(128))
    password_encrypted: Mapped[str | None] = mapped_column(Text)
    store_password: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_connected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MetadataSnapshotRow(Base):
    __tablename__ = "metadata_snapshots"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    extracted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    profiled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    structure_hash: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SemanticModelVersionRow(Base):
    __tablename__ = "semantic_model_versions"
    __table_args__ = (UniqueConstraint("source_id", "version"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))
    snapshot_id: Mapped[str | None] = mapped_column(ForeignKey("metadata_snapshots.id", ondelete="SET NULL"))
    created_by: Mapped[str] = mapped_column(String(128))
    change_note: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ContextItemRow(Base):
    __tablename__ = "context_items"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    entity: Mapped[str | None] = mapped_column(String(128))
    attribute: Mapped[str | None] = mapped_column(String(128))
    text: Mapped[str] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(128))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class GenerationRunRow(Base):
    __tablename__ = "generation_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    semantic_model_version: Mapped[int | None] = mapped_column(Integer)
    model: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(20), default="running")    # running | succeeded | failed
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    prompt: Mapped[str | None] = mapped_column(Text)
    raw_response: Mapped[str | None] = mapped_column(Text)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    generated: Mapped[int] = mapped_column(Integer, default=0)
    invalid: Mapped[int] = mapped_column(Integer, default=0)
    dropped_duplicate: Mapped[int] = mapped_column(Integer, default=0)
    flagged_similar: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)


class RuleRow(Base):
    __tablename__ = "rules"
    __table_args__ = (
        UniqueConstraint("source_id", "fingerprint"),
        Index("ix_rules_source_status", "source_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, comment="= Rule.rule_id")
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    generation_run_id: Mapped[str | None] = mapped_column(ForeignKey("generation_runs.id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(String(20))
    origin: Mapped[str] = mapped_column(String(20))
    category: Mapped[str] = mapped_column(String(20))
    dimension: Mapped[str] = mapped_column(String(30))
    severity: Mapped[str] = mapped_column(String(10))
    target_entity: Mapped[str] = mapped_column(String(128), index=True)
    name: Mapped[str] = mapped_column(String(300))
    fingerprint: Mapped[str] = mapped_column(String(16))
    original_fingerprint: Mapped[str] = mapped_column(String(16), index=True,
                                                      comment="fingerprint as first proposed (before edits)")
    similarity_key: Mapped[str] = mapped_column(String(300), index=True)
    similar_to_rule_id: Mapped[str | None] = mapped_column(ForeignKey("rules.id", ondelete="SET NULL"))
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    reviews: Mapped[list["RuleReviewRow"]] = relationship(back_populates="rule", order_by="RuleReviewRow.created_at",
                                                          cascade="all, delete-orphan")


class RuleReviewRow(Base):
    __tablename__ = "rule_reviews"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    rule_id: Mapped[str] = mapped_column(ForeignKey("rules.id", ondelete="CASCADE"), index=True)
    decision: Mapped[str] = mapped_column(String(20))     # approved | rejected | modified
    reviewer: Mapped[str] = mapped_column(String(128))
    reason_code: Mapped[str | None] = mapped_column(String(30), comment="wrong | not_relevant | too_strict | ...")
    reason: Mapped[str | None] = mapped_column(Text)
    before: Mapped[dict | None] = mapped_column(JSON)
    after: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    rule: Mapped[RuleRow] = relationship(back_populates="reviews")
