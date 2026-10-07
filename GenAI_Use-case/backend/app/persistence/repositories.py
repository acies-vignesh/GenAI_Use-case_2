"""Data-access functions used by services; keeps SQL out of the other layers.

Usage:
    with unit_of_work() as repo:
        source = repo.sources.upsert(config)
        repo.rules.add(rule)
    # committed here (or rolled back if the block raised)

Repositories speak in the app's own types (SourceMetadata, SemanticModel, Rule); rows stay in here.
"""
# Lazy annotations: several repos have a method named `list`, which would otherwise shadow the
# built-in in later `-> list[...]` annotations of the same class.
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from sqlalchemy import func, or_, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.connectivity.connectors import ConnectionConfig
from app.connectivity.credentials import decrypt, encrypt
from app.connectivity.fingerprint import source_fingerprint
from app.persistence.database import session_scope
from app.persistence.models import (
    ContextItemRow,
    GenerationRunRow,
    MetadataSnapshotRow,
    RuleReviewRow,
    RuleRow,
    SemanticModelVersionRow,
    SourceRow,
)
from app.schemas.metadata import SourceMetadata
from app.schemas.rule import ReasonCode, Review, Rule, RuleStatus
from app.schemas.semantic_model import SemanticModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


class NotFoundError(LookupError):
    pass


class DuplicateRuleError(ValueError):
    pass


# --------------------------------------------------------------------------- sources
class SourceRepo:
    def __init__(self, s: Session):
        self.s = s

    def upsert(self, config: ConnectionConfig, remember_password: bool = True) -> SourceRow:
        sid = source_fingerprint(config)
        row = self.s.get(SourceRow, sid)
        if row is None:
            row = SourceRow(id=sid)
            self.s.add(row)
        row.display_name = config.display_name
        row.dialect, row.host, row.port = config.dialect, config.host, config.port
        row.database, row.schema_name, row.username = config.database, config.schema_name, config.username
        row.store_password = remember_password
        password = config.password.get_secret_value() if config.password else None
        if not remember_password:
            row.password_encrypted = None
        elif password:
            row.password_encrypted = encrypt(password)
        row.last_connected_at = _now()
        self.s.flush()
        return row

    def get(self, source_id: str) -> SourceRow:
        row = self.s.get(SourceRow, source_id)
        if row is None:
            raise NotFoundError(f"unknown source '{source_id}'")
        return row

    def list(self) -> list[SourceRow]:
        return list(self.s.scalars(select(SourceRow).order_by(SourceRow.last_connected_at.desc())))

    def connection_config(self, source_id: str, password: str | None = None) -> ConnectionConfig:
        """Rebuild the connection for a known source; `password` is needed when it isn't stored."""
        row = self.get(source_id)
        if password is None and row.password_encrypted:
            password = decrypt(row.password_encrypted)
        return ConnectionConfig(dialect=row.dialect, database=row.database, host=row.host, port=row.port,
                                username=row.username, password=password, schema_name=row.schema_name)


# --------------------------------------------------------------------------- metadata snapshots
def structure_hash(metadata: SourceMetadata) -> str:
    """Hash of tables/columns/types/keys only - unaffected by data or profile changes."""
    shape = [
        [t.name,
         [[c.name, c.native_type, c.nullable, c.is_primary_key] for c in t.columns],
         sorted([fk.columns, fk.referred_table] for fk in t.foreign_keys)]
        for t in sorted(metadata.tables, key=lambda t: t.name)
    ]
    return hashlib.sha256(json.dumps(shape).encode()).hexdigest()


class SnapshotRepo:
    def __init__(self, s: Session):
        self.s = s

    def save(self, metadata: SourceMetadata) -> MetadataSnapshotRow:
        row = MetadataSnapshotRow(source_id=metadata.source_id, extracted_at=metadata.extracted_at,
                                  profiled_at=metadata.profiled_at, structure_hash=structure_hash(metadata),
                                  payload=metadata.model_dump(mode="json"))
        self.s.add(row)
        self.s.flush()
        return row

    def latest_row(self, source_id: str, profiled_only: bool = True) -> MetadataSnapshotRow | None:
        q = select(MetadataSnapshotRow).where(MetadataSnapshotRow.source_id == source_id)
        if profiled_only:
            q = q.where(MetadataSnapshotRow.profiled_at.is_not(None))
        return self.s.scalars(q.order_by(MetadataSnapshotRow.created_at.desc()).limit(1)).first()

    def latest(self, source_id: str, profiled_only: bool = True) -> SourceMetadata | None:
        row = self.latest_row(source_id, profiled_only)
        return SourceMetadata.model_validate(row.payload) if row else None


# --------------------------------------------------------------------------- semantic models
class SemanticModelRepo:
    def __init__(self, s: Session):
        self.s = s

    def _max_version(self, source_id: str) -> int:
        return self.s.scalar(select(func.max(SemanticModelVersionRow.version))
                             .where(SemanticModelVersionRow.source_id == source_id)) or 0

    def save_version(self, model: SemanticModel, created_by: str = "engine", note: str | None = None,
                     snapshot_id: str | None = None) -> SemanticModel:
        """Versions are immutable: every save is a NEW version (max + 1)."""
        model = model.model_copy(update={"version": self._max_version(model.source_id) + 1})
        self.s.add(SemanticModelVersionRow(
            source_id=model.source_id, version=model.version, status=model.status, snapshot_id=snapshot_id,
            created_by=created_by, change_note=note, payload=model.model_dump(mode="json")))
        self.s.flush()
        return model

    @staticmethod
    def _content(model: SemanticModel) -> dict:
        """Everything except the bookkeeping fields that differ between otherwise identical versions."""
        return model.model_dump(mode="json", exclude={"version", "created_at"})

    def save_if_changed(self, model: SemanticModel, created_by: str = "engine", note: str | None = None,
                        snapshot_id: str | None = None) -> tuple[SemanticModel, bool]:
        """Like save_version, but returns (latest, False) when nothing changed - no duplicate versions."""
        latest = self.latest(model.source_id)
        if latest is not None and self._content(latest) == self._content(model):
            return latest, False
        return self.save_version(model, created_by, note, snapshot_id), True

    def get(self, source_id: str, version: int | None = None) -> SemanticModel | None:
        q = select(SemanticModelVersionRow).where(SemanticModelVersionRow.source_id == source_id)
        q = q.where(SemanticModelVersionRow.version == version) if version else \
            q.order_by(SemanticModelVersionRow.version.desc()).limit(1)
        row = self.s.scalars(q).first()
        return SemanticModel.model_validate(row.payload) if row else None

    def latest(self, source_id: str) -> SemanticModel | None:
        return self.get(source_id)

    def history(self, source_id: str) -> list[SemanticModelVersionRow]:
        return list(self.s.scalars(select(SemanticModelVersionRow)
                                   .where(SemanticModelVersionRow.source_id == source_id)
                                   .order_by(SemanticModelVersionRow.version)))


# --------------------------------------------------------------------------- business context
class ContextRepo:
    def __init__(self, s: Session):
        self.s = s

    def add(self, source_id: str, text: str, created_by: str, entity: str | None = None,
            attribute: str | None = None) -> ContextItemRow:
        row = ContextItemRow(source_id=source_id, text=text.strip(), created_by=created_by,
                             entity=entity, attribute=attribute)
        self.s.add(row)
        self.s.flush()
        return row

    def active(self, source_id: str) -> list[ContextItemRow]:
        return list(self.s.scalars(select(ContextItemRow)
                                   .where(ContextItemRow.source_id == source_id, ContextItemRow.active)
                                   .order_by(ContextItemRow.created_at)))

    def retire(self, item_id: str) -> None:
        row = self.s.get(ContextItemRow, item_id)
        if row is None:
            raise NotFoundError(f"unknown context item '{item_id}'")
        row.active = False


# --------------------------------------------------------------------------- generation runs
class RunRepo:
    def __init__(self, s: Session):
        self.s = s

    def start(self, source_id: str, model: str, semantic_model_version: int | None,
              prompt: str | None = None) -> GenerationRunRow:
        row = GenerationRunRow(source_id=source_id, model=model, semantic_model_version=semantic_model_version,
                               prompt=prompt)
        self.s.add(row)
        self.s.flush()
        return row

    def finish(self, run_id: str, status: Literal["succeeded", "failed"], **fields) -> GenerationRunRow:
        row = self.s.get(GenerationRunRow, run_id)
        row.status, row.finished_at = status, _now()
        for k, v in fields.items():
            setattr(row, k, v)
        return row

    def list(self, source_id: str) -> list[GenerationRunRow]:
        return list(self.s.scalars(select(GenerationRunRow).where(GenerationRunRow.source_id == source_id)
                                   .order_by(GenerationRunRow.started_at.desc())))


# --------------------------------------------------------------------------- rules
@dataclass
class AddResult:
    outcome: Literal["stored", "duplicate", "similar"]
    rule: Rule
    existing: Rule | None = None     # the duplicate / similar rule already on file


@dataclass
class PromptHistory:
    """What the LLM must know about past decisions for one source (Steps 12-13)."""
    approved: list[Rule] = field(default_factory=list)
    rejected: list[tuple[Rule, str | None]] = field(default_factory=list)      # (rule, reason)
    superseded: list[tuple[Rule, Rule]] = field(default_factory=list)          # (original suggestion, approved edit)
    pending: list[Rule] = field(default_factory=list)                          # proposed, awaiting review
    reason_codes: dict[str, str | None] = field(default_factory=dict)          # rule_id -> latest reason code


class RuleRepo:
    def __init__(self, s: Session):
        self.s = s

    @staticmethod
    def _to_rule(row: RuleRow) -> Rule:
        return Rule.model_validate(row.payload)

    def _row(self, rule_id: str) -> RuleRow:
        row = self.s.get(RuleRow, rule_id)
        if row is None:
            raise NotFoundError(f"unknown rule '{rule_id}'")
        return row

    def _sync(self, row: RuleRow, rule: Rule) -> None:
        row.status, row.origin = rule.status.value, rule.origin.value
        row.category, row.dimension, row.severity = rule.category.value, rule.dimension.value, rule.severity.value
        row.target_entity, row.name = rule.target.entity, rule.name
        row.fingerprint, row.similarity_key = rule.fingerprint, rule.similarity_key
        row.payload = rule.model_dump(mode="json")

    def find_duplicate(self, source_id: str, fingerprint: str) -> Rule | None:
        row = self.s.scalars(select(RuleRow).where(
            RuleRow.source_id == source_id,
            or_(RuleRow.fingerprint == fingerprint, RuleRow.original_fingerprint == fingerprint))).first()
        return self._to_rule(row) if row else None

    def original_fingerprints(self, source_id: str) -> set[str]:
        return set(self.s.scalars(select(RuleRow.original_fingerprint).where(RuleRow.source_id == source_id)))

    def find_similar(self, source_id: str, similarity_key: str) -> Rule | None:
        row = self.s.scalars(select(RuleRow).where(RuleRow.source_id == source_id,
                                                   RuleRow.similarity_key == similarity_key)
                             .order_by(RuleRow.created_at.desc())).first()
        return self._to_rule(row) if row else None

    def add(self, rule: Rule, created_by: str | None = None) -> AddResult:
        """Store a new rule unless an identical one (current or original form) exists."""
        duplicate = self.find_duplicate(rule.source_id, rule.fingerprint)
        if duplicate:
            return AddResult("duplicate", rule, duplicate)
        similar = self.find_similar(rule.source_id, rule.similarity_key)

        if rule.status != RuleStatus.PROPOSED and not rule.reviews:
            # imported as already decided: record who decided it, so the audit trail is complete
            rule = rule.model_copy(update={"reviews": [Review(decision=rule.status.value,
                                                              reviewer=created_by or rule.origin.value,
                                                              reason="imported")]})
        row = RuleRow(id=rule.rule_id, source_id=rule.source_id, generation_run_id=rule.generation_run_id,
                      original_fingerprint=rule.fingerprint, similar_to_rule_id=similar.rule_id if similar else None)
        self._sync(row, rule)
        self.s.add(row)
        for rv in rule.reviews:
            self.s.add(RuleReviewRow(rule_id=rule.rule_id, decision=rv.decision, reviewer=rv.reviewer,
                                     reason_code=rv.reason_code.value if rv.reason_code else None,
                                     reason=rv.reason, created_at=rv.at))
        self.s.flush()
        return AddResult("similar" if similar else "stored", rule, similar)

    def get(self, rule_id: str) -> Rule:
        return self._to_rule(self._row(rule_id))

    def similar_to(self, rule_id: str) -> str | None:
        return self._row(rule_id).similar_to_rule_id

    def list(self, source_id: str, status: RuleStatus | str | None = None, entity: str | None = None,
             origin: str | None = None) -> list[Rule]:
        q = select(RuleRow).where(RuleRow.source_id == source_id)
        if status:
            q = q.where(RuleRow.status == RuleStatus(status).value)
        if entity:
            q = q.where(RuleRow.target_entity == entity)
        if origin:
            q = q.where(RuleRow.origin == origin)
        return [self._to_rule(r) for r in self.s.scalars(q.order_by(RuleRow.created_at))]

    def approved(self, source_id: str) -> list[Rule]:
        return self.list(source_id, RuleStatus.APPROVED)

    def counts(self, source_id: str) -> dict[str, int]:
        rows = self.s.execute(select(RuleRow.status, func.count()).where(RuleRow.source_id == source_id)
                              .group_by(RuleRow.status))
        return {status: n for status, n in rows}

    def record_review(self, rule_id: str, decision: Literal["approved", "rejected", "modified"], reviewer: str,
                      reason: str | None = None, edited: Rule | None = None,
                      reason_code: ReasonCode | str | None = None) -> Rule:
        """Append a decision. 'modified' = steward edited the rule and approves the edited version."""
        row = self._row(rule_id)
        old = self._to_rule(row)
        code = ReasonCode(reason_code) if reason_code else None
        review = Review(decision=decision, reviewer=reviewer, reason=reason, reason_code=code)
        before = after = None

        if decision == "modified":
            if edited is None:
                raise ValueError("a 'modified' review needs the edited rule")
            new = Rule.model_validate({
                **edited.model_dump(mode="json"),
                # identity and lineage never change through an edit
                "rule_id": old.rule_id, "source_id": old.source_id, "origin": old.origin.value,
                "generation_run_id": old.generation_run_id, "model": old.model,
                "created_at": old.created_at.isoformat(), "status": RuleStatus.APPROVED.value,
                "reviews": [r.model_dump(mode="json") for r in old.reviews + [review]],
            })
            clash = self.s.scalars(select(RuleRow).where(RuleRow.source_id == old.source_id,
                                                         RuleRow.fingerprint == new.fingerprint,
                                                         RuleRow.id != rule_id)).first()
            if clash:
                raise DuplicateRuleError(f"an identical rule already exists: '{clash.name}' ({clash.id})")
            before, after = old.model_dump(mode="json"), new.model_dump(mode="json")
        else:
            new = old.model_copy(update={"status": RuleStatus(decision), "reviews": old.reviews + [review]})

        self._sync(row, new)
        row.updated_at = _now()
        self.s.add(RuleReviewRow(rule_id=rule_id, decision=decision, reviewer=reviewer, reason=reason,
                                 reason_code=code.value if code else None,
                                 before=before, after=after, created_at=review.at))
        self.s.flush()
        return new

    def reviews(self, rule_id: str) -> list[RuleReviewRow]:
        return list(self._row(rule_id).reviews)

    def history_for_prompt(self, source_id: str) -> PromptHistory:
        history = PromptHistory()
        for row in self.s.scalars(select(RuleRow).where(RuleRow.source_id == source_id)
                                  .order_by(RuleRow.created_at)):
            rule = self._to_rule(row)
            if rule.status == RuleStatus.PROPOSED:
                history.pending.append(rule)
            elif rule.status == RuleStatus.APPROVED:
                history.approved.append(rule)
                for rv in row.reviews:
                    if rv.decision == "modified" and rv.before:
                        history.superseded.append((Rule.model_validate(rv.before), rule))
            else:
                last = row.reviews[-1] if row.reviews else None
                history.rejected.append((rule, last.reason if last else None))
            if row.reviews:
                history.reason_codes[rule.rule_id] = row.reviews[-1].reason_code
        return history


# --------------------------------------------------------------------------- unit of work
class Repos:
    def __init__(self, s: Session):
        self.session = s
        self.sources = SourceRepo(s)
        self.snapshots = SnapshotRepo(s)
        self.semantic_models = SemanticModelRepo(s)
        self.context = ContextRepo(s)
        self.runs = RunRepo(s)
        self.rules = RuleRepo(s)


@contextmanager
def unit_of_work(engine: Engine | None = None) -> Iterator[Repos]:
    with session_scope(engine) as s:
        yield Repos(s)
