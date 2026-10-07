"""Semantic model versions - now stored in the app database (semantic_model_versions table).

Thin wrapper kept so callers (CLI, later the API) don't change; see persistence/repositories.py.
"""
from sqlalchemy.engine import Engine

from app.persistence.repositories import unit_of_work
from app.schemas.semantic_model import SemanticModel


def latest_model(source_id: str, engine: Engine | None = None) -> SemanticModel | None:
    with unit_of_work(engine) as repo:
        return repo.semantic_models.latest(source_id)


def save_model(model: SemanticModel, created_by: str = "engine", note: str | None = None,
               snapshot_id: str | None = None, engine: Engine | None = None) -> SemanticModel:
    with unit_of_work(engine) as repo:
        return repo.semantic_models.save_version(model, created_by=created_by, note=note, snapshot_id=snapshot_id)
