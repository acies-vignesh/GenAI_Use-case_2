"""Pipeline use cases for one source - shared by the CLI today and the API routes later.

    connect_source(config)        UI "Connect" button: test the connection, store the source -> source_id
    refresh_metadata(source_id)   extract + profile -> snapshot
    build_model(source_id)        semantic model from the latest snapshot -> new version (only if it changed)

Every function takes `app_engine` (the engine's own database) so tests can run against a temp DB.
Sources are addressed by source_id: connection details come from the `sources` table, never from .env.
"""
from dataclasses import dataclass

from sqlalchemy.engine import Engine

from app.agent.llm_client import LLMClient
from app.config import settings
from app.connectivity.connectors import ConnectionConfig, check_connection, create_source_engine
from app.extraction.profiler import profile_source
from app.extraction.schema_crawler import crawl_schema
from app.persistence.repositories import unit_of_work
from app.schemas.metadata import SourceMetadata
from app.schemas.semantic_model import SemanticModel
from app.semantic.builder import BuildReport, build_semantic_model


@dataclass
class ModelBuild:
    model: SemanticModel
    report: BuildReport
    created: bool              # False when the rebuild produced exactly the latest version
    snapshot_id: str | None


def connect_source(config: ConnectionConfig, remember_password: bool = True,
                   app_engine: Engine | None = None) -> str:
    """Test the connection first; only a reachable source is stored."""
    engine = create_source_engine(config)
    try:
        check_connection(engine)
    finally:
        engine.dispose()
    with unit_of_work(app_engine) as repo:
        return repo.sources.upsert(config, remember_password=remember_password).id


def open_source(source_id: str, password: str | None = None,
                app_engine: Engine | None = None) -> tuple[ConnectionConfig, Engine]:
    """Engine for a known source. `password` is required when the source doesn't store one."""
    with unit_of_work(app_engine) as repo:
        config = repo.sources.connection_config(source_id, password)
    engine = create_source_engine(config)
    check_connection(engine)
    return config, engine


def refresh_metadata(source_id: str, profile: bool = True, sample_limit: int | None = None,
                     include_row_counts: bool = True, password: str | None = None,
                     app_engine: Engine | None = None) -> tuple[SourceMetadata, str]:
    config, engine = open_source(source_id, password, app_engine)
    try:
        metadata = crawl_schema(engine, config, include_row_counts=include_row_counts)
        if profile:
            profile_source(engine, metadata, sample_limit=sample_limit or settings.sample_row_limit)
    finally:
        engine.dispose()
    with unit_of_work(app_engine) as repo:
        snapshot_id = repo.snapshots.save(metadata).id
    return metadata, snapshot_id


def build_model(source_id: str, llm: LLMClient | None = None, reprofile: bool = False,
                password: str | None = None, app_engine: Engine | None = None) -> ModelBuild:
    """Build from the latest profiled snapshot (re-profiling would re-sample and change the LLM prompt)."""
    with unit_of_work(app_engine) as repo:
        snap = None if reprofile else repo.snapshots.latest_row(source_id)
        metadata = SourceMetadata.model_validate(snap.payload) if snap else None
        snapshot_id = snap.id if snap else None
        previous = repo.semantic_models.latest(source_id)
    if metadata is None:
        metadata, snapshot_id = refresh_metadata(source_id, password=password, app_engine=app_engine)

    _, engine = open_source(source_id, password, app_engine)
    try:
        model, report = build_semantic_model(engine, metadata, llm=llm, previous=previous)
    finally:
        engine.dispose()
    with unit_of_work(app_engine) as repo:
        model, created = repo.semantic_models.save_if_changed(
            model, created_by="engine", snapshot_id=snapshot_id,
            note="heuristics + LLM" if llm else "heuristics only")
    return ModelBuild(model, report, created, snapshot_id)
