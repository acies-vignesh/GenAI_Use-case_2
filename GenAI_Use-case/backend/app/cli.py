"""Command-line DEVELOPER tool: runs the pipeline without the API/UI (tests, debugging, scripting).

Users never use this - in the product, sources are connected through the UI connection form and
every step below is an API call. Both paths call the same functions in app/services/pipeline.py.

    cd backend
    .venv/Scripts/python -m app.cli db upgrade                       # create / migrate the app database
    .venv/Scripts/python -m app.cli extract                          # structure only (Phase 1)
    .venv/Scripts/python -m app.cli profile                          # structure + data profile (Phase 2)
    .venv/Scripts/python -m app.cli semantic [--no-llm] [--reprofile] # semantic model (Phase 4)
    .venv/Scripts/python -m app.cli generate [--dry-run] [--no-context] # proposed rules (Phase 6)
    .venv/Scripts/python -m app.cli review list --evidence | show <id> | approve | reject | modify (Phase 7)
    .venv/Scripts/python -m app.cli sources                          # known sources + rule counts
    .venv/Scripts/python -m app.cli context add "COD is not offered in stores" --entity order
    .venv/Scripts/python -m app.cli rules import path/to/rules.json
    .venv/Scripts/python -m app.cli rules list [--status approved]

--url defaults to SOURCE_DB_URL in .env - a developer shortcut pointing at the demo database.
"""
import argparse
import json
import sys
from pathlib import Path

from app.config import PROJECT_ROOT, settings
from app.connectivity.connectors import ConnectionConfig
from app.persistence.database import upgrade
from app.persistence.repositories import unit_of_work
from app.schemas.metadata import SourceMetadata
from app.schemas.rule import Rule
from app.services import pipeline

DEFAULT_METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


# --------------------------------------------------------------------------- helpers
def _config(args: argparse.Namespace) -> ConnectionConfig:
    return ConnectionConfig.from_url(args.url or settings.source_db_url, schema_name=args.schema)


def _connect(args: argparse.Namespace) -> str:
    """The CLI's stand-in for the UI connection form."""
    return pipeline.connect_source(_config(args), remember_password=not args.no_store_password)


def _write_json(metadata: SourceMetadata, out: str) -> Path:
    """Developer convenience: a readable copy next to the app DB (evaluation scripts read it)."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"metadata_{metadata.source_id}.json"
    out_file.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return out_file


def _source_id(args: argparse.Namespace) -> str:
    from app.connectivity.fingerprint import source_fingerprint
    return args.source or source_fingerprint(_config(args))


# --------------------------------------------------------------------------- pipeline commands
def cmd_extract(args: argparse.Namespace) -> None:
    metadata, _ = pipeline.refresh_metadata(_connect(args), profile=False,
                                            include_row_counts=not args.no_row_counts)
    out_file = _write_json(metadata, args.out)

    print(f"Source   : {metadata.database} ({metadata.dialect})  id={metadata.source_id}")
    print(f"{'table':<13}{'rows':>8}  {'cols':>4}  {'PK':<16} foreign keys")
    for t in metadata.tables:
        fks = ", ".join(
            f"{','.join(fk.columns)}->{fk.referred_table}{' (self)' if fk.is_self_referencing else ''}"
            for fk in t.foreign_keys
        ) or "-"
        rows = f"{t.row_count:,}" if t.row_count is not None else "?"
        print(f"{t.name:<13}{rows:>8}  {len(t.columns):>4}  {','.join(t.primary_key):<16} {fks}")
    print(f"Saved    : snapshot in app DB + {out_file}")


def cmd_profile(args: argparse.Namespace) -> None:
    metadata, _ = pipeline.refresh_metadata(_connect(args), sample_limit=args.sample_limit)
    out_file = _write_json(metadata, args.out)

    print(f"Source   : {metadata.database} ({metadata.dialect})  id={metadata.source_id}")
    print(f"{'table':<13}{'rows':>8}  {'sample':>14}  {'cols w/ nulls':>13}  {'sensitive':>9}  FK orphans")
    for t in metadata.tables:
        with_nulls = sum(1 for c in t.columns if c.profile.null_count)
        sensitive = sum(1 for c in t.columns if c.profile.sensitive_hint)
        orphans = ", ".join(f"{','.join(fk.columns)}={fk.orphan_count}" for fk in t.foreign_keys) or "-"
        sample = f"{t.sample_size:,} {t.sample_method}"
        print(f"{t.name:<13}{t.row_count:>8,}  {sample:>14}  {with_nulls:>13}  {sensitive:>9}  {orphans}")
    print(f"Saved    : snapshot in app DB + {out_file}")


def cmd_semantic(args: argparse.Namespace) -> None:
    source_id = _connect(args)
    llm = None
    if not args.no_llm:
        from app.agent.llm_client import OpenAICompatibleClient
        llm = OpenAICompatibleClient(settings)

    build = pipeline.build_model(source_id, llm=llm, reprofile=args.reprofile)
    model, report = build.model, build.report

    if not build.created:
        print(f"No changes - semantic model stays at v{model.version}")
    print(f"Semantic model v{model.version}  ({'heuristics + LLM' if llm else 'heuristics only'})")
    print(f"Domain   : {model.domain.name}" + (f" - {model.domain.description}" if model.domain.description else ""))
    for e in model.entities:
        print(f"  {e.entity_id:<12} {e.entity_type.value:<17} {len(e.attributes):>2} attrs  "
              f"key={','.join(e.business_key) or '-'}  {e.grain}")
    print("Relationships:")
    for r in model.relationships:
        print(f"  {r.from_entity}.{','.join(r.from_attributes)} -> {r.to_entity}  "
              f"[{r.kind}{', mandatory' if r.mandatory else ''}]")
    print("Hierarchies:")
    for hy in model.hierarchies:
        attached = ", ".join(f"{a.entity}@{a.must_attach_at}" for a in hy.attached_entities) or "-"
        print(f"  {hy.name}: {' > '.join(hy.levels)}   attached: {attached}")
        for ev in hy.evidence:
            print(f"      ! {ev}")
    anomalies = [(e.entity_id, a.column, len(a.value_anomalies)) for e in model.entities for a in e.attributes
                 if a.value_anomalies]
    print("Value anomalies: " + (", ".join(f"{e}.{c}={n}" for e, c, n in anomalies) or "-"))
    print(f"Needs review ({len(report.needs_review)}): " + ("; ".join(report.needs_review) or "-"))
    if report.inferred_links:
        print("Inferred relationships:")
        for link in report.inferred_links:
            print(f"  {link}")
    if report.llm_used:
        print(f"LLM changes ({len(report.llm_changes)}):")
        for c in report.llm_changes:
            print(f"  - {c}")
        print(f"LLM usage: {llm.usage}")
    if report.drift:
        print("Schema drift: " + "; ".join(report.drift))
    if report.kept_locked:
        print("Kept steward-locked: " + ", ".join(report.kept_locked))
    if build.created:
        print(f"Saved    : semantic model v{model.version} in app DB")


def cmd_generate(args: argparse.Namespace) -> None:
    from app.services.generation import generate_rules
    source_id = _connect(args)
    llm = None
    if not args.no_llm:
        from app.agent.llm_client import OpenAICompatibleClient
        llm = OpenAICompatibleClient(settings, max_tokens=16000)
    result = generate_rules(source_id, llm=llm, use_context=not args.no_context, entities=args.entity,
                            store=not args.dry_run, rules_per_entity=args.max_rules)

    print(f"Generation {'(dry run - nothing stored)' if args.dry_run else 'run ' + str(result.run_id)}"
          f"  context={'off' if args.no_context else 'on'}  llm={'off' if llm is None else llm.model_name}")
    for eid, g in result.per_entity.items():
        status = f"ERROR {g.error}" if g.error else (f"{len(g.valid)} valid, {len(g.invalid)} invalid"
                                                      f" ({g.repaired} repaired), {g.off_target} off-target")
        print(f"  {eid:<11} {status}")
        for cand, errs in g.invalid:
            print(f"      x {cand.get('name', '?')}: {errs[0][:110]}")
    by_origin: dict[str, int] = {}
    for r in result.new_rules:
        by_origin[r.origin.value] = by_origin.get(r.origin.value, 0) + 1
    print(f"New rules: {len(result.new_rules)} {by_origin}  similar-flagged: {len(result.similar)}  "
          f"duplicates dropped: {len(result.duplicates)}")
    if result.usage:
        print(f"LLM usage: {result.usage}")
    for r in sorted(result.new_rules, key=lambda r: (r.target.entity, r.origin.value)):
        print(f"  [{r.origin.value[:3]}] {r.severity.value:<8} {r.target.entity:<10} {r.check.type:<22} {r.name}")
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps([r.model_dump(mode="json") for r in result.new_rules],
                                                   indent=1), encoding="utf-8")
        print(f"Rules written to {args.save_json}")


# --------------------------------------------------------------------------- review commands (Phase 7)
def _print_item(item, verbose: bool = False) -> None:
    r, ex = item.rule, item.execution
    evid = ""
    if ex is not None:
        evid = "  ERROR" if ex.error else (f"  fails {ex.failed:,}/{ex.total:,} ({ex.failure_rate:.2%})"
                                            + ("" if ex.passed else "  -> ALERT"))
    print(f"{r.rule_id[:8]} {r.status.value:<9} {r.origin.value[:3]} {r.severity.value:<8} {r.target.entity:<10} "
          f"{r.check.type:<22} {r.name}{evid}")
    for f in item.flags:
        print(f"         ! {f}")
    if verbose:
        print(f"         check:     {r.check.model_dump(exclude_none=True)}")
        if r.filter:
            print(f"         filter:    {r.filter}")
        print(f"         columns:   {r.target.columns}   tolerance {r.tolerance}")
        print(f"         rationale: {r.rationale}")
        for rv in r.reviews:
            print(f"         review:    {rv.decision} by {rv.reviewer}"
                  + (f" [{rv.reason_code.value}]" if rv.reason_code else "") + (f" - {rv.reason}" if rv.reason else ""))
        if ex is not None and ex.examples:
            print(f"         examples:  {ex.examples[:3]}")


def cmd_review_list(args: argparse.Namespace) -> None:
    from app.services.review import review_queue
    items = review_queue(_source_id(args), status=args.status, entity=args.entity, origin=args.origin,
                         with_evidence=args.evidence)
    for item in items:
        _print_item(item)
    print(f"{len(items)} rule(s)")


def cmd_review_show(args: argparse.Namespace) -> None:
    from app.services.review import resolve_rule_id, review_queue
    rid = resolve_rule_id(args.rule_id)
    items = [i for i in review_queue(_source_id(args), status=None, with_evidence=False) if i.rule.rule_id == rid]
    from app.services.review import _flags, evidence
    items[0].execution = evidence(_source_id(args), [items[0].rule])[rid]
    items[0].flags = _flags(items[0])
    _print_item(items[0], verbose=True)


def cmd_review_approve(args: argparse.Namespace) -> None:
    from app.services.review import approve, bulk_approve, resolve_rule_id
    if args.all:
        n = bulk_approve(_source_id(args), args.by, entity=args.entity, origin=args.origin, check_type=args.check,
                         note=args.note)
        print(f"Approved {n} proposed rule(s)")
    else:
        rules = approve([resolve_rule_id(x) for x in args.rule_ids], args.by, note=args.note)
        print(f"Approved {len(rules)} rule(s)")


def cmd_review_reject(args: argparse.Namespace) -> None:
    from app.services.review import reject, resolve_rule_id
    r = reject(resolve_rule_id(args.rule_id), args.by, args.code, args.reason)
    print(f"Rejected '{r.name}' [{args.code}]")


def cmd_review_modify(args: argparse.Namespace) -> None:
    from app.review.feedback import parse_value
    from app.services.review import modify, resolve_rule_id
    changes = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        changes[key.strip()] = parse_value(raw.strip())
    r = modify(resolve_rule_id(args.rule_id), args.by, changes, reason=args.reason, reason_code=args.code)
    print(f"Modified and approved '{r.name}': {r.check.model_dump(exclude_none=True)}")


def cmd_review_add(args: argparse.Namespace) -> None:
    from app.services.review import add_steward_rule
    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    for item in data if isinstance(data, list) else [data]:
        result = add_steward_rule(_source_id(args), item, args.by)
        print(f"{result.outcome}: {result.rule.name}")


# --------------------------------------------------------------------------- app database commands
def cmd_db_upgrade(args: argparse.Namespace) -> None:
    print(f"App database: {settings.app_db_url}  (migrated to latest)")


def cmd_db_import_files(args: argparse.Namespace) -> None:
    """One-off: bring metadata/model JSON files from Phases 1-4 into the app database."""
    from app.connectivity.fingerprint import source_fingerprint
    from app.schemas.semantic_model import SemanticModel
    config = _config(args)
    sid = source_fingerprint(config)
    meta_file = Path(args.out) / f"metadata_{sid}.json"
    model_files = sorted((PROJECT_ROOT / "data" / "semantic_models").glob(f"semantic_model_{sid}_v*.json"),
                         key=lambda f: int(f.stem.rsplit("_v", 1)[1]))
    with unit_of_work() as repo:
        repo.sources.upsert(config)
        if repo.semantic_models.latest(sid):
            print("Source already has semantic models in the app DB - nothing imported.")
            return
        snapshot_id = None
        if meta_file.exists():
            snapshot_id = repo.snapshots.save(SourceMetadata.model_validate_json(meta_file.read_text("utf-8"))).id
            print(f"Imported snapshot from {meta_file.name}")
        for f in model_files:
            m = repo.semantic_models.save_version(SemanticModel.model_validate_json(f.read_text("utf-8")),
                                                  created_by="engine", snapshot_id=snapshot_id,
                                                  note=f"imported from {f.name}")
            print(f"Imported {f.name} as v{m.version}")


def cmd_sources(args: argparse.Namespace) -> None:
    with unit_of_work() as repo:
        rows = repo.sources.list()
        if not rows:
            print("No sources yet - run `profile` or `semantic` first.")
        for s in rows:
            counts = repo.rules.counts(s.id)
            versions = repo.semantic_models.history(s.id)
            print(f"{s.id}  {s.display_name} ({s.dialect})  last connected {s.last_connected_at:%Y-%m-%d %H:%M}")
            print(f"    semantic model: v{versions[-1].version if versions else '-'}   rules: "
                  + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none")
                  + f"   context items: {len(repo.context.active(s.id))}")


def cmd_context_add(args: argparse.Namespace) -> None:
    with unit_of_work() as repo:
        row = repo.context.add(_source_id(args), args.text, created_by=args.by, entity=args.entity,
                               attribute=args.attribute)
        print(f"Added context {row.id}")


def cmd_context_list(args: argparse.Namespace) -> None:
    with unit_of_work() as repo:
        for c in repo.context.active(_source_id(args)):
            scope = ".".join(x for x in (c.entity, c.attribute) if x) or "(source)"
            print(f"{c.id}  [{scope}]  {c.text}   - {c.created_by}")


def cmd_context_retire(args: argparse.Namespace) -> None:
    with unit_of_work() as repo:
        repo.context.retire(args.item_id)
        print(f"Retired {args.item_id}")


def cmd_rules_import(args: argparse.Namespace) -> None:
    """Accepts a list of rules, {"rules": [...]}, or the reference format {"rules": [{"rule": {...}}]}."""
    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    items = data["rules"] if isinstance(data, dict) else data
    sid = _source_id(args)
    outcomes: dict[str, int] = {}
    with unit_of_work() as repo:
        repo.sources.get(sid)
        for item in items:
            payload = {**(item.get("rule", item)), "source_id": sid, "origin": "steward"}
            result = repo.rules.add(Rule.model_validate(payload), created_by=args.by)
            outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1
    print(f"Imported into {sid}: " + ", ".join(f"{k} {v}" for k, v in outcomes.items()))


def cmd_rules_list(args: argparse.Namespace) -> None:
    with unit_of_work() as repo:
        rules = repo.rules.list(_source_id(args), status=args.status, entity=args.entity)
    for r in rules:
        print(f"{r.status.value:<9} {r.origin.value:<8} {r.severity.value:<8} {r.target.entity:<11} "
              f"{r.check.type:<22} {r.name}")
    print(f"{len(rules)} rule(s)")


# --------------------------------------------------------------------------- parser
def main() -> None:
    # Windows consoles default to cp1252; LLM text can contain any Unicode character.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="dq-engine")
    sub = parser.add_subparsers(dest="command", required=True)

    def source_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--url", help="SQLAlchemy URL of the source (default: SOURCE_DB_URL)")
        p.add_argument("--schema", help="Schema to crawl (server databases)")
        p.add_argument("--out", default=str(DEFAULT_METADATA_DIR), help="Metadata JSON output directory")
        p.add_argument("--no-store-password", action="store_true",
                       help="Don't keep the password (it must be given on every connection)")

    def target_args(p: argparse.ArgumentParser) -> None:
        source_args(p)
        p.add_argument("--source", help="Source id (default: fingerprint of --url / SOURCE_DB_URL)")
        p.add_argument("--by", default=settings.reviewer, help="Who is acting (default: REVIEWER from .env)")

    p = sub.add_parser("extract", help="Connect to a source and extract its schema metadata")
    source_args(p)
    p.add_argument("--no-row-counts", action="store_true", help="Skip COUNT(*) per table (large sources)")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("profile", help="Extract schema metadata and profile the data")
    source_args(p)
    p.add_argument("--sample-limit", type=int, default=settings.sample_row_limit)
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("semantic", help="Build the semantic model (heuristics + optional LLM)")
    source_args(p)
    p.add_argument("--no-llm", action="store_true", help="Heuristics only - no LLM call")
    p.add_argument("--reprofile", action="store_true", help="Profile again instead of reusing the snapshot")
    p.set_defaults(func=cmd_semantic)

    p = sub.add_parser("generate", help="Generate proposed rules (baseline + one LLM call per entity)")
    source_args(p)
    p.add_argument("--no-llm", action="store_true", help="Baseline (code-generated) rules only")
    p.add_argument("--no-context", action="store_true", help="Ignore steward business context")
    p.add_argument("--dry-run", action="store_true", help="Don't store anything")
    p.add_argument("--entity", action="append", help="Only these entities (repeatable)")
    p.add_argument("--max-rules", type=int, default=12, help="LLM rule budget per entity")
    p.add_argument("--save-json", help="Also write the new rules to this JSON file")
    p.set_defaults(func=cmd_generate)

    db = sub.add_parser("db", help="App database maintenance").add_subparsers(dest="db_cmd", required=True)
    db.add_parser("upgrade", help="Create / migrate the app database").set_defaults(func=cmd_db_upgrade)
    p = db.add_parser("import-files", help="Import Phase 1-4 JSON files for a source")
    source_args(p)
    p.set_defaults(func=cmd_db_import_files)

    sub.add_parser("sources", help="List known sources").set_defaults(func=cmd_sources)

    ctx = sub.add_parser("context", help="Steward business context").add_subparsers(dest="ctx_cmd", required=True)
    p = ctx.add_parser("add", help="Add a business fact")
    target_args(p)
    p.add_argument("text")
    p.add_argument("--entity")
    p.add_argument("--attribute")
    p.set_defaults(func=cmd_context_add)
    p = ctx.add_parser("list", help="List active context")
    target_args(p)
    p.set_defaults(func=cmd_context_list)
    p = ctx.add_parser("retire", help="Deactivate a context item")
    p.add_argument("item_id")
    p.set_defaults(func=cmd_context_retire)

    rules = sub.add_parser("rules", help="Rules").add_subparsers(dest="rules_cmd", required=True)
    p = rules.add_parser("import", help="Import steward-written rules from JSON")
    target_args(p)
    p.add_argument("file")
    p.set_defaults(func=cmd_rules_import)
    p = rules.add_parser("list", help="List rules")
    target_args(p)
    p.add_argument("--status", choices=["proposed", "approved", "rejected"])
    p.add_argument("--entity")
    p.set_defaults(func=cmd_rules_list)

    rv = sub.add_parser("review", help="Steward review").add_subparsers(dest="review_cmd", required=True)
    codes = ["wrong", "not_relevant", "too_strict", "too_loose", "already_enforced", "duplicate", "other"]
    p = rv.add_parser("list", help="Review queue")
    target_args(p)
    p.add_argument("--status", default="proposed", choices=["proposed", "approved", "rejected"])
    p.add_argument("--entity")
    p.add_argument("--origin", choices=["llm", "heuristic", "steward"])
    p.add_argument("--evidence", action="store_true", help="Run each rule on the source and show failures")
    p.set_defaults(func=cmd_review_list)
    p = rv.add_parser("show", help="One rule with evidence")
    target_args(p)
    p.add_argument("rule_id")
    p.set_defaults(func=cmd_review_show)
    p = rv.add_parser("approve", help="Approve rules (ids, or --all with filters)")
    target_args(p)
    p.add_argument("rule_ids", nargs="*")
    p.add_argument("--all", action="store_true", help="Approve every proposed rule matching the filters")
    p.add_argument("--entity")
    p.add_argument("--origin", choices=["llm", "heuristic", "steward"])
    p.add_argument("--check", help="Only this check type")
    p.add_argument("--note")
    p.set_defaults(func=cmd_review_approve)
    p = rv.add_parser("reject", help="Reject a rule (reason required)")
    target_args(p)
    p.add_argument("rule_id")
    p.add_argument("--code", required=True, choices=codes)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_review_reject)
    p = rv.add_parser("modify", help="Edit a rule and approve it")
    target_args(p)
    p.add_argument("rule_id")
    p.add_argument("--set", action="append", required=True, help="path=value, e.g. check.max_rate=0.01")
    p.add_argument("--reason")
    p.add_argument("--code", choices=codes)
    p.set_defaults(func=cmd_review_modify)
    p = rv.add_parser("add", help="Add steward-written rule(s) from JSON")
    target_args(p)
    p.add_argument("file")
    p.set_defaults(func=cmd_review_add)

    args = parser.parse_args()
    upgrade()   # every command may touch the app DB; no-op when already up to date
    args.func(args)


if __name__ == "__main__":
    main()
