"""Rule generation use case (Steps 6-7, 11-13): baseline rules + one LLM call per entity -> proposed rules.

    generate_rules(source_id, llm)                  the UI's "Generate rules" / "Generate more" button
    generate_rules(source_id, llm, store=False)     dry run: nothing written (used for evaluation)
"""
import json
from dataclasses import dataclass, field

from sqlalchemy.engine import Engine

from app.agent.dedup import KnownRules, unique_within_run
from app.agent.llm_client import LLMClient
from app.agent.rule_generator import EntityGeneration, generate_for_entity
from app.orchestration.context_assembler import build_entity_context
from app.orchestration.prompt_builder import rule_generation_prompts
from app.persistence.repositories import PromptHistory, unit_of_work
from app.rules.baseline import baseline_rules
from app.schemas.rule import Rule

DEFAULT_RULES_PER_ENTITY = 12
REPEAT_RUN_BUDGET = 5          # "generate more" on a well-covered entity: fewer, better rules
WELL_COVERED = 20              # rules already on file (baseline + approved + pending) for the entity


@dataclass
class GenerationResult:
    run_id: str | None
    stored: list[Rule] = field(default_factory=list)
    similar: list[tuple[Rule, Rule]] = field(default_factory=list)     # (new rule, look-alike on file)
    duplicates: list[Rule] = field(default_factory=list)              # dropped
    per_entity: dict[str, EntityGeneration] = field(default_factory=dict)
    usage: dict = field(default_factory=dict)

    @property
    def new_rules(self) -> list[Rule]:
        return self.stored + [r for r, _ in self.similar]

    @property
    def invalid(self) -> int:
        return sum(max(0, len(g.invalid) - g.repaired) for g in self.per_entity.values())


class NoSemanticModelError(RuntimeError):
    pass


def generate_rules(source_id: str, llm: LLMClient | None = None, include_baseline: bool = True,
                   use_context: bool = True, entities: list[str] | None = None, store: bool = True,
                   rules_per_entity: int = DEFAULT_RULES_PER_ENTITY,
                   app_engine: Engine | None = None) -> GenerationResult:
    with unit_of_work(app_engine) as repo:
        model = repo.semantic_models.latest(source_id)
        metadata = repo.snapshots.latest(source_id)
        if model is None or metadata is None:
            raise NoSemanticModelError("build the semantic model first")
        context_items = repo.context.active(source_id) if use_context else []
        history: PromptHistory = repo.rules.history_for_prompt(source_id)
        known = KnownRules()
        for r in repo.rules.list(source_id):
            known.remember(r)
        known.fingerprints |= repo.rules.original_fingerprints(source_id)
        run_id = None
        if store:
            run_id = repo.runs.start(source_id, model=getattr(llm, "model_name", "baseline-only"),
                                     semantic_model_version=model.version).id

    try:
        return _generate(source_id, llm, model, metadata, context_items, history, known, run_id, include_baseline,
                         entities, store, rules_per_entity, app_engine)
    except Exception as e:
        if run_id:
            with unit_of_work(app_engine) as repo:
                repo.runs.finish(run_id, "failed", error=f"{type(e).__name__}: {e}")
        raise


def _generate(source_id, llm, model, metadata, context_items, history, known, run_id, include_baseline,
              entities, store, rules_per_entity, app_engine) -> GenerationResult:
    targets = entities or [e.entity_id for e in model.entities]
    result = GenerationResult(run_id)
    usage_before = dict(getattr(llm, "usage", {}))

    baseline = baseline_rules(model, metadata, source_id) if include_baseline else []
    baseline = [r.model_copy(update={"generation_run_id": run_id}) for r in baseline if r.target.entity in targets]
    candidates: list[Rule] = list(baseline)

    if llm is not None:
        for entity_id in targets:
            ctx = build_entity_context(model, metadata, entity_id, context_items, history, baseline)
            budget = rules_per_entity if len(ctx.covered) < WELL_COVERED else min(rules_per_entity,
                                                                                  REPEAT_RUN_BUDGET)
            system, user = rule_generation_prompts(ctx, model.domain.description or model.domain.name, budget)
            gen = generate_for_entity(llm, system, user, model, entity_id, source_id, run_id)
            gen.budget = budget
            result.per_entity[entity_id] = gen
            candidates += gen.valid

    candidates, in_run_dupes = unique_within_run(candidates)
    result.duplicates += in_run_dupes

    if store:
        with unit_of_work(app_engine) as repo:
            for rule in candidates:
                added = repo.rules.add(rule, created_by="engine")
                if added.outcome == "duplicate":
                    result.duplicates.append(rule)
                elif added.outcome == "similar":
                    result.similar.append((rule, added.existing))
                else:
                    result.stored.append(rule)
    else:
        for rule in candidates:
            outcome = known.classify(rule)
            if outcome == "duplicate":
                result.duplicates.append(rule)
            elif outcome == "similar":
                result.similar.append((rule, known.similarity[rule.similarity_key]))
            else:
                result.stored.append(rule)
            known.remember(rule)

    usage_after = getattr(llm, "usage", {})
    result.usage = {k: usage_after.get(k, 0) - usage_before.get(k, 0) for k in usage_after}

    if store:
        with unit_of_work(app_engine) as repo:
            repo.runs.finish(
                run_id, "succeeded" if not any(g.error for g in result.per_entity.values()) else "failed",
                prompt=json.dumps({e: g.prompt for e, g in result.per_entity.items()}),
                raw_response=json.dumps({e: g.responses for e, g in result.per_entity.items()}),
                prompt_tokens=result.usage.get("prompt_tokens"),
                completion_tokens=result.usage.get("completion_tokens"),
                generated=len(result.new_rules), invalid=result.invalid,
                dropped_duplicate=len(result.duplicates), flagged_similar=len(result.similar),
                error="; ".join(f"{e}: {g.error}" for e, g in result.per_entity.items() if g.error) or None)
    return result
