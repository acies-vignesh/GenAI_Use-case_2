# AI-Powered Data Quality Rule Recommendation Engine

Connects to a source database, builds a semantic layer (JSON) from its metadata and profiled data, and uses an LLM to recommend data quality, hierarchy, and business rules. A data steward approves, rejects, or edits each rule. Every decision is saved and fed back into later runs on the same source.

See [docs/architecture-review.md](docs/architecture-review.md) for design decisions and open points.

## Stack

| Concern | Choice | Why |
|---|---|---|
| Backend API | Python + FastAPI | The data tooling (SQLAlchemy, pandas) and LLM SDKs are Python-first |
| Source connectivity | SQLAlchemy (inspector + engines) | One API for Postgres, MySQL, SQL Server, Oracle, SQLite |
| Profiling | pandas on bounded samples | Simple; can be swapped for SQL-side aggregates on large tables |
| App DB | SQLite (dev) → Postgres | Stores sources, semantic model versions, rules, reviews |
| LLM | OpenAI-compatible client (default: Groq + gpt-oss-120b, open-weight) | Open-source models; switch to Ollama (local) or others via `.env` only |
| Frontend | React + Vite (TypeScript) | Built in a later phase |
| Export | python-docx, reportlab | Word / PDF |

## Layout (mapped to the architecture layers)

```
backend/app/
  api/routes/        L1  REST endpoints the UI calls (the product entry point)
  cli.py                 Developer tool only: runs the same services without the UI
  connectivity/      L2  engines, encrypted credentials, source fingerprint
  extraction/        L3  schema crawler, sampler, profiler
  semantic/          L4  heuristics, relationship mapper, LLM enrichment, builder
  user_input/        L5  steward rules, thresholds, business context
  orchestration/     L6  context assembler, prompt builder, prompt templates
  agent/             L7  LLM client, rule generator, dedup
  rules/                 Rule logic shared by all origins: expressions (sqlglot), fingerprints, validation
  review/            L8  approve / reject / modify logic
  persistence/       L9  ORM models, repositories, Alembic migrations (app DB = data/app.db)
  export/            L10 DOCX / PDF / JSON exporters
  services/              Use cases shared by CLI and API: connect, refresh metadata, build model (+ generate)
  schemas/           Shared contracts: metadata, semantic model, rule
backend/tests/
backend/evaluation/  Scores engine output against the demo answer key (never imported by app/)
frontend/            React UI (later)
samples/source_db/   Demo retail DB generator + planted_issues.json (answer key)
samples/semantic_models/  Reference semantic model for the demo DB (steward-written)
samples/rules/       Reference rules covering all 44 planted issues (answer key)
data/                Runtime files (app DB, exports); gitignored
docs/
```

## Roadmap (one phase at a time)

- [x] **Phase 0: Foundation.** venv, install deps, config, demo source DB + planted-issue answer key
- [x] **Phase 1: Connectivity + schema extraction.** Connect, reflect schema, emit `metadata.json`
- [x] **Phase 2: Profiling.** Null rate, cardinality, ranges, patterns, top values
- [x] **Phase 3: Contracts.** Pydantic models for the semantic layer and rules (the JSON formats)
- [x] **Phase 4: Semantic layer.** Heuristics, relationships and hierarchies, LLM enrichment; write `semantic_model.json`
- [x] **Phase 5: Persistence.** App DB tables, repositories, semantic model versioning
- [x] **Phase 6: Rule generation.** Context assembler, prompt templates, LLM call, validation, dedup
- [x] **Phase 7: Review + feedback loop.** Approve/reject (reason codes)/modify/bulk, rule executor with evidence, rejection patterns in prompts, execution-based scoring
- [ ] **Phase 8: API + UI.** FastAPI routes, React screens
- [ ] **Phase 9: Export.** PDF / DOCX / JSON
- [ ] **Phase 10 (stretch).** Scheduled runs of approved rules (executor exists since Phase 7); push to a DQ platform (Great Expectations / Soda)

## Getting started (Phase 0)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy ..\.env.example ..\.env    # then fill in values
```
