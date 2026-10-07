# Architecture Review

The proposed 10-layer architecture and the first-run vs repeat-run loop are sound. The points below fill gaps that would otherwise surface mid-build.

## Decisions

1. **Rules are structured objects, not free text.** Each rule has a category, a dimension, a target, a check type, parameters, a severity, and a rationale (`schemas/rule.py`). This is what makes the following possible:
   - deterministic "don't repeat rejected rules" checks
   - editing individual fields in the UI
   - exporting to an execution platform later

2. **Dedup is enforced in code, not left to the LLM.** The prompt includes the approved/rejected history, but a rule fingerprint (target + check type + normalized parameters) also filters candidates after generation (`agent/dedup.py`). LLMs do not reliably follow "never repeat X".

3. **"Learning" means memory in the prompt context, not fine-tuning.** History grows with every run, so later runs summarize it: approved rules become the baseline, and rejected rules are grouped with their rejection reasons. Each rejection should capture a reason.

4. **Privacy: raw rows don't go to the LLM by default.** Only aggregated profile stats and masked/synthetic example values are sent (`SEND_RAW_SAMPLES_TO_LLM=false`). Columns flagged as PII in the semantic layer are never sampled into prompts.

5. **Credentials are encrypted at rest** (Fernet). Use a read-only DB user, and run sampling queries with row limits and timeouts.

6. **Source identity uses a fingerprint** (dialect + host + database + schema). The semantic model is versioned. On a repeat visit, schema drift is detected, and approved rules whose targets no longer exist are flagged.

7. **The semantic layer is editable by the steward**, just like rules. Mislabelled attributes are the main cause of bad recommendations downstream, so fixing them is cheaper than rejecting rules one by one.

8. **Hybrid semantic build.** A deterministic heuristic pass (names, regex patterns, FK/value overlap) runs first. The LLM then adds business meaning on top. This keeps the build cheaper, more explainable, and testable without an LLM.

9. **Rules are validated before review.** Generated rules are checked to confirm the target exists and the rule can actually run. Ideally each rule is also dry-run on the sample, so the steward sees a pass rate when deciding.

10. **Large schemas are prompted per entity or domain**, with related entities as context. The whole schema is not sent in one prompt.

## Open questions for the team

- Which source databases must be supported first (Postgres? SQL Server? Snowflake?)
- Which LLM provider or deployment is approved (direct API, Bedrock, Azure)?
- Single-user tool or multi-user with auth and audit trail of reviewers?
- Target DQ execution platform for the eventual push (Great Expectations, Soda, in-house)?
