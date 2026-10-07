"""Deterministic filter over LLM candidates using app.rules.fingerprint: drop exact matches of past rules, flag similar ones.

Two layers:
  - within one run  : the baseline and the LLM (or two entity calls) may propose the same rule
  - against history : handled by RuleRepo.add() when storing; `KnownRules` mirrors it for dry runs
"""
from dataclasses import dataclass, field

from app.schemas.rule import Rule


@dataclass
class KnownRules:
    fingerprints: set[str] = field(default_factory=set)        # current + original fingerprints
    similarity: dict[str, Rule] = field(default_factory=dict)  # similarity_key -> a rule on file

    def classify(self, rule: Rule) -> str:
        if rule.fingerprint in self.fingerprints:
            return "duplicate"
        return "similar" if rule.similarity_key in self.similarity else "stored"

    def remember(self, rule: Rule) -> None:
        self.fingerprints.add(rule.fingerprint)
        self.similarity.setdefault(rule.similarity_key, rule)


def unique_within_run(rules: list[Rule]) -> tuple[list[Rule], list[Rule]]:
    """(kept, dropped): first occurrence of each fingerprint wins (baseline comes first)."""
    seen: set[str] = set()
    kept, dropped = [], []
    for r in rules:
        (dropped if r.fingerprint in seen else kept).append(r)
        seen.add(r.fingerprint)
    return kept, dropped
