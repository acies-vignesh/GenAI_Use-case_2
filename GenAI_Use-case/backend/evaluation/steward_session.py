"""A scripted, realistic steward review of the demo source's proposed rules (reproducible).

    cd backend
    .venv/Scripts/python -m evaluation.steward_session

Decisions were made from the execution evidence (`app.cli review list --evidence`), the way a steward
would. Rules are found by name, so the script works on any run that proposed them; missing names are
reported and skipped. Everything still proposed at the end is approved.
"""
import sys

from app.persistence.repositories import unit_of_work
from app.schemas.rule import RuleStatus
from app.services import review

REVIEWER = "steward.demo"
ROW = "row"

REJECT = [
    # (rule name, reason code, reason)
    ("Online orders cannot use Cash‑On‑Delivery", "wrong",
     "Inverts the business fact: COD is only disallowed for in-store (STORE) orders; online COD is allowed"),
    ("Line amount rounded to two decimals", "wrong", "Regex on a decimal column - every value fails"),
    ("ZONE rows must be top‑level", "wrong",
     "Logic error (AND instead of if-then) - flags every non-ZONE row; the hierarchy rule covers parent levels"),
    ("CITY rows must have a parent", "wrong",
     "Logic error (AND instead of if-then) - flags every non-CITY row; the hierarchy rule covers parent levels"),
    ("Active customers must have at least one order", "wrong", "New customers may not have ordered yet"),
    ("Parent level consistency", "duplicate", "Covered by the hierarchy parent-level rule; SQL uses entity names"),
    ("Products must link to subcategories", "duplicate", "Covered by the hierarchy attach-level rule"),
    ("All customers must link to CITY regions", "duplicate", "Covered by the hierarchy attach-level rule"),
    ("Order total matches sum of line amounts", "duplicate",
     "Covered by the aggregate rule on order; this SQL also has the discount sign wrong"),
    ("Unit price close to current product price", "not_relevant",
     "Line prices legitimately differ from list price (promotions, price revisions)"),
    ("Parent category null rate limit", "not_relevant", "Top-level departments have no parent by design"),
    ("Category code length bounds", "too_strict", "Department codes are 4 characters"),
    ("Brand must be from known list", "not_relevant", "New brands are added all the time"),
    ("Launch date must be on or after 2019-01-01", "not_relevant", "Arbitrary cut-off with no business meaning"),
    ("Phone number must be unique", "not_relevant",
     "Households may share a phone; duplicate customers are caught by email uniqueness"),
    ("Order ID must be present", "already_enforced", "Primary key - enforced by the database"),
    ("Order ID must be unique", "already_enforced", "Primary key - enforced by the database"),
    ("Product ID must be unique", "already_enforced", "Primary key - enforced by the database"),
    ("Region ID must be present", "already_enforced", "Primary key - enforced by the database"),
    ("Order Number must be present", "already_enforced", "Declared NOT NULL in the schema"),
    ("Order Date must be present", "already_enforced", "Declared NOT NULL in the schema"),
    ("First name length within reasonable bounds", "not_relevant", "Length limits on names add no value"),
    ("Last name length within reasonable bounds", "not_relevant", "Length limits on names add no value"),
    ("Product name length within reasonable bounds", "not_relevant", "Length limits on names add no value"),
    ("Region name length limits", "not_relevant", "Length limits on names add no value"),
    ("Region code length bounds", "not_relevant", "The code format rule already constrains it"),
    ("Disallow placeholder region names", "not_relevant", "Region names come from a controlled list"),
]

MODIFY = [
    # (rule name, changes, reason code, reason)
    ("Email address must be unique", {"tolerance": 0.0}, "too_loose",
     "No duplicate emails are acceptable - each duplicate is a duplicated customer"),
    ("Email missing rate acceptable", {"check.max_rate": 0.01, "severity": "high"}, "too_loose",
     "The business target is at most 1% missing emails"),
    ("Phone missing rate acceptable", {"check.max_rate": 0.01}, "too_loose",
     "The business target is at most 1% missing phones"),
    ("Customers must be at least 18 at signup",
     {"check": {"type": "row_condition", "expression": "signup_date >= DATE_ADD(date_of_birth, 18, 'YEAR')"},
      "rationale": "Legal requirement: customers must be 18+ at signup (53 customers currently violate it)"},
     "other", "Rewritten as a plain condition; the original rationale wrongly claimed 0 violations"),
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with unit_of_work() as repo:
        source_id = repo.sources.list()[0].id
        by_name = {}
        for r in repo.rules.list(source_id, status="proposed"):
            by_name.setdefault(r.name, r)

    missing = []
    for name, code, reason in REJECT:
        r = by_name.get(name)
        if r is None:
            missing.append(name)
            continue
        review.reject(r.rule_id, REVIEWER, code, reason)
    for name, changes, code, reason in MODIFY:
        r = by_name.get(name)
        if r is None:
            missing.append(name)
            continue
        review.modify(r.rule_id, REVIEWER, changes, reason=reason, reason_code=code)
    approved = review.bulk_approve(source_id, REVIEWER, note="reviewed with execution evidence")

    with unit_of_work() as repo:
        counts = repo.rules.counts(source_id)
    print(f"Rejected {len(REJECT) - sum(1 for n, *_ in REJECT if n in missing)}, "
          f"modified {len(MODIFY) - sum(1 for n, *_ in MODIFY if n in missing)}, bulk-approved {approved}")
    print(f"Rule library now: {counts}")
    if missing:
        print("Not found (skipped): " + "; ".join(missing))
    assert RuleStatus.PROPOSED.value not in counts


if __name__ == "__main__":
    main()
