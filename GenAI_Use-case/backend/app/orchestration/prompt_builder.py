"""Render prompt templates from the assembled context; chunk per entity/domain for large schemas.

Templates live in prompts/*.j2 so prompt wording can be improved without touching code.
"""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.orchestration.context_assembler import EntityContext

_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "prompts"), undefined=StrictUndefined,
                   keep_trailing_newline=True, trim_blocks=False)


def rule_generation_prompts(ctx: EntityContext, domain: str, max_rules: int) -> tuple[str, str]:
    """(system, user) for one entity."""
    system = _env.get_template("rule_generation_system.j2").render(max_rules=max_rules)
    user = _env.get_template("rule_generation_user.j2").render(ctx=ctx, domain=domain, max_rules=max_rules)
    return system, user
