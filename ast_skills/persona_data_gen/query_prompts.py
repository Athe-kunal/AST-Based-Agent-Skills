"""Prompt templates for persona-conditioned query generation."""

from __future__ import annotations

from collections.abc import Sequence

from ast_skills.data_gen.skills_data_collect import SkillMdRecord
from ast_skills.persona_data_gen.template_loader import (
    load_template_text,
    render_template,
)

QUERY_GENERATION_SYSTEM_PROMPT = load_template_text("query_generation_system.jinja")
QUERY_GENERATION_USER_PROMPT_TEMPLATE = load_template_text(
    "query_generation_user.jinja"
)


def build_persona_query_prompt(
    skill_record: SkillMdRecord, personas: Sequence[str]
) -> str:
    """Build a user prompt for query generation for a persona set and one skill."""
    skill_name = skill_record.metadata.get("name") or skill_record.relative_path
    persona_lines = [f"- {text}" for text in personas]
    personas_text = "\n".join(persona_lines)
    variables = {
        "skill_name": skill_name,
        "personas_text": personas_text,
        "content": skill_record.content,
    }
    return render_template(QUERY_GENERATION_USER_PROMPT_TEMPLATE, variables)
