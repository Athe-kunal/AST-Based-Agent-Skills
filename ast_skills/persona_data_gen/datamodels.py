"""Public data models for persona-driven retrieval data generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pydantic


@dataclass(frozen=True)
class PersonaGenerationPromptRow:
    """Prompt row for generating personas from a SKILL.md file."""

    custom_id: str
    relative_path: str
    skill_name: str
    prompt: str


@dataclass(frozen=True)
class PersonaQueryPromptRow:
    """Prompt row for generating one query from the persona set + SKILL.md."""

    custom_id: str
    relative_path: str
    skill_name: str
    personas: list[str]
    prompt: str


class PersonaProfile(pydantic.BaseModel):
    """Structured model output for persona generation (OpenAI JSON / batch message body)."""

    model_config = pydantic.ConfigDict(extra="forbid")

    personas: list[str] = pydantic.Field(min_length=5, max_length=5)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _normalize_personas_to_strings(cls, data: Any) -> Any:
        """Coerce ``{"persona": "..."}`` items (legacy shape) into plain strings."""
        if not isinstance(data, dict):
            return data
        raw = data.get("personas")
        if not isinstance(raw, list):
            return data
        normalized: list[str] = []
        for item in raw:
            if isinstance(item, str):
                normalized.append(item.strip())
            elif isinstance(item, dict):
                direct = item.get("persona")
                if isinstance(direct, str) and direct.strip():
                    normalized.append(direct.strip())
                    continue
                role = str(item.get("role", "")).strip()
                scenario = str(item.get("scenario", "")).strip()
                parts = [
                    role,
                    str(item.get("expertise_level", "")).strip(),
                    str(item.get("domain_context", "")).strip(),
                    str(item.get("intent_type", "")).strip(),
                    scenario,
                ]
                merged = "\n".join(part for part in parts if part)
                normalized.append(merged or json.dumps(item, ensure_ascii=False))
            else:
                normalized.append(str(item).strip())
        return {**data, "personas": normalized}


class PersonaQueryGeneration(pydantic.BaseModel):
    """Structured model output for persona-conditioned query generation."""

    model_config = pydantic.ConfigDict(extra="forbid")

    query: str
