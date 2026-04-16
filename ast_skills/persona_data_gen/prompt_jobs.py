"""Build JSONL prompt inputs for persona and query generation workflows."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import fire
from loguru import logger
from pydantic import ValidationError

from ast_skills.data_gen.dataset import (
    last_row_by_custom_id,
    parsed_batch_output_content,
    read_jsonl_paths,
)
from ast_skills.data_gen.openai_batch_chat_request import (
    DEFAULT_OPENAI_BATCH_MODEL,
    openai_batch_chat_completion_request,
)
from ast_skills.data_gen.synthetic_data_gen import (
    OPENAI_BATCH_MAX_FILE_TOKENS,
    _batch_chat_messages_token_count,
    _would_exceed_batch_file_tokens,
)
from ast_skills.data_gen.skills_data_collect import (
    SkillMdRecord,
    collect_english_skill_md_records,
)
from ast_skills.persona_data_gen.datamodels import (
    PersonaGenerationPromptRow,
    PersonaProfile,
    PersonaQueryGeneration,
    PersonaQueryPromptRow,
)
from ast_skills.persona_data_gen.persona_prompts import (
    PERSONA_GENERATION_SYSTEM_PROMPT,
    build_persona_generation_prompt,
)
from ast_skills.persona_data_gen.query_prompts import (
    QUERY_GENERATION_SYSTEM_PROMPT,
    build_persona_query_prompt,
)


PERSONA_GENERATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "persona_profile",
        "schema": PersonaProfile.model_json_schema(),
        "strict": True,
    },
}

QUERY_GENERATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "persona_query_generation_response",
        "schema": PersonaQueryGeneration.model_json_schema(),
        "strict": True,
    },
}


class _PersonaJsonlRow(NamedTuple):
    """Parsed persona output row with skill metadata and persona list."""

    relative_path: str
    skill_name: str
    personas: list[str]


PERSONA_GENERATION_BATCH_STEM = "persona_generation_batch"
QUERY_GENERATION_BATCH_STEM = "query_generation_batch"


def _enumerated_batch_input_path(
    output_dir: Path, filename_stem: str, shard_index: int
) -> Path:
    """Return ``output_dir / {stem}_{shard_index+1}.jsonl`` (1-based shard filenames)."""
    number = shard_index + 1
    return output_dir / f"{filename_stem}_{number}.jsonl"


def persona_generation_batch_input_base_path(output_dir: str) -> Path:
    """Return path to shard ``1``: ``output_dir/persona_generation_batch_1.jsonl``."""
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return _enumerated_batch_input_path(root, PERSONA_GENERATION_BATCH_STEM, 0)


def query_generation_batch_input_base_path(output_dir: str) -> Path:
    """Return path to shard ``1``: ``output_dir/query_generation_batch_1.jsonl``."""
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return _enumerated_batch_input_path(root, QUERY_GENERATION_BATCH_STEM, 0)


def _coerce_jsonl_path_list(paths: str | Sequence[str]) -> list[str]:
    """Normalize CLI or API input into a non-empty list of JSONL paths."""
    if isinstance(paths, str):
        stripped = paths.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            loaded = json.loads(stripped)
            if not isinstance(loaded, list):
                raise ValueError(f"Expected JSON array of paths, got {type(loaded)}")
            return [str(p).strip() for p in loaded if str(p).strip()]
        if "," in stripped:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        return [stripped]
    return [str(p).strip() for p in paths if str(p).strip()]


def load_filtered_skill_md_records(skills_root: str) -> list[SkillMdRecord]:
    """Load skill records using the shared data_gen collector and token filtering."""
    records = collect_english_skill_md_records(skills_root)
    logger.info(f"{skills_root=}")
    logger.info(f"{len(records)=}")
    return records


def _skill_name(skill_record: SkillMdRecord) -> str:
    """Get a skill display name from metadata with a stable fallback."""
    return skill_record.metadata.get("name") or skill_record.relative_path


def build_persona_generation_prompt_rows(
    skill_md_records: list[SkillMdRecord],
) -> list[PersonaGenerationPromptRow]:
    """Build one persona-generation prompt row per skill markdown record."""
    rows: list[PersonaGenerationPromptRow] = []
    for index, skill_md_record in enumerate(skill_md_records):
        custom_id = f"persona-{index}"
        row = PersonaGenerationPromptRow(
            custom_id=custom_id,
            relative_path=skill_md_record.relative_path,
            skill_name=_skill_name(skill_md_record),
            prompt=build_persona_generation_prompt(skill_md_record),
        )
        rows.append(row)
    logger.info(f"{len(rows)=}")
    return rows


def write_persona_generation_prompts_jsonl(
    rows: list[PersonaGenerationPromptRow],
    output_dir: str | Path,
    *,
    filename_stem: str = PERSONA_GENERATION_BATCH_STEM,
    model: str = DEFAULT_OPENAI_BATCH_MODEL,
    max_tokens: int = 2048,
    max_file_tokens: int = OPENAI_BATCH_MAX_FILE_TOKENS,
) -> int:
    """
    Write OpenAI Batch input JSONL (POST /v1/chat/completions) for persona generation.

    Shards are written as ``{filename_stem}_1.jsonl``, ``{filename_stem}_2.jsonl``, … under
    ``output_dir`` when cumulative ``body.messages`` tiktoken count would exceed
    ``max_file_tokens`` (same token budget policy as ``synthetic_data_gen`` batch helpers).

    Returns:
        Number of request lines written.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written_count = 0
    current_file_tokens = 0
    shard_index = 0
    shard_path = _enumerated_batch_input_path(out_dir, filename_stem, shard_index)
    output_file = shard_path.open("w", encoding="utf-8")

    try:
        for row in rows:
            messages = [
                {"role": "system", "content": PERSONA_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": row.prompt},
            ]
            request_line = openai_batch_chat_completion_request(
                custom_id=row.custom_id,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format=PERSONA_GENERATION_RESPONSE_FORMAT,
            )
            line_tokens = _batch_chat_messages_token_count(request_line)
            would_exceed = _would_exceed_batch_file_tokens(
                current_file_tokens=current_file_tokens,
                next_line_tokens=line_tokens,
                max_file_tokens=max_file_tokens,
            )
            if would_exceed:
                if current_file_tokens == 0:
                    logger.warning(
                        "Single batch row exceeds max file token budget; writing it anyway: "
                        f"{line_tokens=} {max_file_tokens=} {shard_path=}"
                    )
                else:
                    output_file.close()
                    shard_index += 1
                    shard_path = _enumerated_batch_input_path(
                        out_dir, filename_stem, shard_index
                    )
                    logger.info(
                        "Continuing batch row writes in next shard: "
                        f"{shard_path=} {current_file_tokens=} {line_tokens=} "
                        f"{max_file_tokens=}"
                    )
                    output_file = shard_path.open("w", encoding="utf-8")
                    current_file_tokens = 0

            output_file.write(json.dumps(request_line, ensure_ascii=False) + "\n")
            current_file_tokens += line_tokens
            written_count += 1
    finally:
        output_file.close()

    logger.info(f"{out_dir=} {filename_stem=} {shard_index=}")
    logger.info(f"{written_count=}")
    logger.info(f"{current_file_tokens=} (last shard)")
    logger.info(f"{max_file_tokens=}")
    logger.info(f"{model=}")
    logger.info(f"{max_tokens=}")
    return written_count


def _persona_text_from_dict(raw_persona: dict) -> str:
    """Parse one raw persona object into a single natural-language string."""
    direct = raw_persona.get("persona")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    role = str(raw_persona.get("role", "")).strip()
    scenario = str(raw_persona.get("scenario", "")).strip()
    parts = [
        role,
        str(raw_persona.get("expertise_level", "")).strip(),
        str(raw_persona.get("domain_context", "")).strip(),
        str(raw_persona.get("intent_type", "")).strip(),
        scenario,
    ]
    merged = "\n".join(part for part in parts if part)
    return merged or json.dumps(raw_persona, ensure_ascii=False)


def _persona_from_string(raw_persona: str) -> str:
    """Normalize a plain-text persona string."""
    return raw_persona.strip()


def _parse_persona_jsonl_line(raw_row: dict) -> _PersonaJsonlRow:
    """Parse one persona JSONL row supporting both structured and list[str] personas."""
    relative_path = str(raw_row.get("relative_path", "")).strip()
    skill_name = str(raw_row.get("skill_name", "")).strip() or relative_path
    raw_personas = raw_row.get("personas", [])

    personas: list[str] = []
    if isinstance(raw_personas, list):
        for raw_persona in raw_personas:
            if isinstance(raw_persona, dict):
                personas.append(_persona_text_from_dict(raw_persona))
            elif isinstance(raw_persona, str):
                personas.append(_persona_from_string(raw_persona))

    if not personas:
        personas_text = _extract_personas_text(raw_row)
        personas = _parse_persona_list_text(personas_text)

    return _PersonaJsonlRow(
        relative_path=relative_path,
        skill_name=skill_name,
        personas=personas,
    )


def _extract_personas_text(raw_row: dict) -> str:
    """Extract persona text from common output keys in a JSONL row."""
    text_keys = ["personas_text", "response_text", "model_output", "personas"]
    for key in text_keys:
        value = raw_row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _clean_persona_line(line: str) -> str:
    """Normalize one raw persona line by removing numbering and markdown bullets."""
    cleaned = line.strip()
    cleaned = cleaned.lstrip("-*").strip()

    if ". " in cleaned:
        prefix, suffix = cleaned.split(". ", maxsplit=1)
        if prefix.isdigit() and suffix:
            return suffix.strip()
    return cleaned


def _parse_persona_list_text(personas_text: str) -> list[str]:
    """Parse a numbered or bulleted persona list from natural-language text."""
    blocks = _split_persona_text_blocks(personas_text)
    personas: list[str] = []

    for block in blocks:
        cleaned = _clean_persona_line(block)
        if not cleaned:
            continue
        personas.append(_persona_from_string(cleaned))
    return personas


def _split_persona_text_blocks(personas_text: str) -> list[str]:
    """Split model output into persona-sized text blocks."""
    numbered_item_re = re.compile(r"(?:^|\n)\s*\d+\.\s+")
    has_numbered_items = bool(numbered_item_re.search(personas_text))
    if has_numbered_items:
        parts = numbered_item_re.split(personas_text)
        return [part.strip() for part in parts if part.strip()]

    paragraph_parts = personas_text.split("\n\n")
    paragraphs = [part.strip() for part in paragraph_parts if part.strip()]
    if paragraphs:
        return paragraphs

    lines = [line.strip() for line in personas_text.splitlines() if line.strip()]
    return lines


def read_persona_generation_output_jsonl(input_path: str) -> list[_PersonaJsonlRow]:
    """Load persona generation outputs from JSONL."""
    path = Path(input_path)
    parsed_rows: list[_PersonaJsonlRow] = []

    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if not stripped:
                continue
            raw_row = json.loads(stripped)
            parsed_row = _parse_persona_jsonl_line(raw_row)
            if parsed_row.relative_path and parsed_row.personas:
                parsed_rows.append(parsed_row)

    logger.info(f"{input_path=}")
    logger.info(f"{len(parsed_rows)=}")
    return parsed_rows


def read_persona_generation_batch_output_paths(
    paths: Sequence[str],
    skill_md_records: list[SkillMdRecord],
) -> list[_PersonaJsonlRow]:
    """
    Load persona generation results from OpenAI Batch output JSONL file(s).

    Rows are combined then de-duplicated by ``custom_id`` (last wins). Each
    ``persona-{index}`` id maps to ``skill_md_records[index]``; message content is parsed
    with the same rules as ``dataset.parsed_batch_output_content`` and validated as
    ``PersonaProfile``.
    """
    path_list = [p for p in paths if p.strip()]
    if not path_list:
        logger.warning(f"{paths=} produced empty path list")
        return []

    rows = read_jsonl_paths(list(path_list))
    by_custom_id = last_row_by_custom_id(rows)
    logger.info(f"{len(rows)=} {len(by_custom_id)=}")

    parsed_rows: list[_PersonaJsonlRow] = []
    for index, skill_record in enumerate(skill_md_records):
        custom_id = f"persona-{index}"
        batch_row = by_custom_id.get(custom_id)
        if batch_row is None:
            logger.warning(f"No batch output row for {custom_id=}")
            continue

        payload = parsed_batch_output_content(batch_row)
        if payload is None:
            logger.warning(f"Unparseable batch output for {custom_id=}")
            continue
        try:
            profile = PersonaProfile.model_validate(payload)
        except ValidationError as exc:
            logger.warning(f"Invalid PersonaProfile for {custom_id=}: {exc}")
            continue

        personas = list(profile.personas)
        relative_path = skill_record.relative_path
        skill_name = _skill_name(skill_record)
        parsed_rows.append(
            _PersonaJsonlRow(
                relative_path=relative_path,
                skill_name=skill_name,
                personas=personas,
            )
        )

    logger.info(f"{path_list=} {len(parsed_rows)=}")
    return parsed_rows


def _map_skill_records_by_relative_path(
    skill_md_records: list[SkillMdRecord],
) -> dict[str, SkillMdRecord]:
    """Index skill records by relative path for fast joins with persona output."""
    mapping = {record.relative_path: record for record in skill_md_records}
    logger.info(f"{len(mapping)=}")
    return mapping


def build_persona_query_prompt_rows(
    skill_md_records: list[SkillMdRecord],
    persona_rows: list[_PersonaJsonlRow],
) -> list[PersonaQueryPromptRow]:
    """Build one query-generation prompt row per skill, using the full persona list."""
    skill_records_by_path = _map_skill_records_by_relative_path(skill_md_records)
    rows: list[PersonaQueryPromptRow] = []

    for persona_row in persona_rows:
        skill_record = skill_records_by_path.get(persona_row.relative_path)
        if skill_record is None:
            logger.warning(f"Missing skill for {persona_row.relative_path=}")
            continue

        if not persona_row.personas:
            logger.warning(f"Empty personas for {persona_row.relative_path=}")
            continue

        custom_id = f"query-{persona_row.relative_path}"
        row = PersonaQueryPromptRow(
            custom_id=custom_id,
            relative_path=persona_row.relative_path,
            skill_name=persona_row.skill_name,
            personas=list(persona_row.personas),
            prompt=build_persona_query_prompt(skill_record, list(persona_row.personas)),
        )
        rows.append(row)

    logger.info(f"{len(rows)=}")
    return rows


def write_persona_query_prompts_jsonl(
    rows: list[PersonaQueryPromptRow],
    output_dir: str | Path,
    *,
    filename_stem: str = QUERY_GENERATION_BATCH_STEM,
    model: str = DEFAULT_OPENAI_BATCH_MODEL,
    max_tokens: int = 2048,
    max_file_tokens: int = OPENAI_BATCH_MAX_FILE_TOKENS,
) -> int:
    """
    Write OpenAI Batch input JSONL (POST /v1/chat/completions) for query generation.

    Shards as ``{filename_stem}_1.jsonl``, ``{filename_stem}_2.jsonl``, … using the same
    token budget rules as ``write_persona_generation_prompts_jsonl``.

    Returns:
        Number of request lines written.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written_count = 0
    current_file_tokens = 0
    shard_index = 0
    shard_path = _enumerated_batch_input_path(out_dir, filename_stem, shard_index)
    output_file = shard_path.open("w", encoding="utf-8")

    try:
        for row in rows:
            messages = [
                {"role": "system", "content": QUERY_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": row.prompt},
            ]
            request_line = openai_batch_chat_completion_request(
                custom_id=row.custom_id,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format=QUERY_GENERATION_RESPONSE_FORMAT,
            )
            line_tokens = _batch_chat_messages_token_count(request_line)
            would_exceed = _would_exceed_batch_file_tokens(
                current_file_tokens=current_file_tokens,
                next_line_tokens=line_tokens,
                max_file_tokens=max_file_tokens,
            )
            if would_exceed:
                if current_file_tokens == 0:
                    logger.warning(
                        "Single batch row exceeds max file token budget; writing it anyway: "
                        f"{line_tokens=} {max_file_tokens=} {shard_path=}"
                    )
                else:
                    output_file.close()
                    shard_index += 1
                    shard_path = _enumerated_batch_input_path(
                        out_dir, filename_stem, shard_index
                    )
                    logger.info(
                        "Continuing batch row writes in next shard: "
                        f"{shard_path=} {current_file_tokens=} {line_tokens=} "
                        f"{max_file_tokens=}"
                    )
                    output_file = shard_path.open("w", encoding="utf-8")
                    current_file_tokens = 0

            output_file.write(json.dumps(request_line, ensure_ascii=False) + "\n")
            current_file_tokens += line_tokens
            written_count += 1
    finally:
        output_file.close()

    logger.info(f"{out_dir=} {filename_stem=} {shard_index=}")
    logger.info(f"{shard_index=}")
    logger.info(f"{written_count=}")
    logger.info(f"{current_file_tokens=} (last shard)")
    logger.info(f"{max_file_tokens=}")
    logger.info(f"{model=}")
    logger.info(f"{max_tokens=}")
    return written_count


DEFAULT_PERSONA_BATCH_INPUT_DIR = Path("data/persona_batch_inputs")
DEFAULT_PERSONA_BATCH_RESULTS_DIR = Path("data/persona_batch_results")
DEFAULT_PERSONA_BATCH_DONE_DIR = Path("data/persona_batch_done")


class PersonaPromptJobs:
    """CLI jobs for persona and persona-conditioned query prompt generation."""

    def submit_batches(
        self,
        input_dir: str = str(DEFAULT_PERSONA_BATCH_INPUT_DIR),
        results_dir: str = str(DEFAULT_PERSONA_BATCH_RESULTS_DIR),
        done_dir: str = str(DEFAULT_PERSONA_BATCH_DONE_DIR),
    ) -> None:
        """
        Upload every ``*.jsonl`` under ``input_dir`` to the OpenAI Batch API, wait for
        completion, write outputs under ``results_dir``, then move each input into
        ``done_dir`` (same flow as ``openai_batch_jobs.run_batch_mode``).

        Requires ``OPENAI_API_KEY``. Uses ``OPENAI_PROJECT`` when set (see
        ``openai_batch_jobs.resolve_openai_project``).
        """
        from ast_skills.data_gen.openai_batch_jobs import run_batch_mode

        in_path = Path(input_dir).expanduser()
        in_path.mkdir(parents=True, exist_ok=True)
        jsonl_files = sorted(in_path.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {in_path}")

        logger.info(f"{in_path=} {len(jsonl_files)=} {results_dir=} {done_dir=}")
        run_batch_mode(
            jsonl_files,
            batch_output_dir=Path(results_dir).expanduser(),
            input_done_dir=Path(done_dir).expanduser(),
        )

    def submit_batch_files(
        self,
        jsonl_paths: str | Sequence[str],
        results_dir: str = str(DEFAULT_PERSONA_BATCH_RESULTS_DIR),
        done_dir: str = str(DEFAULT_PERSONA_BATCH_DONE_DIR),
    ) -> None:
        """
        Submit explicit batch input JSONL paths (e.g. sharded ``persona_prompts.jsonl`` files).

        Same OpenAI Batch lifecycle as ``submit_batches``, but does not glob a directory.
        Each path must exist; completed inputs are moved into ``done_dir``.
        """
        from ast_skills.data_gen.openai_batch_jobs import (
            build_client,
            process_one_jsonl_batch,
            resolve_openai_project,
        )

        path_strings = _coerce_jsonl_path_list(jsonl_paths)
        paths = [Path(p).expanduser().resolve() for p in path_strings]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Batch input not found: {path}")

        res_dir = Path(results_dir).expanduser()
        dn_dir = Path(done_dir).expanduser()
        res_dir.mkdir(parents=True, exist_ok=True)
        dn_dir.mkdir(parents=True, exist_ok=True)

        project = resolve_openai_project()
        logger.info(f"{project=} {paths=} {res_dir=} {dn_dir=}")
        client = build_client(project=project)
        for path in paths:
            try:
                process_one_jsonl_batch(
                    client=client,
                    path=path,
                    batch_output_dir=res_dir,
                    input_done_dir=dn_dir,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception(f"{path=} {exc=}")

    def build_persona_prompts(
        self,
        skills_root: str,
        output_dir: str,
        model: str = DEFAULT_OPENAI_BATCH_MODEL,
        max_tokens: int = 4096,
        max_file_tokens: int = OPENAI_BATCH_MAX_FILE_TOKENS,
    ) -> None:
        """
        Generate OpenAI Batch input JSONL for persona generation from SKILL.md files.

        ``output_dir`` is a directory. Shards are ``persona_generation_batch_1.jsonl``,
        ``persona_generation_batch_2.jsonl``, … (1-based enumeration).
        """
        skill_md_records = load_filtered_skill_md_records(skills_root)
        rows = build_persona_generation_prompt_rows(skill_md_records)
        first_path = persona_generation_batch_input_base_path(output_dir)
        written = write_persona_generation_prompts_jsonl(
            rows,
            output_dir,
            model=model,
            max_tokens=max_tokens,
            max_file_tokens=max_file_tokens,
        )
        logger.info(f"{written=} {output_dir=} {first_path=}")

    def build_query_prompts(
        self,
        skills_root: str,
        persona_jsonl_paths: str | Sequence[str],
        output_dir: str,
        model: str = DEFAULT_OPENAI_BATCH_MODEL,
        max_tokens: int = 2048,
        max_file_tokens: int = OPENAI_BATCH_MAX_FILE_TOKENS,
    ) -> None:
        """
        Generate OpenAI Batch input JSONL for query generation from persona batch outputs.

        ``persona_jsonl_paths`` is one path, a comma-separated list, or a JSON array string
        of paths to OpenAI Batch *output* JSONL files. Rows are read and merged; ``custom_id``
        is used to align each ``persona-{index}`` completion with ``skill_md_records[index]``.

        ``output_dir`` is a directory. Shards are ``query_generation_batch_1.jsonl``,
        ``query_generation_batch_2.jsonl``, … (1-based enumeration).
        """
        skill_md_records = load_filtered_skill_md_records(skills_root)
        path_list = _coerce_jsonl_path_list(persona_jsonl_paths)
        persona_rows = read_persona_generation_batch_output_paths(
            path_list, skill_md_records
        )
        rows = build_persona_query_prompt_rows(skill_md_records, persona_rows)
        first_path = query_generation_batch_input_base_path(output_dir)
        written = write_persona_query_prompts_jsonl(
            rows,
            output_dir,
            model=model,
            max_tokens=max_tokens,
            max_file_tokens=max_file_tokens,
        )
        logger.info(f"{written=} {output_dir=} {first_path=}")


if __name__ == "__main__":
    fire.Fire(PersonaPromptJobs)
