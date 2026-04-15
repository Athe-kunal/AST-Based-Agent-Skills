"""Build training/validation datasets with hybrid-mined in-batch negatives."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, NamedTuple

import fire
from loguru import logger as log

from ast_skills.data_gen.datamodels import SummaryRetrieverDataModel, TrainingData
from ast_skills.retriever.search import hybrid_search


class _DatasetSplit(NamedTuple):
    """Training/validation dataset split."""

    train_rows: list[SummaryRetrieverDataModel]
    validation_rows: list[SummaryRetrieverDataModel]


class _NegativeTextLookup(NamedTuple):
    """Text lookup for negatives by custom_id."""

    summary_by_custom_id: dict[str, str]
    description_by_custom_id: dict[str, str]


class _HybridSearchConfig(NamedTuple):
    """Hybrid search configuration for mining hard negatives."""

    chroma_root_dir: str
    embedding_base_url: str
    embedding_model: str
    api_key: str
    top_k: int
    rrf_k: int


class _NegativeWindowConfig(NamedTuple):
    """Rank window and sample size for hard negatives."""

    window_start_rank: int
    window_end_rank: int
    negatives_per_row: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Reads a JSONL file into dictionaries."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            row = json.loads(stripped_line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected dict row at {line_number=}, {type(row)=}")
            rows.append(row)
    log.info(f"{path=}, {len(rows)=}")
    return rows


def _write_jsonl(path: Path, rows: list[TrainingData]) -> None:
    """Writes training rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")
    log.info(f"{path=}, {len(rows)=}")


def _rows_to_summary_models(rows: list[dict[str, Any]]) -> list[SummaryRetrieverDataModel]:
    """Converts dictionaries to ``SummaryRetrieverDataModel`` objects."""
    models: list[SummaryRetrieverDataModel] = []
    for row in rows:
        model = SummaryRetrieverDataModel(
            custom_id=str(row.get("custom_id", "")),
            markdown_content=str(row.get("markdown_content", "")),
            seed_questions=[str(question) for question in row.get("seed_questions", [])],
            summary=str(row.get("summary", "")),
            name=str(row.get("name", "")),
            description=str(row.get("description", "")),
            metadata={str(key): str(value) for key, value in dict(row.get("metadata", {})).items()},
        )
        models.append(model)
    log.info(f"{len(models)=}")
    return models


def _split_rows(
    rows: list[SummaryRetrieverDataModel],
    validation_ratio: float,
    seed: int,
) -> _DatasetSplit:
    """Splits rows into train/validation partitions."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1).")

    shuffled_rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled_rows)

    validation_count = max(1, int(len(shuffled_rows) * validation_ratio))
    if validation_count >= len(shuffled_rows):
        validation_count = len(shuffled_rows) - 1

    validation_rows = shuffled_rows[:validation_count]
    train_rows = shuffled_rows[validation_count:]
    log.info(f"{len(train_rows)=}, {len(validation_rows)=}, {validation_ratio=}")
    return _DatasetSplit(train_rows=train_rows, validation_rows=validation_rows)


def _pick_question(seed_questions: list[str], rng: random.Random) -> str:
    """Picks one random non-empty seed question."""
    candidates = [question.strip() for question in seed_questions if question.strip()]
    if not candidates:
        return ""
    return rng.choice(candidates)


def _build_negative_lookup(rows: list[SummaryRetrieverDataModel]) -> _NegativeTextLookup:
    """Builds lookup maps from custom_id to summary/description text."""
    summary_by_custom_id: dict[str, str] = {}
    description_by_custom_id: dict[str, str] = {}
    for row in rows:
        summary_by_custom_id[row.custom_id] = row.summary.strip()
        description_by_custom_id[row.custom_id] = row.description.strip()
    return _NegativeTextLookup(
        summary_by_custom_id=summary_by_custom_id,
        description_by_custom_id=description_by_custom_id,
    )


def _parse_hybrid_search_ids(payload_json: str) -> list[str]:
    """Parses ordered document ids from ``hybrid_search`` output."""
    rows = json.loads(payload_json)
    if not isinstance(rows, list):
        return []
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        if isinstance(row_id, str) and row_id:
            ids.append(row_id)
    return ids


def _slice_hard_negative_window(
    ordered_ids: list[str],
    positive_custom_id: str,
    config: _NegativeWindowConfig,
) -> list[str]:
    """Returns candidate ids from the configured hard-negative rank window."""
    filtered_ids = [candidate_id for candidate_id in ordered_ids if candidate_id != positive_custom_id]

    if config.window_start_rank < 1 or config.window_end_rank < config.window_start_rank:
        raise ValueError("Invalid hard-negative rank window.")

    start_index = config.window_start_rank - 1
    end_index = config.window_end_rank
    return filtered_ids[start_index:end_index]


def _hybrid_ranked_ids(
    question: str,
    field: str,
    config: _HybridSearchConfig,
) -> list[str]:
    """Runs hybrid retrieval and returns ranked ids."""
    payload_json = hybrid_search(
        query=question,
        field=field,
        root_dir=config.chroma_root_dir,
        embedding_base_url=config.embedding_base_url,
        embedding_model=config.embedding_model,
        api_key=config.api_key,
        limit=config.top_k,
        rrf_k=config.rrf_k,
    )
    return _parse_hybrid_search_ids(payload_json)


def _mine_field_negatives(
    question: str,
    positive_custom_id: str,
    lookup_by_custom_id: dict[str, str],
    field: str,
    hybrid_config: _HybridSearchConfig,
    window_config: _NegativeWindowConfig,
    rng: random.Random,
) -> list[str]:
    """Mines negatives for one field using hybrid retrieval + rank window."""
    ranked_ids = _hybrid_ranked_ids(question=question, field=field, config=hybrid_config)
    hard_window_ids = _slice_hard_negative_window(
        ordered_ids=ranked_ids,
        positive_custom_id=positive_custom_id,
        config=window_config,
    )

    negatives: list[str] = []
    for candidate_id in hard_window_ids:
        negative_text = lookup_by_custom_id.get(candidate_id, "").strip()
        if negative_text:
            negatives.append(negative_text)

    if len(negatives) <= window_config.negatives_per_row:
        return negatives

    sampled_negatives = rng.sample(negatives, window_config.negatives_per_row)
    log.info(f"{field=}, {len(negatives)=}, {len(sampled_negatives)=}")
    return sampled_negatives


def _build_training_dataset(
    rows: list[SummaryRetrieverDataModel],
    all_rows: list[SummaryRetrieverDataModel],
    hybrid_config: _HybridSearchConfig,
    window_config: _NegativeWindowConfig,
    seed: int,
) -> list[TrainingData]:
    """Builds ``TrainingData`` rows for one split."""
    lookup = _build_negative_lookup(all_rows)
    rng = random.Random(seed)

    output_rows: list[TrainingData] = []
    for row in rows:
        question = _pick_question(row.seed_questions, rng)
        if not question:
            log.warning(f"Skipping row with empty question for {row.custom_id=}")
            continue

        negative_summaries = _mine_field_negatives(
            question=question,
            positive_custom_id=row.custom_id,
            lookup_by_custom_id=lookup.summary_by_custom_id,
            field="summary",
            hybrid_config=hybrid_config,
            window_config=window_config,
            rng=rng,
        )
        negative_descriptions = _mine_field_negatives(
            question=question,
            positive_custom_id=row.custom_id,
            lookup_by_custom_id=lookup.description_by_custom_id,
            field="description",
            hybrid_config=hybrid_config,
            window_config=window_config,
            rng=rng,
        )

        output_rows.append(
            TrainingData(
                question=question,
                name=row.name,
                summary=row.summary,
                description=row.description,
                in_batch_negatives_descriptions=negative_descriptions,
                in_batch_negatives_summary=negative_summaries,
            )
        )

    log.info(f"{len(output_rows)=}")
    return output_rows


def build_training_and_validation_datasets(
    input_jsonl_path: str,
    output_train_jsonl_path: str = "artifacts/retriever_training/train.jsonl",
    output_validation_jsonl_path: str = "artifacts/retriever_training/validation.jsonl",
    validation_ratio: float = 0.1,
    random_seed: int = 13,
    chroma_root_dir: str = "artifacts/chroma",
    embedding_base_url: str = "http://127.0.0.1:8000/v1",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    api_key: str = "EMPTY",
    top_k: int = 100,
    rrf_k: int = 60,
    window_start_rank: int = 5,
    window_end_rank: int = 36,
    negatives_per_row: int = 32,
) -> None:
    """Builds train/validation datasets with hybrid-mined hard negatives.

    Args:
      input_jsonl_path: Input JSONL with ``SummaryRetrieverDataModel`` rows.
      output_train_jsonl_path: Output path for train dataset.
      output_validation_jsonl_path: Output path for validation dataset.
      validation_ratio: Fraction of rows allocated to validation split.
      random_seed: Global seed for split/question/negative sampling.
      chroma_root_dir: Root directory containing Chroma and BM25 artifacts.
      embedding_base_url: OpenAI-compatible embedding endpoint.
      embedding_model: Embedding model used for dense query encoding.
      api_key: API key for embedding endpoint.
      top_k: Hybrid retrieval top-K before hard-negative windowing.
      rrf_k: Reciprocal-rank-fusion constant.
      window_start_rank: First rank (1-indexed) of hard-negative window.
      window_end_rank: Last rank (1-indexed, inclusive) of hard-negative window.
      negatives_per_row: Number of sampled negatives per field.
    """
    input_path = Path(input_jsonl_path)
    train_path = Path(output_train_jsonl_path)
    validation_path = Path(output_validation_jsonl_path)

    raw_rows = _read_jsonl(input_path)
    all_rows = _rows_to_summary_models(raw_rows)
    split = _split_rows(all_rows, validation_ratio=validation_ratio, seed=random_seed)

    hybrid_config = _HybridSearchConfig(
        chroma_root_dir=chroma_root_dir,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        api_key=api_key,
        top_k=top_k,
        rrf_k=rrf_k,
    )
    window_config = _NegativeWindowConfig(
        window_start_rank=window_start_rank,
        window_end_rank=window_end_rank,
        negatives_per_row=negatives_per_row,
    )

    train_rows = _build_training_dataset(
        rows=split.train_rows,
        all_rows=all_rows,
        hybrid_config=hybrid_config,
        window_config=window_config,
        seed=random_seed,
    )
    validation_rows = _build_training_dataset(
        rows=split.validation_rows,
        all_rows=all_rows,
        hybrid_config=hybrid_config,
        window_config=window_config,
        seed=random_seed + 1,
    )

    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)


def main() -> None:
    """CLI entrypoint."""
    fire.Fire({"build": build_training_and_validation_datasets})


if __name__ == "__main__":
    main()
