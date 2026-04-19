from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import pandas as pd

from ast_skills.retriever.datamodels import ValidatedSkillQuestionRow
from ast_skills.retriever.maximal_marginal_relevance_question import (
    MmrSelectionConfig,
    select_diverse_questions_for_rows,
)


def _load_rows(path: str = "artifacts/validated_training_data_list.jsonl") -> list[ValidatedSkillQuestionRow]:
    rows: list[ValidatedSkillQuestionRow] = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append(ValidatedSkillQuestionRow(**d))
    return rows


def _rows_to_dataframe(rows: list[ValidatedSkillQuestionRow]) -> pd.DataFrame:
    return pd.DataFrame([dataclasses.asdict(row) for row in rows])


async def build_train_val_parquets(
    config: MmrSelectionConfig,
    input_path: str = "artifacts/validated_training_data_list.jsonl",
    output_dir: str = "artifacts",
) -> None:
    rows = _load_rows(input_path)
    mmr_config = config._replace(selected_question_count=3)
    diverse_by_id = await select_diverse_questions_for_rows(rows=rows, config=mmr_config)

    train_rows: list[ValidatedSkillQuestionRow] = []
    val_rows: list[ValidatedSkillQuestionRow] = []

    for row in rows:
        mmr_questions = diverse_by_id.get(row.custom_id, [])
        if len(mmr_questions) < 3:
            continue
        row_with_mmr = dataclasses.replace(row, mmr_questions=mmr_questions)
        train_rows.append(dataclasses.replace(row_with_mmr, filtered_questions=mmr_questions[:2]))
        val_rows.append(dataclasses.replace(row_with_mmr, filtered_questions=mmr_questions[2:3]))

    output = Path(output_dir)
    _rows_to_dataframe(train_rows).to_parquet(output / "train.parquet", index=False)
    _rows_to_dataframe(val_rows).to_parquet(output / "val.parquet", index=False)


def _main(
    base_url: str = "http://127.0.0.1:8001/v1",
    api_key: str = "EMPTY",
    embedding_model: str = "Qwen/Qwen3-Embedding-8B",
    mmr_lambda: float = 0.5,
    batch_size: int = 64,
    max_concurrency: int = 32,
    input_path: str = "artifacts/validated_training_data_list.jsonl",
    output_dir: str = "artifacts",
) -> None:
    config = MmrSelectionConfig(
        base_url=base_url,
        api_key=api_key,
        embedding_model=embedding_model,
        mmr_lambda=mmr_lambda,
        selected_question_count=3,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    )
    asyncio.run(build_train_val_parquets(config=config, input_path=input_path, output_dir=output_dir))


if __name__ == "__main__":
    import fire

    fire.Fire(_main)
