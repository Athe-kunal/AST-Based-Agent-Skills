"""Async progress bar utilities for pipeline tasks."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Coroutine

from tqdm.asyncio import tqdm


def _write_progress(done: int, total: int, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"done": done, "total": total}), encoding="utf-8")


async def gather_with_progress(
    tasks: list[Coroutine[Any, Any, Any]],
    desc: str = "Processing",
    progress_file: str | None = None,
) -> list[Any]:
    """Run coroutines concurrently with a tqdm bar and optional JSON progress file.

    The progress file is written after each completed task:
      {"done": 42, "total": 200}
    """
    total = len(tasks)
    futures = [asyncio.ensure_future(t) for t in tasks]
    results: list[Any] = []

    with tqdm(total=total, desc=desc, unit="row") as pbar:
        for future in asyncio.as_completed(futures):
            result = await future
            results.append(result)
            pbar.update(1)
            if progress_file:
                _write_progress(len(results), total, progress_file)

    return results
