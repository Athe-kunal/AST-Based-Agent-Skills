"""OpenAI Batch API input line builder for POST /v1/chat/completions."""

from __future__ import annotations

from typing import Any

DEFAULT_OPENAI_BATCH_MODEL = "gpt-4o-mini"


def openai_batch_chat_completion_request(
    custom_id: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    One OpenAI Batch API input line for POST /v1/chat/completions.

    See https://developers.openai.com/api/docs/guides/batch
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        body["response_format"] = response_format

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
