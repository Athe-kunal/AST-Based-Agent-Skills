from __future__ import annotations

import json
import time
from pathlib import Path

from openai import OpenAI


INPUT_DIR = Path("data")
OUTPUT_DIR = Path("batch_results")
POLL_INTERVAL_SECONDS = 30

# Choose the endpoint that matches the requests inside each JSONL file.
# For modern usage, /v1/responses is usually the best default.
BATCH_ENDPOINT = "/v1/responses"

client = OpenAI()


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_jsonl_lines(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}: {exc}"
                ) from exc
    return records


def validate_batch_jsonl(path: Path) -> None:
    """
    Light validation so you fail early before uploading.
    Expected shape per line:
      {
        "custom_id": "...",
        "method": "POST",
        "url": "/v1/responses",
        "body": { ... }
      }
    """
    records = load_jsonl_lines(path)
    if not records:
        raise ValueError(f"{path} is empty.")

    for idx, record in enumerate(records, start=1):
        if "custom_id" not in record:
            raise ValueError(f"{path} line {idx}: missing 'custom_id'")
        if record.get("method") != "POST":
            raise ValueError(f"{path} line {idx}: expected method='POST'")
        if "url" not in record:
            raise ValueError(f"{path} line {idx}: missing 'url'")
        if "body" not in record or not isinstance(record["body"], dict):
            raise ValueError(f"{path} line {idx}: missing or invalid 'body'")


def upload_batch_file(path: Path) -> str:
    with path.open("rb") as f:
        uploaded = client.files.create(
            file=f,
            purpose="batch",
        )
    return uploaded.id


def create_batch(input_file_id: str, source_file: Path) -> str:
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint=BATCH_ENDPOINT,
        completion_window="24h",
        metadata={
            "source_filename": source_file.name,
            "pipeline": "jsonl-sequential-runner",
        },
    )
    return batch.id


def wait_for_batch(batch_id: str) -> object:
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"batch_id={batch_id} status={status}")

        if status in {"completed", "failed", "expired", "cancelled"}:
            return batch

        time.sleep(POLL_INTERVAL_SECONDS)


def download_file_content(file_id: str) -> str:
    content = client.files.content(file_id)
    # SDKs may return either raw text-like content or an object with `.text`
    if hasattr(content, "text"):
        return content.text
    return str(content)


def process_one_jsonl(path: Path) -> None:
    print(f"Processing {path}")
    validate_batch_jsonl(path)

    input_file_id = upload_batch_file(path)
    print(f"Uploaded {path.name} -> input_file_id={input_file_id}")

    batch_id = create_batch(input_file_id=input_file_id, source_file=path)
    print(f"Created batch {batch_id} for {path.name}")

    batch = wait_for_batch(batch_id=batch_id)

    base_name = path.stem
    batch_dir = OUTPUT_DIR / base_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save final batch metadata
    batch_summary = {
        "id": batch.id,
        "status": batch.status,
        "input_file_id": batch.input_file_id,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": getattr(batch, "request_counts", None),
        "metadata": getattr(batch, "metadata", None),
    }
    save_text(
        batch_dir / "batch_summary.json",
        json.dumps(batch_summary, indent=2, default=str),
    )

    output_file_id = getattr(batch, "output_file_id", None)
    if output_file_id:
        output_text = download_file_content(output_file_id)
        save_text(batch_dir / "output.jsonl", output_text)
        print(f"Saved output for {path.name}")

    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        error_text = download_file_content(error_file_id)
        save_text(batch_dir / "errors.jsonl", error_text)
        print(f"Saved errors for {path.name}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(INPUT_DIR.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {INPUT_DIR}")

    for path in jsonl_files:
        try:
            process_one_jsonl(path)
        except Exception as exc:
            print(f"Failed for {path.name}: {exc}")


if __name__ == "__main__":
    main()
