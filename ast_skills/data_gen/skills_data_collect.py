from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import tiktoken

# Primary Han block (simplified + traditional in common use).
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")
# Optional: also match CJK Extension A/B etc. if you need stricter “any CJK ideograph”:
# _HAN_RE = re.compile(
#     r"[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf]"
# )

encoding = tiktoken.encoding_for_model("gpt-4o-mini")  # or similar

threshold = 63_000


@dataclass(frozen=True)
class SkillMdRecord:
    relative_path: str
    content: str


def contains_chinese(text: str) -> bool:
    return _HAN_RE.search(text) is not None


def collect_skill_md_without_chinese(root: str | Path) -> list[SkillMdRecord]:
    root_path = Path(root).resolve()
    records: list[SkillMdRecord] = []

    # Use tqdm to wrap the rglob generator for progress display
    paths = list(root_path.rglob("SKILL.md"))
    for path in tqdm(paths, desc="Scanning SKILL.md files"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        token_len = len(encoding.encode(text, disallowed_special=set()))
        if token_len > threshold or token_len == 0:
            continue
        if contains_chinese(text):
            continue
        rel = path.relative_to(root_path).as_posix()
        records.append(SkillMdRecord(relative_path=rel, content=text))

    records.sort(key=lambda r: r.relative_path)
    return records


if __name__ == "__main__":
    root = "skills/skills"
    records = collect_skill_md_without_chinese(root)
    print(f"{root=}")
    print(f"{len(records)=}")
    for r in records[:5]:
        print("---")
        print(r.relative_path)
        print(r.content[:200] + ("…" if len(r.content) > 200 else ""))
