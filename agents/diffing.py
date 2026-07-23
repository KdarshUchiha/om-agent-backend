"""Diff-based editing — apply surgical search/replace edits to existing files.

Phase 2 of the Om agent roadmap: instead of asking the model to re-emit an
entire file for every refinement (slow, and prone to regressions where the
model "forgets" a working section), we ask it for *targeted* edits and apply
them deterministically here.

We use **search/replace blocks** rather than raw unified diffs. Unified diffs
require exact line numbers and surrounding context, which LLMs get wrong far
more often than they get a verbatim snippet of the code they just read. A
search/replace block only needs the model to quote the exact old text and the
new text:

    FILE: index.html
    <<<<<<< SEARCH
    const speed = 5;
    =======
    const speed = 8;
    >>>>>>> REPLACE

Applying is a plain string replacement, so it is fully deterministic and
testable without an LLM. Each block either applies cleanly or is reported as a
failure — the caller decides whether to fall back to a full rewrite.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# A search/replace block:
#   FILE: <name>
#   <<<<<<< SEARCH
#   <old text>
#   =======
#   <new text>
#   >>>>>>> REPLACE
# The FILE: marker is optional for the *first* block when there is only one
# file in play; subsequent blocks should name their file.
_BLOCK_RE = re.compile(
    r"(?:FILE:\s*(?P<file>[\w.\-/]+)\s*\n)?"
    r"<{5,}\s*SEARCH\s*\n"
    r"(?P<search>[\s\S]*?)\n"
    r"={5,}\s*\n"
    r"(?P<replace>[\s\S]*?)\n"
    r">{5,}\s*REPLACE",
    re.MULTILINE,
)


@dataclass
class EditBlock:
    """A single parsed search/replace edit."""

    file: str | None
    search: str
    replace: str


@dataclass
class EditResult:
    """Outcome of applying a set of edit blocks to a file list."""

    files: list[dict]
    applied: int = 0
    failed: list[str] = field(default_factory=list)

    @property
    def all_applied(self) -> bool:
        return not self.failed and self.applied > 0


def parse_edit_blocks(raw: str) -> list[EditBlock]:
    """Parse every search/replace block from raw LLM output.

    Returns an empty list when the output contains no blocks (the caller should
    then treat the output as a full-file rewrite and route it to ``parse_files``).
    """
    blocks: list[EditBlock] = []
    for m in _BLOCK_RE.finditer(raw):
        blocks.append(
            EditBlock(
                file=(m.group("file") or "").strip() or None,
                search=m.group("search"),
                replace=m.group("replace"),
            )
        )
    return blocks


def _resolve_file(block: EditBlock, files: list[dict], last_file: str | None) -> str | None:
    """Pick which file a block targets.

    Priority: the block's explicit FILE: name → the previously edited file →
    the sole file when there is exactly one. Returns None if it can't decide.
    """
    if block.file:
        return block.file
    if last_file:
        return last_file
    if len(files) == 1:
        return files[0]["name"]
    return None


def apply_edits(files: list[dict], blocks: list[EditBlock]) -> EditResult:
    """Apply search/replace ``blocks`` to ``files`` (a list of ``{name, content}``).

    Files are copied — the input list is never mutated. Each block that cannot
    be located (unknown file, or search text not found) is recorded in
    ``failed`` and skipped; the rest still apply. A block whose SEARCH text
    appears more than once is treated as ambiguous and reported as failed,
    because a blind replace-first could corrupt the file.
    """
    working = {f["name"]: f.get("content", "") for f in files}
    order = [f["name"] for f in files]

    result = EditResult(files=files)
    applied = 0
    last_file: str | None = None

    for idx, block in enumerate(blocks):
        target = _resolve_file(block, files, last_file)
        if target is None:
            result.failed.append(
                f"block #{idx + 1}: could not determine target file "
                "(no FILE: marker and multiple files present)"
            )
            continue

        if target not in working:
            result.failed.append(f"block #{idx + 1}: file '{target}' not found")
            continue

        content = working[target]
        occurrences = content.count(block.search)

        if occurrences == 0:
            result.failed.append(
                f"block #{idx + 1} ({target}): SEARCH text not found"
            )
            continue
        if occurrences > 1:
            result.failed.append(
                f"block #{idx + 1} ({target}): SEARCH text is ambiguous "
                f"(matches {occurrences} times) — add more surrounding context"
            )
            continue

        working[target] = content.replace(block.search, block.replace, 1)
        last_file = target
        applied += 1

    result.applied = applied
    result.files = [{"name": name, "content": working[name]} for name in order]
    return result
