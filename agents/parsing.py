"""Shared parsing helpers for turning LLM output into a structured file list.

Extracted from ReviewerAgent so the Reviewer, Verifier, and RepairAgent all
parse files identically. The parser is deliberately forgiving: LLMs vary their
whitespace, fence labels, and FILE: markers, so we try several strategies in
order of reliability and always return *something*.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Map a fenced-block language to a default filename when the model omits a
# FILE: marker.
_EXT_MAP = {
    "html": "index.html",
    "css": "style.css",
    "javascript": "script.js",
    "js": "script.js",
    "python": "main.py",
    "py": "main.py",
}


def parse_summary(raw: str, default: str = "Project completed successfully.") -> str:
    """Extract the text following a ``SUMMARY:`` marker, if present."""
    match = re.search(r"SUMMARY:\s*(.+?)(?:\n|$)", raw)
    if match:
        return match.group(1).strip()
    return default


def parse_files(raw: str, fallback_text: str = "") -> list[dict]:
    """Parse LLM output into ``[{name, content}, ...]``.

    Strategies, tried in order:
      1. ``FILE: <name>`` followed by a fenced block.
      2. Any fenced code block in ``raw`` (filename guessed from language).
      3. Fenced blocks in ``fallback_text`` (e.g. an earlier agent's output).
      4. Last resort: strip fences from ``fallback_text`` and wrap as index.html.
    """
    files = _parse_file_blocks(raw)
    if files:
        logger.info("Parsed %d file(s) via FILE: pattern", len(files))
        return files

    logger.warning("FILE: pattern miss — trying bare fenced blocks")
    files = _extract_fenced_blocks(raw)
    if files:
        logger.info("Parsed %d file(s) via bare fenced blocks", len(files))
        return files

    if fallback_text:
        logger.warning("No fenced blocks in raw — falling back to provided text")
        files = _extract_fenced_blocks(fallback_text)
        if files:
            logger.info("Parsed %d file(s) from fallback text", len(files))
            return files

        logger.warning("Last resort — stripping fences from fallback text")
        content = re.sub(r"^```\w*\n?", "", fallback_text.strip(), flags=re.MULTILINE)
        content = re.sub(r"\n?```$", "", content, flags=re.MULTILINE)
        if content.strip():
            return [{"name": "index.html", "content": content}]

    # Nothing usable anywhere — wrap the raw text so the caller still gets a file.
    if raw.strip():
        return [{"name": "index.html", "content": raw.strip()}]
    return []


def parse_output(raw: str, fallback_text: str = "") -> tuple[list[dict], str]:
    """Convenience wrapper returning ``(files, summary)``."""
    return parse_files(raw, fallback_text), parse_summary(raw)


def _parse_file_blocks(text: str) -> list[dict]:
    """Match ``FILE: <name>`` ... ```<lang>\\n<content>\\n``` with flexible spacing."""
    files: list[dict] = []
    parts = re.split(r"FILE:\s*([\w.\-/]+)", text)
    # parts = [preamble, name1, block1, name2, block2, ...]
    i = 1
    while i + 1 < len(parts):
        name = parts[i].strip()
        block = parts[i + 1]
        fence_match = re.search(r"```[^\n]*\n([\s\S]*?)```", block)
        if fence_match:
            content = fence_match.group(1)
            if content.strip():
                files.append({"name": name, "content": content})
        i += 2
    return files


def _extract_fenced_blocks(text: str) -> list[dict]:
    """Extract fenced code blocks, guessing filenames from the language tag."""
    files: list[dict] = []
    seen: dict[str, int] = {}
    pattern = re.compile(r"```(\w+)?\n([\s\S]*?)```", re.MULTILINE)
    for match in pattern.finditer(text):
        lang = (match.group(1) or "html").lower()
        content = match.group(2)
        if not content.strip():
            continue
        base_name = _EXT_MAP.get(lang, f"file.{lang}")
        count = seen.get(base_name, 0)
        if count == 0:
            name = base_name
        else:
            stem, _, ext = base_name.rpartition(".")
            name = f"{stem}_{count}.{ext}"
        seen[base_name] = count + 1
        files.append({"name": name, "content": content})
    return files
