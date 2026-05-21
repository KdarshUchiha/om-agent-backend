"""ReviewerAgent — reviews code, fixes bugs, and emits the final file list."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncGenerator

from .base import BaseAgent

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    name = "Reviewer"
    emoji = "✅"
    system_prompt = (
        "You are a senior code reviewer and QA engineer. Review the provided code "
        "thoroughly and produce the final, polished deliverable.\n\n"
        "Your responsibilities:\n"
        "1. Fix ALL bugs — syntax errors, logic errors, off-by-one errors, missing event "
        "   listeners, uninitialized variables, broken game loops, etc.\n"
        "2. Ensure the code is COMPLETE — no missing sections, no placeholders.\n"
        "3. Integrate the designer's CSS into the final file.\n"
        "4. For games: confirm game loop, collision detection, score, and restart all work.\n"
        "5. For web apps: ensure all buttons, forms, and interactions work correctly.\n\n"
        "OUTPUT FORMAT — output exactly this structure, nothing else:\n\n"
        "SUMMARY: <one or two sentences describing what was built and what was fixed>\n\n"
        "FILE: index.html\n"
        "```html\n"
        "<!DOCTYPE html>\n"
        "...complete file content here, nothing truncated...\n"
        "```\n\n"
        "Rules:\n"
        "- Output the COMPLETE file. Never truncate. Never write '...' or 'rest of code here'.\n"
        "- If there are multiple files (e.g. separate CSS or JS), add more FILE: blocks.\n"
        "- The code must run correctly when opened in a browser with zero modifications.\n"
        "- Do NOT add any explanation before SUMMARY: or after the last ``` block."
    )

    async def run(
        self,
        context: dict[str, Any],
        api_key: str,
        provider: str,
    ) -> AsyncGenerator[dict, None]:
        """
        Override run to also parse the final output and emit a `final_output` event.
        """
        full_text = ""
        async for event in self._stream_events(
            self._build_prompt(context), api_key, provider
        ):
            # Accumulate full text from done event
            if event.get("type") == "agent_done":
                full_text = event.pop("_full_text", "")
            yield event

        # Parse the reviewer's JSON output into a structured final_output event
        files, summary = self._parse_output(full_text, context)
        yield {
            "type": "final_output",
            "files": [{"name": f["name"], "content": f["content"]} for f in files],
            "summary": summary,
        }

    def _build_prompt(self, context: dict[str, Any]) -> str:
        user_prompt = context.get("user_prompt", "")
        coder_output = context.get("coder_output", "")
        architect_plan = context.get("architect_output", "")

        return (
            f"Original user request: {user_prompt}\n\n"
            f"## Architect's Plan (for reference)\n{architect_plan}\n\n"
            f"## Coder's Output (to review and finalize)\n{coder_output}\n\n"
            "Please review, fix any issues, and output the final JSON as specified."
        )

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(
        self, raw: str, context: dict[str, Any]
    ) -> tuple[list[dict], str]:
        """
        Parse the reviewer's structured output. Tries multiple patterns
        in order of reliability, always returns something.
        """
        # Extract summary from SUMMARY: line
        summary = "Project completed successfully."
        summary_match = re.search(r"SUMMARY:\s*(.+?)(?:\n|$)", raw)
        if summary_match:
            summary = summary_match.group(1).strip()

        # --- Strategy 1: FILE: <name> followed by fenced block (flexible whitespace) ---
        files = self._parse_file_blocks(raw)
        if files:
            logger.info("Parsed %d file(s) via FILE: pattern", len(files))
            return files, summary

        # --- Strategy 2: any fenced code block in reviewer output ---
        logger.warning("FILE: pattern miss — trying bare fenced blocks in reviewer output")
        files = self._extract_fenced_blocks(raw)
        if files:
            logger.info("Parsed %d file(s) via bare fenced blocks", len(files))
            return files, summary

        # --- Strategy 3: fenced blocks from coder output ---
        logger.warning("No fenced blocks in reviewer — falling back to coder output")
        coder_output = context.get("coder_output", "")
        files = self._extract_fenced_blocks(coder_output)
        if files:
            logger.info("Parsed %d file(s) from coder output", len(files))
            return files, summary

        # --- Strategy 4: strip markdown fences from coder output and wrap ---
        logger.warning("Last resort — stripping fences from coder output")
        content = re.sub(r"^```\w*\n?", "", coder_output.strip(), flags=re.MULTILINE)
        content = re.sub(r"\n?```$", "", content, flags=re.MULTILINE)
        if not content.strip():
            content = raw.strip()
        return [{"name": "index.html", "content": content}], summary

    @staticmethod
    def _parse_file_blocks(text: str) -> list[dict]:
        """
        Match FILE: <name> ... ```<lang>\n<content>\n``` with flexible spacing.
        Handles cases where the LLM adds extra blank lines or varies whitespace.
        """
        files: list[dict] = []
        # Split on FILE: markers first
        # e.g.  FILE: index.html\n```html\n...\n```
        parts = re.split(r"FILE:\s*([\w.\-/]+)", text)
        # parts = [preamble, name1, block1, name2, block2, ...]
        i = 1
        while i + 1 < len(parts):
            name = parts[i].strip()
            block = parts[i + 1]
            # Extract content from the first fenced block in this segment
            fence_match = re.search(r"```[^\n]*\n([\s\S]*?)```", block)
            if fence_match:
                content = fence_match.group(1)
                if content.strip():
                    files.append({"name": name, "content": content})
            i += 2
        return files

    @staticmethod
    def _extract_fenced_blocks(text: str) -> list[dict]:
        """Extract fenced code blocks, guessing filenames from language."""
        ext_map = {
            "html": "index.html",
            "css": "style.css",
            "javascript": "script.js",
            "js": "script.js",
            "python": "main.py",
            "py": "main.py",
        }
        files: list[dict] = []
        seen: dict[str, int] = {}
        pattern = re.compile(r"```(\w+)?\n([\s\S]*?)```", re.MULTILINE)
        for match in pattern.finditer(text):
            lang = (match.group(1) or "html").lower()
            content = match.group(2)
            if not content.strip():
                continue
            base_name = ext_map.get(lang, f"file.{lang}")
            count = seen.get(base_name, 0)
            name = base_name if count == 0 else f"{base_name.rsplit('.', 1)[0]}_{count}.{base_name.rsplit('.', 1)[-1]}"
            seen[base_name] = count + 1
            files.append({"name": name, "content": content})
        return files
