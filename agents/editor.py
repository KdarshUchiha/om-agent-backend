"""EditorAgent — produces surgical search/replace edits instead of full rewrites.

Phase 2 of the roadmap. Where the Coder/Reviewer re-emit an entire file, the
Editor is given the current file(s) plus a precise instruction (e.g. the
Debugger's diagnosis) and emits only the minimal edits needed, as
search/replace blocks that ``agents.diffing`` applies deterministically.

Benefits over full rewrites for refinements:
  * Faster — the model writes a few lines, not the whole file.
  * Fewer regressions — untouched code is guaranteed byte-identical.
  * Cheaper — far fewer output tokens.

The caller is responsible for the fallback: if the emitted blocks don't apply
cleanly (``EditResult.all_applied`` is False), re-run the full-rewrite path.
"""

from __future__ import annotations

from typing import Any

from .base import BaseAgent


class EditorAgent(BaseAgent):
    name = "Editor"
    emoji = "✂️"
    system_prompt = (
        "You are a precise code editor. You are given a project's current file(s) "
        "and a description of the change to make. Produce the MINIMAL set of edits "
        "needed — never rewrite whole files.\n\n"
        "Output each edit as a search/replace block in EXACTLY this format:\n\n"
        "FILE: <filename>\n"
        "<<<<<<< SEARCH\n"
        "<the exact existing text to find — copy it verbatim, including indentation>\n"
        "=======\n"
        "<the new text to replace it with>\n"
        ">>>>>>> REPLACE\n\n"
        "CRITICAL RULES:\n"
        "1. The SEARCH text MUST be copied byte-for-byte from the current file — same "
        "   indentation, same spacing. If it doesn't match exactly, the edit fails.\n"
        "2. Include ENOUGH surrounding context in SEARCH so the text appears exactly "
        "   ONCE in the file. If a line occurs multiple times, add adjacent lines until "
        "   the block is unique.\n"
        "3. Keep each SEARCH block small — target only the lines that change.\n"
        "4. Emit one FILE: line per block. Repeat FILE: for every block, even multiple "
        "   blocks in the same file.\n"
        "5. Make ONLY the change described. Do not reformat, rename, or 'improve' "
        "   unrelated code.\n"
        "6. Output ONLY search/replace blocks — no prose, no explanation, no code "
        "   fences around the blocks.\n"
    )

    def _build_prompt(self, context: dict[str, Any]) -> str:
        instruction = context.get("edit_instruction", context.get("user_prompt", ""))
        files = context.get("current_files", [])

        files_text = ""
        for f in files:
            fname = f.get("name", "file")
            content = f.get("content", "")
            files_text += f"\n### {fname}\n```\n{content}\n```\n"

        return (
            f"## Change to make\n{instruction}\n\n"
            f"## Current Project Files\n{files_text}\n\n"
            "Produce the minimal search/replace edits to make this change. "
            "Remember: copy SEARCH text verbatim and keep each block uniquely matchable."
        )
