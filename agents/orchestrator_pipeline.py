"""
Pipeline orchestration — runs all six agents in the correct order,
wires their outputs into a shared context dict, and yields SSE event dicts
for the FastAPI endpoint to forward to the browser.

Flow:
  Orchestrator
      ↓
  Architect ──── (parallel) ──── Designer ──── (parallel) ──── Asset Artist
      └─────────────────────────────┬──────────────────────────────┘
                                    ↓
                                  Coder
                                    ↓
                                 Reviewer
                                    ↓
                            final_output / done
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator

from .architect import ArchitectAgent
from .asset_artist import AssetArtistAgent
from .coder import CoderAgent
from .designer import DesignerAgent
from .orchestrator import OrchestratorAgent
from .parsing import parse_files
from .repair import RepairAgent
from .reviewer import ReviewerAgent
from .verifier import verify_files

logger = logging.getLogger(__name__)

# Max verify→repair iterations before we give up and ship the best effort.
MAX_REPAIR_ITERATIONS = 2


async def run_pipeline(
    user_prompt: str,
    api_key: str,
    provider: str,
    mode: str | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Execute the full Om Agent pipeline and yield SSE event dicts.

    Each agent's full-text output is captured from the `agent_done` event
    (where `_full_text` is stashed by BaseAgent) and stored in a shared
    `context` dict so later agents can read it.
    """

    context: dict[str, Any] = {
        "user_prompt": user_prompt,
        "mode": mode or "",
    }

    orchestrator = OrchestratorAgent()
    architect = ArchitectAgent()
    designer = DesignerAgent()
    asset_artist = AssetArtistAgent()
    coder = CoderAgent()
    reviewer = ReviewerAgent()

    # ------------------------------------------------------------------
    # 1. Orchestrator
    # ------------------------------------------------------------------
    yield {
        "type": "agent_thinking",
        "agent": orchestrator.name,
        "emoji": orchestrator.emoji,
        "message": "Analyzing your request and creating a task brief…",
    }

    orchestrator_text = ""
    async for event in orchestrator.run(context, api_key, provider):
        if event.get("type") == "agent_done":
            orchestrator_text = event.pop("_full_text", "")
        yield event

    context["orchestrator_output"] = orchestrator_text

    # ------------------------------------------------------------------
    # 2. Architect + Designer + Asset Artist in parallel
    # ------------------------------------------------------------------
    yield {
        "type": "agent_thinking",
        "agent": "Architect & Designer & Asset Artist",
        "emoji": "⚡",
        "message": "Architect, Designer, and Asset Artist working in parallel…",
    }

    architect_text = ""
    designer_text = ""
    asset_artist_text = ""

    architect_events: list[dict] = []
    designer_events: list[dict] = []
    asset_artist_events: list[dict] = []

    async def _collect(agent_gen: AsyncGenerator[dict, None], out: list[dict]) -> None:
        async for ev in agent_gen:
            out.append(ev)

    await asyncio.gather(
        _collect(architect.run(context, api_key, provider), architect_events),
        _collect(designer.run(context, api_key, provider), designer_events),
        _collect(asset_artist.run(context, api_key, provider), asset_artist_events),
    )

    # Emit events in order: architect → designer → asset artist
    for event in architect_events:
        if event.get("type") == "agent_done":
            architect_text = event.pop("_full_text", "")
        yield event

    for event in designer_events:
        if event.get("type") == "agent_done":
            designer_text = event.pop("_full_text", "")
        yield event

    for event in asset_artist_events:
        if event.get("type") == "agent_done":
            asset_artist_text = event.pop("_full_text", "")
        yield event

    context["architect_output"] = architect_text
    context["designer_output"] = designer_text
    context["asset_artist_output"] = asset_artist_text

    # ------------------------------------------------------------------
    # 3. Coder
    # ------------------------------------------------------------------
    yield {
        "type": "agent_thinking",
        "agent": coder.name,
        "emoji": coder.emoji,
        "message": "Writing complete code with assets, design, and architecture…",
    }

    coder_text = ""
    async for event in coder.run(context, api_key, provider):
        if event.get("type") == "agent_done":
            coder_text = event.pop("_full_text", "")
        yield event

    context["coder_output"] = coder_text

    # ------------------------------------------------------------------
    # 4. Reviewer — produces the first candidate build. We intercept its
    #    final_output so the verify→repair loop can run before the client
    #    receives the definitive files.
    # ------------------------------------------------------------------
    yield {
        "type": "agent_thinking",
        "agent": reviewer.name,
        "emoji": reviewer.emoji,
        "message": "Reviewing, fixing bugs, and packaging the final output…",
    }

    candidate_files: list[dict] = []
    candidate_summary = "Project completed successfully."
    async for event in reviewer.run(context, api_key, provider):
        if event.get("type") == "final_output":
            # Hold this back — the loop below emits the authoritative one.
            candidate_files = event.get("files", [])
            candidate_summary = event.get("summary", candidate_summary)
            continue
        yield event

    # ------------------------------------------------------------------
    # 5. Verify → Repair loop (agentic self-correction)
    # ------------------------------------------------------------------
    async for event in _verify_repair_loop(
        candidate_files,
        candidate_summary,
        user_prompt,
        api_key,
        provider,
    ):
        yield event

    # ------------------------------------------------------------------
    # 6. Done
    # ------------------------------------------------------------------
    yield {"type": "done"}


async def _verify_repair_loop(
    files: list[dict],
    summary: str,
    user_prompt: str,
    api_key: str,
    provider: str,
) -> AsyncGenerator[dict, None]:
    """Run the deterministic verifier and, if it finds errors, dispatch the
    RepairAgent to fix them — repeating up to ``MAX_REPAIR_ITERATIONS`` times.

    Emits additive SSE events (``verify_start``, ``verify_result``,
    ``repair_start``) and finishes by yielding the authoritative
    ``final_output`` event.
    """
    repair = RepairAgent()

    for iteration in range(MAX_REPAIR_ITERATIONS + 1):
        yield {
            "type": "verify_start",
            "agent": "Verifier",
            "emoji": "🔎",
            "message": "Running automated checks on the generated code…",
        }

        report = verify_files(files)

        yield {
            "type": "verify_result",
            "agent": "Verifier",
            "emoji": "🔎",
            "passed": report.passed,
            "error_count": len(report.errors),
            "warning_count": len(report.warnings),
            "findings": [f.to_dict() for f in report.findings],
            "message": (
                "All checks passed ✅"
                if report.passed
                else f"Found {len(report.errors)} issue(s) to fix."
            ),
        }

        if report.passed or iteration == MAX_REPAIR_ITERATIONS:
            break

        # Dispatch the RepairAgent with the concrete defect list.
        yield {
            "type": "repair_start",
            "agent": repair.name,
            "emoji": repair.emoji,
            "message": (
                f"Fixing {len(report.errors)} issue(s) "
                f"(attempt {iteration + 1}/{MAX_REPAIR_ITERATIONS})…"
            ),
        }

        repair_context: dict[str, Any] = {
            "user_prompt": user_prompt,
            "current_files": files,
            "verification_report": report.as_prompt_text(),
        }

        repair_text = ""
        async for event in repair.run(repair_context, api_key, provider):
            if event.get("type") == "agent_done":
                repair_text = event.pop("_full_text", "")
            yield event

        # Parse the repaired files; if parsing fails, keep the prior candidate.
        repaired = parse_files(repair_text)
        if repaired:
            files = repaired

    yield {
        "type": "final_output",
        "files": [{"name": f["name"], "content": f["content"]} for f in files],
        "summary": summary,
    }


def _is_bug_report(prompt: str) -> bool:
    """Detect if the user is reporting a bug vs requesting a feature."""
    bug_keywords = [
        "broken", "doesn't work", "not working", "crash", "error", "bug",
        "fix", "issue", "wrong", "fails", "stuck", "freezes", "blank",
        "nothing happens", "can't", "won't", "undefined", "null",
        "NaN", "infinite loop", "doesn't load", "white screen",
        "console error", "black screen", "no response", "glitch",
    ]
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in bug_keywords)


async def run_refine_pipeline(
    refinement_prompt: str,
    current_files: list[dict],
    conversation: list[dict],
    api_key: str,
    provider: str,
) -> AsyncGenerator[dict, None]:
    """
    Smart refine pipeline:
    - If user reports a BUG: Debugger diagnoses → Coder fixes → Reviewer validates
    - If user requests a FEATURE: Full 6-agent pipeline with context
    """
    from .debugger import DebuggerAgent

    is_bug = _is_bug_report(refinement_prompt)

    if is_bug:
        # BUG FIX FLOW: Debugger → Coder → Reviewer
        yield {
            "type": "agent_thinking",
            "agent": "Pipeline",
            "emoji": "🔍",
            "message": "Bug detected — running diagnostic first…",
        }

        debugger = DebuggerAgent()
        coder = CoderAgent()
        reviewer = ReviewerAgent()

        # Step 1: Debugger analyzes the issue
        debug_context: dict[str, Any] = {
            "user_prompt": refinement_prompt,
            "current_files": current_files,
            "conversation": conversation,
        }

        debugger_text = ""
        async for event in debugger.run(debug_context, api_key, provider):
            if event.get("type") == "agent_done":
                debugger_text = event.pop("_full_text", "")
            yield event

        # Step 2: Coder applies the fix (with debugger's diagnosis as context)
        files_text = ""
        for f in current_files:
            name = f.get("name", "file")
            content = f.get("content", "")
            trimmed = content[:6000] + ("\n...[trimmed]..." if len(content) > 6000 else "")
            files_text += f"\n### {name}\n```\n{trimmed}\n```\n"

        coder_context: dict[str, Any] = {
            "user_prompt": refinement_prompt,
            "mode": "bugfix",
            "orchestrator_output": (
                f"This is a BUG FIX. The debugger has identified the issue.\n\n"
                f"## Debugger's Diagnosis\n{debugger_text}\n\n"
                f"## User's Bug Report\n{refinement_prompt}\n\n"
                f"Apply the fix described by the debugger. Output the complete corrected file(s)."
            ),
            "architect_output": f"## Current Project Files\n{files_text}",
            "designer_output": "",
            "asset_artist_output": "",
        }

        yield {
            "type": "agent_thinking",
            "agent": coder.name,
            "emoji": coder.emoji,
            "message": "Applying the fix…",
        }

        coder_text = ""
        async for event in coder.run(coder_context, api_key, provider):
            if event.get("type") == "agent_done":
                coder_text = event.pop("_full_text", "")
            yield event

        coder_context["coder_output"] = coder_text

        # Step 3: Reviewer validates the fix
        yield {
            "type": "agent_thinking",
            "agent": reviewer.name,
            "emoji": reviewer.emoji,
            "message": "Verifying the fix…",
        }

        async for event in reviewer.run(coder_context, api_key, provider):
            yield event

        yield {"type": "done"}

    else:
        # FEATURE REQUEST FLOW: Full 6-agent pipeline with context
        files_text = ""
        for f in current_files:
            name = f.get("name", "file")
            content = f.get("content", "")
            trimmed = content[:4000] + ("\n...[trimmed]..." if len(content) > 4000 else "")
            files_text += f"\n### {name}\n```\n{trimmed}\n```\n"

        conv_text = ""
        if conversation:
            for turn in conversation[-8:]:
                role = turn.get("role", "user")
                msg = turn.get("content", "")[:300]
                conv_text += f"\n{role}: {msg}\n"

        augmented_prompt = (
            f"{refinement_prompt}\n\n"
            f"---\n"
            f"CONTEXT: This is a refinement of an existing project. "
            f"The current project files are provided below. "
            f"You may EDIT specific parts or REWRITE entirely based on the user's request.\n\n"
            f"## Previous Conversation\n{conv_text}\n\n"
            f"## Current Project Files\n{files_text}"
        )

        async for event in run_pipeline(
            user_prompt=augmented_prompt,
            api_key=api_key,
            provider=provider,
            mode="refine",
        ):
            yield event
