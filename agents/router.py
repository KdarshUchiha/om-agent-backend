"""Model router — decides which backend serves each agent's request.

Phase 3 of the roadmap: "Cursor-quality orchestration wrapping frontier models."
Instead of each agent hardcoding its backend, the router maps an agent to an
ordered chain of backends to try. The fine-tuned **Om** models (Om-Think,
Om-Code) become the *style layer* for light/personality work; **frontier**
models (Claude, or the user's Gemini/Groq) are the *brains* for hard reasoning
and code.

The router is pure policy — no I/O. `base.BaseAgent` consults it to build the
backend chain, then executes with fallback. Each returned chain always ends at
the user's own provider (which is guaranteed to have a key), so a request can
never hard-fail because a preferred backend is down.

Configuration (env vars, read once):
  OM_ROUTER_MODE   frontier | hybrid | om   (default: hybrid)
    - frontier : brains AND style go to the frontier tier; Om is only a
                 last-resort fallback. Highest quality, no Om personality.
    - hybrid   : brains → frontier tier; style/chat → Om first (the "style
                 layer"), frontier as fallback. Implements the Phase 3 vision.
    - om       : Om first for every agent, frontier as fallback. Closest to the
                 pre-router behavior.
  ANTHROPIC_API_KEY  when set, Claude joins the frontier tier ahead of the
                     user's provider. When unset, the frontier tier is just the
                     user's provider — fully backward compatible.
"""

from __future__ import annotations

import os
from enum import Enum


class Backend(str, Enum):
    """A concrete model backend the pipeline can stream from."""

    CLAUDE = "claude"
    GEMINI = "gemini"
    GROQ = "groq"
    OM_THINK = "om-think"
    OM_CODE = "om-code"


class TaskType(str, Enum):
    """The kind of work an agent does — drives which tier and Om variant fit."""

    REASONING = "reasoning"  # planning, architecture, diagnosis — text brains
    CODE = "code"            # writing/reviewing/repairing/editing code
    STYLE = "style"          # visual/design/asset work — personality-leaning
    CHAT = "chat"            # conversational


# Which task type each agent performs. Agent names match the `name` attribute
# on each *Agent class. Unknown agents default to REASONING (frontier-first).
AGENT_TASK_TYPE: dict[str, TaskType] = {
    "Orchestrator": TaskType.REASONING,
    "Architect": TaskType.REASONING,
    "Debugger": TaskType.REASONING,
    "Coder": TaskType.CODE,
    "Reviewer": TaskType.CODE,
    "Repair": TaskType.CODE,
    "Editor": TaskType.CODE,
    "Designer": TaskType.STYLE,
    "Asset Artist": TaskType.STYLE,
    "WorkspaceChat": TaskType.CHAT,
}

# The Om "style layer" variant that best matches each task type.
_OM_FOR_TASK: dict[TaskType, Backend] = {
    TaskType.REASONING: Backend.OM_THINK,
    TaskType.CODE: Backend.OM_CODE,
    TaskType.STYLE: Backend.OM_CODE,   # CSS/SVG is code-shaped
    TaskType.CHAT: Backend.OM_THINK,
}

_DEFAULT_MODE = "hybrid"
_VALID_MODES = {"frontier", "hybrid", "om"}


def _mode() -> str:
    mode = os.environ.get("OM_ROUTER_MODE", _DEFAULT_MODE).strip().lower()
    return mode if mode in _VALID_MODES else _DEFAULT_MODE


def claude_available() -> bool:
    """True when an Anthropic key is configured, so Claude can serve requests."""
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def _provider_backend(provider: str) -> Backend:
    """Map the user's chosen provider string to its Backend."""
    return Backend.GROQ if provider == "groq" else Backend.GEMINI


def _frontier_tier(provider: str) -> list[Backend]:
    """Frontier backends, best first: Claude (if keyed) then the user's provider."""
    tier: list[Backend] = []
    if claude_available():
        tier.append(Backend.CLAUDE)
    tier.append(_provider_backend(provider))
    return tier


def resolve(agent_name: str, provider: str, mode: str | None = None) -> list[Backend]:
    """Return the ordered backend chain to try for ``agent_name``.

    ``provider`` is the user's chosen provider ("gemini" | "groq") — always the
    guaranteed-available floor of every chain. ``mode`` overrides the env
    default (mainly for tests). Duplicates are removed while preserving order.
    """
    mode = (mode or _mode())
    task = AGENT_TASK_TYPE.get(agent_name, TaskType.REASONING)
    om = _OM_FOR_TASK[task]
    frontier = _frontier_tier(provider)

    if mode == "om":
        chain = [om, *frontier]
    elif mode == "hybrid" and task in (TaskType.STYLE, TaskType.CHAT):
        # Style/chat lead with the Om "style layer", frontier as fallback.
        chain = [om, *frontier]
    else:
        # frontier mode (any task) or hybrid brains: frontier first, Om last.
        chain = [*frontier, om]

    # De-dupe while preserving order.
    seen: set[Backend] = set()
    ordered: list[Backend] = []
    for b in chain:
        if b not in seen:
            seen.add(b)
            ordered.append(b)
    return ordered
