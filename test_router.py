"""Tests for the model router (agents/router.py)."""

import os
from agents.router import resolve, Backend, claude_available


def _no_claude(monkeypatch=None):
    os.environ.pop("ANTHROPIC_API_KEY", None)


def _with_claude():
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"


def test_hybrid_reasoning_is_frontier_first():
    _no_claude()
    # No Claude key → frontier tier is just the provider, Om last.
    assert resolve("Orchestrator", "gemini", mode="hybrid") == [Backend.GEMINI, Backend.OM_THINK]


def test_hybrid_code_is_frontier_first():
    _no_claude()
    assert resolve("Coder", "gemini", mode="hybrid") == [Backend.GEMINI, Backend.OM_CODE]


def test_hybrid_style_leads_with_om():
    _no_claude()
    # Style leads with the Om "style layer".
    assert resolve("Designer", "gemini", mode="hybrid") == [Backend.OM_CODE, Backend.GEMINI]


def test_hybrid_reasoning_with_claude():
    _with_claude()
    # Claude joins frontier ahead of the provider; Om is the final fallback.
    assert resolve("Orchestrator", "gemini", mode="hybrid") == [
        Backend.CLAUDE, Backend.GEMINI, Backend.OM_THINK
    ]


def test_hybrid_code_with_claude_and_groq():
    _with_claude()
    assert resolve("Coder", "groq", mode="hybrid") == [
        Backend.CLAUDE, Backend.GROQ, Backend.OM_CODE
    ]


def test_frontier_mode_puts_om_last_even_for_style():
    _with_claude()
    # frontier mode: style also goes frontier-first.
    assert resolve("Designer", "gemini", mode="frontier") == [
        Backend.CLAUDE, Backend.GEMINI, Backend.OM_CODE
    ]


def test_om_mode_leads_with_om_everywhere():
    _with_claude()
    assert resolve("Orchestrator", "gemini", mode="om") == [
        Backend.OM_THINK, Backend.CLAUDE, Backend.GEMINI
    ]
    assert resolve("Coder", "gemini", mode="om") == [
        Backend.OM_CODE, Backend.CLAUDE, Backend.GEMINI
    ]


def test_unknown_agent_defaults_to_reasoning():
    _no_claude()
    assert resolve("MysteryAgent", "gemini", mode="hybrid") == [Backend.GEMINI, Backend.OM_THINK]


def test_provider_is_always_in_chain():
    """The user's provider is the guaranteed floor — always present."""
    _with_claude()
    for agent in ["Orchestrator", "Coder", "Designer", "Reviewer", "Editor", "Debugger"]:
        for provider in ["gemini", "groq"]:
            for mode in ["frontier", "hybrid", "om"]:
                chain = resolve(agent, provider, mode=mode)
                pb = Backend.GROQ if provider == "groq" else Backend.GEMINI
                assert pb in chain, f"{agent}/{provider}/{mode} missing provider floor"


def test_no_duplicate_backends():
    _with_claude()
    for agent in AGENT_NAMES:
        chain = resolve(agent, "gemini", mode="hybrid")
        assert len(chain) == len(set(chain)), f"{agent} chain has dupes: {chain}"


def test_claude_available_reflects_env():
    _no_claude()
    assert claude_available() is False
    _with_claude()
    assert claude_available() is True
    _no_claude()


AGENT_NAMES = [
    "Orchestrator", "Architect", "Debugger", "Coder", "Reviewer",
    "Repair", "Editor", "Designer", "Asset Artist", "WorkspaceChat",
]


if __name__ == "__main__":
    import sys

    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    _no_claude()  # cleanup
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    sys.exit(1 if failed else 0)
