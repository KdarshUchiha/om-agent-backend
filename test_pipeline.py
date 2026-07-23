"""
Diagnostic test — runs the full pipeline and prints every event + raw agent output.
Usage: python test_pipeline.py <api_key> [provider] [prompt]
"""
import asyncio
import sys
from agents.orchestrator_pipeline import run_pipeline

async def main():
    api_key = sys.argv[1] if len(sys.argv) > 1 else ""
    provider = sys.argv[2] if len(sys.argv) > 2 else "gemini"
    prompt   = sys.argv[3] if len(sys.argv) > 3 else "build a snake game"

    if not api_key:
        print("Usage: python test_pipeline.py <api_key> [gemini|groq] [prompt]")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  PROMPT   : {prompt}")
    print(f"  PROVIDER : {provider}")
    print(f"{'='*60}\n")

    final_files = []
    agent_outputs = {}

    async for event in run_pipeline(prompt, api_key, provider):
        t = event.get("type")
        agent = event.get("agent", "")

        if t == "agent_start":
            print(f"\n▶ [{agent}] STARTED")
        elif t == "agent_thinking":
            print(f"  ⚡ [{agent}] {event.get('message','')}")
        elif t == "agent_output":
            agent_outputs.setdefault(agent, "")
            agent_outputs[agent] += event.get("chunk", "")
        elif t == "agent_done":
            out = agent_outputs.get(agent, "")
            print(f"✅ [{agent}] DONE — {len(out)} chars")
            if agent == "Reviewer":
                print(f"\n--- REVIEWER RAW OUTPUT (first 1000 chars) ---")
                print(repr(out[:1000]))
                print(f"--- REVIEWER RAW OUTPUT (last 500 chars) ---")
                print(repr(out[-500:]))
        elif t == "final_output":
            final_files = event.get("files", [])
            summary = event.get("summary", "")
            print(f"\n{'='*60}")
            print(f"  FINAL OUTPUT — Summary: {summary}")
            print(f"  Files ({len(final_files)}):")
            for f in final_files:
                content = f.get("content", "")
                has_js = "<script" in content or "function" in content
                has_html = "<!DOCTYPE" in content or "<html" in content
                has_css = "<style" in content or "{" in content
                print(f"\n  📄 {f['name']} — {len(content)} chars | html:{has_html} css:{has_css} js:{has_js}")
                print(f"     Last 300 chars: {content[-300:]}")
            print(f"{'='*60}")
        elif t == "error":
            print(f"\n❌ ERROR [{agent}]: {event.get('message','')}")
        elif t == "done":
            print("\n✅ Pipeline complete.")

    if not final_files:
        print("\n⚠️  NO FILES in final_output.")

asyncio.run(main())
