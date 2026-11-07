"""Interactive command-line chat with the multi-agent support system.

    python main.py
Commands:  /metrics  latency + cost table   /reset  clear memory   /quit  exit
"""
import asyncio

from agents_app.logging_setup import setup_logging
from agents_app.memory import ConversationMemory
from agents_app.workflow import SupportWorkflow

BANNER = """
==============================================================
  Autonomous Agentic AI for Adaptive Customer Support
  Planner -> Retriever || Tool agent -> Troubleshooter -> Verifier
  Try:  "My router keeps disconnecting"
        "Where is my order ORD-1003?"
        "Still not working, please escalate"   (then reply YES)
  Commands: /metrics  /reset  /quit
==============================================================
"""


async def main() -> None:
    setup_logging()
    wf = SupportWorkflow()
    print("Starting (connecting MCP tools, loading knowledge index)...")
    await wf.start()
    print(BANNER)
    memory = ConversationMemory()
    try:
        while True:
            try:
                user = (await asyncio.to_thread(input, "\nYou: ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user:
                continue
            if user == "/quit":
                break
            if user == "/reset":
                memory.clear()
                print("(memory cleared)")
                continue
            if user == "/metrics":
                print(wf.metrics.table())
                continue

            r = await wf.handle(user, memory)
            print("\n--- what the agents did ---")
            for line in r["debug"]["trace"]:
                print("  *", line)
            print(f"\nSupport: {r['answer']}")
            if r["sources"]:
                print("\nsources:", ", ".join(f"{s['source']} ({s['score']})" for s in r["sources"]))
            if r["needs_confirmation"]:
                print("(waiting for your YES / NO)")
            m = r["debug"]["metrics"]
            print(f"[{r['outcome']} | {m['llm_calls']} LLM calls | {m['total_latency_s']}s | "
                  f"{m['total_tokens']} tokens | ${m['total_cost_usd']}]")
    finally:
        await wf.stop()
        print("\nSession totals:\n" + wf.metrics.table())


if __name__ == "__main__":
    asyncio.run(main())
