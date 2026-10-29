"""Interactive command-line chat with the multi-agent support system.

    python main.py
Commands:  /metrics  show latency + cost table   /reset  clear memory   /quit  exit
"""
import asyncio

from agents_app.memory import ConversationMemory
from agents_app.workflow import SupportWorkflow

BANNER = """
==============================================================
  Autonomous Agentic AI for Adaptive Customer Support
  Planner -> Retriever / Tool agent -> Troubleshooter -> Verifier
  Try:  "My router keeps disconnecting"
        "Where is my order ORD-1003?"
        "Still not working, please escalate"
  Commands: /metrics  /reset  /quit
==============================================================
"""


async def main() -> None:
    wf = SupportWorkflow()
    print("Starting (connecting MCP tools, loading FAISS index)...")
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

            result = await wf.handle(user, memory)
            print("\n--- what the agents did ---")
            print("plan :", " -> ".join(s["agent"] for s in result["plan"]["steps"]))
            for line in result["trace"]:
                print("  *", line)
            print(f"\nSupport: {result['answer']}")
            if result["sources"]:
                print("\nsources:", ", ".join(result["sources"]))
            m = result["metrics"]
            print(f"[{m['llm_calls']} LLM calls | {m['total_latency_s']}s | "
                  f"{m['total_tokens']} tokens | ${m['total_cost_usd']}]")
    finally:
        await wf.stop()
        print("\nSession totals:\n" + wf.metrics.table())


if __name__ == "__main__":
    asyncio.run(main())
