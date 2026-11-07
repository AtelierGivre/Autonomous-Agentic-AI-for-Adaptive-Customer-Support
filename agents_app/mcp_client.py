"""Validating, timeout-guarded client for the tool server.

Only allow-listed tools and argument names can reach the server. That is the
last line of defence against a model that tries to invent a tool or smuggle an
unexpected field (for example a ``customer_id`` override).
"""

from __future__ import annotations

import asyncio

from agents_app.mcp_server import TOOL_SCHEMAS, ToolServer

TOOL_ARGS: dict[str, set[str]] = {
    "lookup_order": {"order_id"},
    "list_orders": set(),
    "create_ticket": {"subject", "details"},
    "escalate_to_human": {"reason"},
}
REQUIRED_ARGS: dict[str, set[str]] = {
    "lookup_order": {"order_id"},
    "create_ticket": {"subject"},
}


class ToolValidationError(ValueError):
    pass


class MCPClient:
    def __init__(self, server: ToolServer | None = None, timeout_s: float = 5.0) -> None:
        self.server = server or ToolServer()
        self.timeout_s = timeout_s
        self.started = False
        self.calls = 0

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def ping(self) -> bool:
        try:
            return await asyncio.wait_for(self.server.ping(), timeout=1.0)
        except Exception:  # noqa: BLE001
            return False

    def validate(self, name: str, args: dict) -> None:
        if name not in TOOL_ARGS:
            raise ToolValidationError(f"tool not allowed: {name}")
        unknown = set(args) - TOOL_ARGS[name]
        if unknown:
            raise ToolValidationError(f"unexpected argument(s): {', '.join(sorted(unknown))}")
        missing = REQUIRED_ARGS.get(name, set()) - set(args)
        if missing:
            raise ToolValidationError(f"missing argument(s): {', '.join(sorted(missing))}")

    async def call(self, name: str, args: dict, customer_id: str) -> dict:
        self.validate(name, args)
        self.calls += 1
        try:
            return await asyncio.wait_for(
                self.server.call(name, args, customer_id), timeout=self.timeout_s
            )
        except TimeoutError:
            return {"ok": False, "tool": name, "error": "tool_timeout"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "tool": name, "error": "tool_failed", "detail": str(exc)}

    def tools(self) -> dict:
        return dict(TOOL_SCHEMAS)

    async def readiness(self) -> dict:
        return {"ok": await self.ping(), "tools": len(TOOL_ARGS), "calls": self.calls}
