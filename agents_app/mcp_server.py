"""Tool server (MCP-compatible surface).

The demo ships an in-process server so offline tests need no subprocess or
network. Every tool enforces the caller's customer ID *inside* the tool; the
model neither supplies nor sees that ID. A foreign order and a missing order
are deliberately indistinguishable.

Run standalone to speak newline-delimited JSON-RPC on stdio:

    python -m agents_app.mcp_server
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading

# Fake order data. ORD-9001 belongs to a different customer on purpose.
ORDERS: dict[str, dict] = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "customer_id": "demo",
        "item": "AeroRouter X2",
        "status": "delivered",
        "carrier": "BlueDart",
        "eta": None,
        "delivered_on": "2026-10-28",
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "customer_id": "demo",
        "item": "Mesh Wi-Fi Extender",
        "status": "in transit",
        "carrier": "Delhivery",
        "eta": "2026-11-08",
        "delivered_on": None,
    },
    "ORD-1003": {
        "order_id": "ORD-1003",
        "customer_id": "demo",
        "item": "AeroRouter X2 + installation kit",
        "status": "delayed",
        "carrier": "BlueDart",
        "eta": "2026-11-12",
        "delivered_on": None,
        "delay_reason": "weather disruption at the regional hub",
    },
    "ORD-9001": {
        "order_id": "ORD-9001",
        "customer_id": "other",
        "item": "AeroRouter X2",
        "status": "delivered",
        "carrier": "BlueDart",
        "eta": None,
        "delivered_on": "2026-10-30",
    },
}

TOOL_SCHEMAS = {
    "lookup_order": {
        "description": "Look up one order owned by the calling customer.",
        "args": {"order_id": "string, e.g. ORD-1003"},
    },
    "list_orders": {
        "description": "List the calling customer's recent orders.",
        "args": {},
    },
    "create_ticket": {
        "description": "Open a support ticket for the calling customer (write action).",
        "args": {"subject": "string", "details": "string"},
    },
    "escalate_to_human": {
        "description": "Escalate to a human agent (write action).",
        "args": {"reason": "string"},
    },
}


class ToolServer:
    def __init__(self) -> None:
        self._tickets: list[dict] = []
        self._ticket_seq = 1000
        self._lock = threading.Lock()

    async def ping(self) -> bool:
        return True

    async def call(self, name: str, args: dict, customer_id: str) -> dict:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"ok": False, "error": "unknown_tool", "tool": name}
        try:
            result = await handler(args or {}, customer_id)
        except (KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "error": "invalid_arguments", "detail": str(exc), "tool": name}
        return {"ok": True, "tool": name, "result": result}

    # ------------------------------------------------------------------ read tools
    async def _tool_lookup_order(self, args: dict, customer_id: str) -> dict:
        order_id = str(args.get("order_id", "")).upper().strip()
        order = ORDERS.get(order_id)
        if not order or order["customer_id"] != customer_id:
            # Do not reveal that a foreign order exists.
            return {"found": False, "order_id": order_id, "message": "no order with that ID on this account"}
        return {"found": True, **{k: v for k, v in order.items() if k != "customer_id"}}

    async def _tool_list_orders(self, args: dict, customer_id: str) -> dict:
        orders = [
            {k: v for k, v in order.items() if k != "customer_id"}
            for order in ORDERS.values()
            if order["customer_id"] == customer_id
        ]
        return {"found": bool(orders), "orders": orders}

    # ----------------------------------------------------------------- write tools
    async def _tool_create_ticket(self, args: dict, customer_id: str) -> dict:
        subject = str(args.get("subject", "")).strip()
        details = str(args.get("details", "")).strip()
        if not subject:
            raise ValueError("subject is required")
        with self._lock:
            for ticket in self._tickets:
                if ticket["customer_id"] == customer_id and ticket["subject"] == subject:
                    return {**ticket, "duplicate": True}
            self._ticket_seq += 1
            ticket = {
                "ticket_id": f"TCK-{self._ticket_seq}",
                "customer_id": customer_id,
                "subject": subject,
                "details": details,
                "status": "open",
                "duplicate": False,
            }
            self._tickets.append(ticket)
            return ticket

    async def _tool_escalate_to_human(self, args: dict, customer_id: str) -> dict:
        reason = str(args.get("reason", "")).strip() or "customer requested escalation"
        return await self._tool_create_ticket(
            {"subject": "Human escalation requested", "details": reason}, customer_id
        )

    def ticket_count(self) -> int:
        return len(self._tickets)

    # --------------------------------------------------------------- JSON-RPC
    async def handle_jsonrpc(self, payload: dict, customer_id: str = "demo") -> dict:
        method = payload.get("method", "")
        if method == "ping":
            return {"jsonrpc": "2.0", "id": payload.get("id"), "result": {"ok": await self.ping()}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": payload.get("id"), "result": {"tools": TOOL_SCHEMAS}}
        if method == "tools/call":
            params = payload.get("params", {})
            result = await self.call(params.get("name", ""), params.get("arguments", {}), customer_id)
            return {"jsonrpc": "2.0", "id": payload.get("id"), "result": result}
        return {
            "jsonrpc": "2.0",
            "id": payload.get("id"),
            "error": {"code": -32601, "message": f"unknown method: {method}"},
        }


async def _stdio_loop() -> None:
    server = ToolServer()
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        try:
            payload = json.loads(line)
            response = await server.handle_jsonrpc(payload)
        except json.JSONDecodeError as exc:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    asyncio.run(_stdio_loop())
