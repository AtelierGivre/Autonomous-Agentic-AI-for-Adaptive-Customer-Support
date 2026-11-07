from __future__ import annotations

import asyncio

import pytest

from agents_app.mcp_client import MCPClient, ToolValidationError
from agents_app.mcp_server import ToolServer


@pytest.fixture()
def server() -> ToolServer:
    return ToolServer()


async def test_order_lookup_is_scoped_to_the_customer(server):
    own = await server.call("lookup_order", {"order_id": "ORD-1001"}, "demo")
    foreign = await server.call("lookup_order", {"order_id": "ORD-9001"}, "demo")
    assert own["ok"] is True and own["result"]["found"] is True
    assert foreign["ok"] is True and foreign["result"]["found"] is False


async def test_a_foreign_order_is_indistinguishable_from_a_missing_one(server):
    foreign = await server.call("lookup_order", {"order_id": "ORD-9001"}, "demo")
    missing = await server.call("lookup_order", {"order_id": "ORD-0000"}, "demo")
    # The caller already knows the ID it sent; the security property is that the
    # outcome and wording are identical, so existence is not disclosed.
    assert foreign["result"]["found"] is False
    assert foreign["result"]["message"] == missing["result"]["message"]


async def test_list_orders_only_returns_owned_orders(server):
    result = await server.call("list_orders", {}, "demo")
    ids = {order["order_id"] for order in result["result"]["orders"]}
    assert {"ORD-1001", "ORD-1002", "ORD-1003"} <= ids
    assert "ORD-9001" not in ids


async def test_ticket_creation_is_idempotent_per_customer(server):
    first = await server.call("create_ticket", {"subject": "refund", "details": "x"}, "demo")
    second = await server.call("create_ticket", {"subject": "refund", "details": "x"}, "demo")
    assert first["result"]["duplicate"] is False
    assert second["result"]["duplicate"] is True
    assert first["result"]["ticket_id"] == second["result"]["ticket_id"]
    assert server.ticket_count() == 1


async def test_unknown_tool_and_bad_arguments_are_rejected(server):
    assert (await server.call("drop_tables", {}, "demo"))["error"] == "unknown_tool"
    assert (await server.call("create_ticket", {"details": "no subject"}, "demo"))["error"] == "invalid_arguments"


def test_client_allow_list_rejects_unknown_tools_and_fields():
    client = MCPClient()
    with pytest.raises(ToolValidationError):
        client.validate("drop_tables", {})
    with pytest.raises(ToolValidationError):
        client.validate("lookup_order", {"order_id": "ORD-1001", "customer_id": "other"})
    with pytest.raises(ToolValidationError):
        client.validate("lookup_order", {})
    client.validate("lookup_order", {"order_id": "ORD-1001"})


async def test_client_returns_a_timeout_error(monkeypatch):
    client = MCPClient(timeout_s=0.01)

    async def slow(*_args, **_kwargs):
        await asyncio.sleep(1)
        return {"ok": True}

    monkeypatch.setattr(client.server, "call", slow)
    result = await client.call("lookup_order", {"order_id": "ORD-1001"}, "demo")
    assert result == {"ok": False, "tool": "lookup_order", "error": "tool_timeout"}


async def test_client_readiness_and_ping():
    client = MCPClient()
    await client.start()
    assert await client.ping() is True
    report = await client.readiness()
    assert report["ok"] is True
    assert report["tools"] == 4
    await client.stop()


async def test_jsonrpc_surface(server):
    ping = await server.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    listed = await server.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    called = await server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "lookup_order", "arguments": {"order_id": "ORD-1001"}},
        }
    )
    unknown = await server.handle_jsonrpc({"jsonrpc": "2.0", "id": 4, "method": "nope"})
    assert ping["result"]["ok"] is True
    assert "lookup_order" in listed["result"]["tools"]
    assert called["result"]["result"]["found"] is True
    assert unknown["error"]["code"] == -32601
