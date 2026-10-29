"""Tiny loopback HTTP control API (no extra dependency): GET /status, POST /hangup, POST /say {text}."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)
Routes = dict[tuple[str, str], Callable[[dict], Awaitable[dict]]]


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, routes: Routes) -> None:
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
        request_line, *header_lines = head.decode("latin1").split("\r\n")
        method, path, _ = (request_line.split(" ") + ["", ""])[:3]
        headers = {k.strip().lower(): v.strip() for k, _, v in (h.partition(":") for h in header_lines if ":" in h)}
        length = int(headers.get("content-length") or 0)
        body = json.loads((await reader.readexactly(length)).decode() or "{}") if length else {}
        route = routes.get((method.upper(), path.split("?")[0]))
        if route is None:
            status, payload = 404, {"ok": False, "error": "no such route"}
        else:
            try:
                status, payload = 200, await route(body if isinstance(body, dict) else {})
            except Exception as e:  # noqa: BLE001
                logger.exception("control route %s failed", path)
                status, payload = 500, {"ok": False, "error": str(e)}
    except (asyncio.IncompleteReadError, asyncio.TimeoutError, ValueError, json.JSONDecodeError) as e:
        status, payload = 400, {"ok": False, "error": f"bad request: {e}"}
    data = json.dumps(payload, ensure_ascii=False).encode()
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"}[status]
    writer.write(f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode() + data)
    try:
        await writer.drain()
    finally:
        writer.close()


async def serve(host: str, port: int, routes: Routes) -> asyncio.AbstractServer:
    server = await asyncio.start_server(lambda r, w: _handle(r, w, routes), host, port)
    logger.info("control api on http://%s:%d", host, port)
    return server
