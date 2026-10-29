import asyncio
import json

import pytest

from spark_calld import control
from spark_calld.concierge import Concierge, plain
from spark_calld.config import Config


def test_config_defaults_and_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_API_ID", raising=False)
    p = tmp_path / "c.yaml"
    p.write_text("telegram:\n  api_id: 5\n  allowed_users: [1, 2]\naudio:\n  vad_end_ms: 500\n  unknown_key: 1\nhermes:\n  key: k\n")
    cfg = Config.load(p)
    assert cfg.telegram.api_id == 5 and cfg.telegram.allowed_users == [1, 2]
    assert cfg.audio.vad_end_ms == 500 and cfg.audio.vad_start_ms == 200
    assert cfg.hermes.key == "k" and cfg.control.port == 8797


def test_plain_strips_markdown():
    out = plain("<think>hmm</think>Reasoning: because.\n\n# Title\n\n- **a**\n- `b`\n```py\nx=1\n```\nsee https://x.y/z at 127.0.0.1:8181 in ~/.hermes/config.yaml via run_agent() now")
    assert out == "because. Title - a - b the code see at the local address in the file via that now"


@pytest.mark.asyncio
async def test_concierge_brief_and_summary(monkeypatch):
    c = Concierge(["http://127.0.0.1:1"])
    assert await c.brief("short one") == "short one"
    assert await c.brief(" ".join(["w"] * 20)) == " ".join(["w"] * 20)   # server down -> passthrough

    async def fake(system, user, max_tokens=300):
        return "Goal: g\n1. a"
    monkeypatch.setattr(c, "_chat", fake)
    out = await c.brief(" ".join(["w"] * 20))
    assert out.startswith("Goal: g") and "Original spoken request" in out
    assert await c.spoken_summary("x" * 1000) == "Goal: g 1. a"
    assert await c.spoken_summary("Done, see /tmp/a.log") == "Goal: g 1. a"   # technical token -> rephrased
    assert await c.spoken_summary("fine.") == "fine."


@pytest.mark.asyncio
async def test_control_server_routes():
    async def status(_):
        return {"ok": True, "active": False}

    async def say(body):
        return {"ok": True, "text": body.get("text")}

    server = await control.serve("127.0.0.1", 0, {("GET", "/status"): status, ("POST", "/say"): say})
    port = server.sockets[0].getsockname()[1]

    async def req(method, path, body=None):
        r, w = await asyncio.open_connection("127.0.0.1", port)
        data = json.dumps(body).encode() if body is not None else b""
        w.write(f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {len(data)}\r\n\r\n".encode() + data)
        await w.drain()
        raw = await r.read()
        w.close()
        head, _, payload = raw.partition(b"\r\n\r\n")
        return int(head.split()[1]), json.loads(payload)

    assert await req("GET", "/status") == (200, {"ok": True, "active": False})
    assert await req("POST", "/say", {"text": "hi"}) == (200, {"ok": True, "text": "hi"})
    code, body = await req("GET", "/nope")
    assert code == 404 and body["ok"] is False
    server.close()
