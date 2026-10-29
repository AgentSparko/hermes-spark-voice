import asyncio
import base64
import json

import numpy as np
import pytest

from spark_calld import bridge
from spark_calld.audio import TG_RATE, f32_to_pcm16
from spark_calld.config import Config
from spark_calld.hermes_client import Answer
from spark_calld.realtime import Session


class FakeRT:
    def __init__(self, cb):
        self.cb = cb
        self.sent = []
        self.session = Session(input_rate=16000, output_rate=24000)

    async def open(self, **kw):
        self.open_kwargs = kw
        return self.session

    async def send(self, e):
        self.sent.append(e)

    async def append_audio(self, pcm):
        self.sent.append({"type": "input_audio_buffer.append", "n": len(pcm)})

    async def commit_and_respond(self):
        self.sent.append({"type": "commit"})

    async def say_text(self, text):
        self.sent.append({"type": "say", "text": text})

    async def tool_result(self, call_id, output):
        self.sent.append({"type": "tool_result", "call_id": call_id, "output": output})

    async def cancel(self):
        self.sent.append({"type": "cancel"})

    async def close(self):
        self.sent.append({"type": "close"})


class FakeTransport:
    caller_name, caller_id, chat_id = "Alex", "42", "42"

    def __init__(self):
        self.hung_up = False
        self.posted = []

    async def hangup(self):
        self.hung_up = True

    async def post_text(self, text):
        self.posted.append(text)


class FakeHermes:
    def __init__(self):
        self.requests = []

    async def ask(self, request, *, session_id, caller="", want_full=False):
        self.requests.append((request, session_id, caller))
        return Answer("It is sunny.", full_text="# Weather\n\nIt is sunny in Helsinki, 14 C, wind 3 m/s...")


def frame(ms=20, loud=False):
    n = TG_RATE * ms // 1000
    x = (np.random.default_rng(0).standard_normal(n) * 0.2).astype(np.float32) if loud else np.zeros(n, dtype=np.float32)
    return f32_to_pcm16(x)


async def make_session(cfg=None):
    cfg = cfg or Config()
    cfg.audio.filler_text = ""
    t = FakeTransport()
    holder = {}

    def factory(cb):
        holder["rt"] = FakeRT(cb)
        return holder["rt"]

    s = bridge.CallSession(cfg, t, hermes=FakeHermes(), omni=None, realtime_factory=factory)
    await s.start()
    return s, holder["rt"], t


@pytest.mark.asyncio
async def test_start_greets_and_sets_tools():
    s, rt, _ = await make_session()
    assert [t["name"] for t in rt.open_kwargs["tools"]] == ["ask_hermes", "look", "hang_up"]
    assert "Alex" in rt.open_kwargs["instructions"]
    assert rt.sent[0]["type"] == "say" and "Hermes here" in rt.sent[0]["text"]
    assert s.state == "listening"


@pytest.mark.asyncio
async def test_vad_opens_turn_and_commits_after_silence():
    s, rt, _ = await make_session()
    for _ in range(12):            # 240 ms of speech > vad_start_ms (200)
        s.on_audio_in(frame(loud=True))
        await asyncio.sleep(0)
    await asyncio.sleep(0.01)
    assert s._in_turn is True
    appended = [e for e in rt.sent if e["type"] == "input_audio_buffer.append"]
    assert appended and all(e["n"] > 0 for e in appended)
    for _ in range(40):            # 800 ms silence > vad_end_ms (700)
        s.on_audio_in(frame())
        await asyncio.sleep(0)
    await asyncio.sleep(0.01)
    assert any(e["type"] == "commit" for e in rt.sent)
    assert s.state == "thinking" and s._responding


@pytest.mark.asyncio
async def test_audio_out_is_resampled_and_pulled_in_fixed_chunks():
    s, rt, _ = await make_session()
    pcm24 = f32_to_pcm16(np.ones(2400, dtype=np.float32) * 0.3)   # 100 ms @ 24k
    await rt.cb({"type": "response.output_audio.delta", "delta": base64.b64encode(pcm24).decode()})
    first = s.next_audio_out(960 * 2)                              # 20 ms @ 48k
    assert len(first) == 1920 and s.state == "speaking"
    total = len(first)
    while s.has_pending_audio():
        total += len(s.next_audio_out(1920))
    assert total == 4800 * 2 + (1920 - (4800 * 2) % 1920) % 1920  # 100 ms @48k, padded to chunk
    assert s.next_audio_out(10) == b"\x00" * 10


@pytest.mark.asyncio
async def test_tool_call_goes_to_hermes_and_full_text_to_chat():
    s, rt, t = await make_session()
    await rt.cb({"type": "response.output_item.added", "item": {"type": "function_call", "call_id": "c1", "name": "ask_hermes"}})
    await rt.cb({"type": "response.function_call_arguments.done", "call_id": "c1", "name": "ask_hermes",
                 "arguments": json.dumps({"request": "weather in Helsinki"})})
    await rt.cb({"type": "response.done"})
    await asyncio.sleep(0.05)
    assert s.hermes.requests[0][0] == "weather in Helsinki" and s.hermes.requests[0][1].startswith("call:42:")
    result = [e for e in rt.sent if e["type"] == "tool_result"][0]
    assert result["call_id"] == "c1" and result["output"] == "It is sunny."
    assert t.posted and t.posted[0].startswith("# Weather")
    assert any(line.startswith("tool: ask_hermes") for line in s.transcript)


@pytest.mark.asyncio
async def test_hang_up_tool_ends_call():
    cfg = Config()
    s, rt, t = await make_session(cfg)
    await rt.cb({"type": "response.output_item.added", "item": {"type": "function_call", "call_id": "c2", "name": "hang_up"}})
    await rt.cb({"type": "response.function_call_arguments.done", "call_id": "c2", "arguments": "{}"})
    s._hangup_after = lambda seconds: t.hangup()  # skip the goodbye wait
    await rt.cb({"type": "response.done"})
    await asyncio.sleep(0.05)
    assert t.hung_up


@pytest.mark.asyncio
async def test_barge_in_cancels_response():
    s, rt, _ = await make_session()
    s._responding = True
    await rt.cb({"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x00" * 4800).decode()})
    for _ in range(12):
        s.on_audio_in(frame(loud=True))
        await asyncio.sleep(0)
    await asyncio.sleep(0.01)
    assert any(e["type"] == "cancel" for e in rt.sent) and not s.has_pending_audio() and s._in_turn


@pytest.mark.asyncio
async def test_look_without_video_and_status():
    s, rt, _ = await make_session()
    assert await s._look("what") == "There is no video from the caller right now."
    st = s.status()
    assert st["active"] and st["caller"]["name"] == "Alex" and st["video"] is False
    await s.close()
    assert s.status()["active"] is False
