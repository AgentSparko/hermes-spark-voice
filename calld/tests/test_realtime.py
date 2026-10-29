import asyncio
import base64
import json

import pytest

from spark_calld import realtime


class FakeWS:
    def __init__(self, incoming):
        self._incoming = list(incoming)
        self.sent = []
        self._queue = asyncio.Queue()
        for m in self._incoming:
            self._queue.put_nowait(json.dumps(m))

    async def send(self, data):
        self.sent.append(json.loads(data))
        if self.sent[-1]["type"] == "session.update":
            await self._queue.put(json.dumps({"type": "session.updated", "session": self.sent[-1]["session"]}))

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def close(self):
        await self._queue.put(None)


@pytest.mark.asyncio
async def test_open_reads_rates_and_sends_session_update():
    ws = FakeWS([{"type": "session.created", "session": {"audio": {"input": {"format": {"rate": 16000}},
                                                                     "output": {"format": {"rate": 24000}}}}}])
    events = []

    async def connect(url, **kw):
        assert url == "ws://127.0.0.1:8181/v1/realtime"
        return ws

    rt = realtime.RealtimeClient("http://127.0.0.1:8181", on_event=events.append, connect=connect)
    session = await rt.open(instructions="be nice", tools=[{"name": "x"}], output_modalities=["audio"], max_output_tokens=50)
    assert (session.input_rate, session.output_rate) == (16000, 24000)
    upd = ws.sent[0]
    assert upd["type"] == "session.update" and upd["session"]["instructions"] == "be nice"
    assert upd["session"]["tools"] == [{"name": "x"}] and upd["session"]["max_output_tokens"] == 50
    await rt.append_audio(b"\x01\x02")
    await rt.commit_and_respond()
    await rt.tool_result("c1", "done")
    types = [m["type"] for m in ws.sent]
    assert types[1:] == ["input_audio_buffer.append", "input_audio_buffer.commit", "response.create",
                         "conversation.item.create", "response.create"]
    assert base64.b64decode(ws.sent[1]["audio"]) == b"\x01\x02"
    await rt.close()
    assert any(e["type"] == "session.created" for e in events)


def test_ws_url_and_decode():
    assert realtime.ws_url("https://host/") == "wss://host/v1/realtime"
    assert realtime.decode_audio_delta({"delta": base64.b64encode(b"ab").decode()}) == b"ab"
    assert realtime.decode_audio_delta({}) == b""
