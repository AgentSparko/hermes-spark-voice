"""Minimal client for llama-server's ``/v1/realtime`` (OpenAI Realtime schema) used with LFM3-Audio."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import websockets

logger = logging.getLogger(__name__)
Handler = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass
class Session:
    input_rate: int = 16000
    output_rate: int = 24000
    raw: dict = field(default_factory=dict)


def ws_url(base_http_url: str) -> str:
    url = base_http_url.rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    return url + "/v1/realtime"


class RealtimeClient:
    def __init__(self, base_url: str, *, on_event: Handler, connect=None):
        self._url = ws_url(base_url)
        self._on_event = on_event
        self._connect = connect or websockets.connect
        self._ws = None
        self._reader: asyncio.Task | None = None
        self.session = Session()
        self._created = asyncio.Event()
        self._updated = asyncio.Event()

    async def open(self, *, instructions: str, tools: list[dict], output_modalities: list[str],
                   max_output_tokens: int | None = None, timeout: float = 20.0) -> Session:
        self._ws = await self._connect(self._url, max_size=16 * 1024 * 1024, ping_interval=20)
        self._reader = asyncio.create_task(self._read_loop(), name="realtime-reader")
        await asyncio.wait_for(self._created.wait(), timeout)
        update: dict[str, Any] = {"type": "realtime", "instructions": instructions, "tools": tools,
                                  "tool_choice": "auto", "output_modalities": output_modalities}
        if max_output_tokens:
            update["max_output_tokens"] = max_output_tokens
        await self.send({"type": "session.update", "session": update})
        try:
            await asyncio.wait_for(self._updated.wait(), timeout)
        except asyncio.TimeoutError:
            logger.warning("realtime: no session.updated within %ss, continuing", timeout)
        return self.session

    async def send(self, event: dict) -> None:
        if self._ws is None:
            raise RuntimeError("realtime socket not open")
        await self._ws.send(json.dumps(event))

    async def append_audio(self, pcm16: bytes) -> None:
        if pcm16:
            await self.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm16).decode("ascii")})

    async def commit_and_respond(self) -> None:
        await self.send({"type": "input_audio_buffer.commit"})
        await self.send({"type": "response.create"})

    async def say_text(self, text: str) -> None:
        """Inject a user text turn (used for the greeting and for tool results that need a spoken summary)."""
        await self.send({"type": "conversation.item.create",
                         "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})
        await self.send({"type": "response.create"})

    async def tool_result(self, call_id: str, output: str) -> None:
        await self.send({"type": "conversation.item.create",
                         "item": {"type": "function_call_output", "call_id": call_id, "output": output}})
        await self.send({"type": "response.create"})

    async def cancel(self) -> None:
        await self.send({"type": "response.cancel"})

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            finally:
                self._ws = None

    def _apply_session(self, session: dict) -> None:
        self.session.raw = session
        audio = session.get("audio") or {}
        try:
            self.session.input_rate = int(((audio.get("input") or {}).get("format") or {}).get("rate") or self.session.input_rate)
            self.session.output_rate = int(((audio.get("output") or {}).get("format") or {}).get("rate") or self.session.output_rate)
        except (TypeError, ValueError):
            pass

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = event.get("type", "")
                if kind == "session.created":
                    self._apply_session(event.get("session") or {})
                    self._created.set()
                elif kind == "session.updated":
                    self._apply_session(event.get("session") or {})
                    self._updated.set()
                elif kind == "error":
                    logger.warning("realtime error: %s", (event.get("error") or {}).get("message") or event)
                result = self._on_event(event)
                if asyncio.iscoroutine(result):
                    await result
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - connection drop is reported through the closed event
            logger.info("realtime socket closed: %s", e)
            result = self._on_event({"type": "_closed", "error": str(e)})
            if asyncio.iscoroutine(result):
                await result


def decode_audio_delta(event: dict) -> bytes:
    data = event.get("delta") or event.get("audio") or ""
    return base64.b64decode(data) if data else b""
