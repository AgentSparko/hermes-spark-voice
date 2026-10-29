"""One live call: caller audio -> LFM3-Audio realtime -> spoken reply, with Hermes as the brain.

The transport (Matrix WebRTC or Telegram p2p) is anything that implements :class:`Transport`:
it pushes 48 kHz PCM16 frames in with ``on_audio_in`` and pulls PCM16 out with ``next_audio_out``.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import time
from typing import Awaitable, Callable, Optional, Protocol

import numpy as np

from .audio import Resampler, f32_to_pcm16, pcm16_to_f32, resample, rms, TG_RATE
from .config import Config
from .concierge import Concierge
from .hermes_client import HermesClient
from .omni import OmniClient
from .realtime import RealtimeClient, decode_audio_delta

logger = logging.getLogger(__name__)

TOOLS = [
    {"type": "function", "name": "ask_hermes",
     "description": "Hand a task to Hermes, the full agent (web, files, email, calendar, code, home, anything "
                    "that takes real work). Give it a complete, self-contained request. Returns a short "
                    "spoken-style summary; the full answer is posted to the chat.",
     "parameters": {"type": "object", "properties": {"request": {"type": "string", "description": "The task, as a clear brief with every detail the caller gave."}},
                    "required": ["request"]}},
    {"type": "function", "name": "look",
     "description": "Look at the caller's camera (video call) and describe what is visible, or answer a question about it.",
     "parameters": {"type": "object", "properties": {"question": {"type": "string", "description": "What to look for."}}, "required": []}},
    {"type": "function", "name": "hang_up",
     "description": "End the call after saying goodbye.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
]


class Transport(Protocol):
    """What the bridge needs from a call transport."""
    async def hangup(self) -> None: ...
    async def post_text(self, text: str) -> None: ...   # full Hermes answers go to the chat as text
    caller_name: str
    caller_id: str
    chat_id: str


class CallSession:
    def __init__(self, cfg: Config, transport: Transport, *, hermes: HermesClient | None = None,
                 omni: OmniClient | None = None, realtime_factory=None, tts_filler: Callable[[str], Awaitable[bytes]] | None = None,
                 on_state: Callable[[str, float], None] | None = None):
        self.cfg = cfg
        self.t = transport
        concierge = Concierge([cfg.hermes.concierge_url, cfg.audio.url], language=cfg.audio.language)
        self.hermes = hermes or HermesClient(cfg.hermes.url, cfg.hermes.key, cfg.hermes.model, cfg.hermes.timeout_s, concierge=concierge)
        self.omni = omni or (OmniClient(cfg.omni.url) if cfg.omni.enabled else None)
        self.rt = (realtime_factory or (lambda cb: RealtimeClient(cfg.audio.url, on_event=cb)))(self._on_event)
        self._tts_filler = tts_filler
        self._on_state = on_state or (lambda s, lvl: None)
        self.started = time.time()
        self.state = "connecting"
        self.transcript: collections.deque[str] = collections.deque(maxlen=cfg.control.transcript_lines)
        self.latest_frame: Optional[np.ndarray] = None   # RGB of the caller's camera
        self._out = collections.deque()                   # PCM16 48k mono chunks to send
        self._out_lock = asyncio.Lock()
        self._in_resampler: Resampler | None = None
        self._speech_ms = 0
        self._silence_ms = 0
        self._in_turn = False
        self._turn_started = 0.0
        self._pre_roll: collections.deque[bytes] = collections.deque(maxlen=15)  # ~300 ms of 20 ms frames
        self._responding = False
        self._text_buf = ""
        self._pending_tools: dict[str, dict] = {}
        self._closed = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()
        self._filler_task: asyncio.Task | None = None
        self._session_id = f"call:{transport.chat_id}:{int(self.started)}"

    # ---------- lifecycle ----------
    async def start(self) -> None:
        a = self.cfg.audio
        session = await self.rt.open(instructions=a.instructions + self._caller_note(), tools=TOOLS,
                                     output_modalities=a.output_modalities, max_output_tokens=a.max_output_tokens)
        self._in_resampler = Resampler(TG_RATE, 1, session.input_rate)
        self._set_state("listening")
        greeting = self.cfg.telegram.greeting
        if greeting:
            await self.rt.say_text(f"(The call just connected. Greet the caller with: \"{greeting}\")")

    def _caller_note(self) -> str:
        return f"\n\nThe caller is {self.t.caller_name or 'unknown'} (id {self.t.caller_id})."

    async def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        for task in list(self._tasks):
            task.cancel()
        await self.rt.close()
        self._set_state("ended")

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def _set_state(self, state: str, level: float = 0.0) -> None:
        self.state = state
        self._on_state(state, level)

    # ---------- audio in: VAD and turn taking ----------
    def on_audio_in(self, pcm16: bytes, rate: int = TG_RATE, channels: int = 1) -> None:
        """Called by the transport for every incoming frame (any size); schedules the async work."""
        if self._closed.is_set() or self._in_resampler is None:
            return
        if rate != TG_RATE or channels != 1:
            x = pcm16_to_f32(pcm16)
            if channels > 1:
                n = (len(x) // channels) * channels
                x = x[:n].reshape(-1, channels).mean(axis=1)
            pcm16 = f32_to_pcm16(resample(x, rate, TG_RATE))
        self._spawn(self._handle_in(pcm16))

    async def _handle_in(self, pcm16_48k_mono: bytes) -> None:
        a = self.cfg.audio
        x = pcm16_to_f32(pcm16_48k_mono)
        frame_ms = max(1, int(len(x) * 1000 / TG_RATE))
        level = rms(x)
        loud = level > a.vad_threshold
        model_pcm = self._in_resampler.feed(pcm16_48k_mono) if self._in_resampler else b""
        if self._in_turn:
            await self.rt.append_audio(model_pcm)
            if loud:
                self._speech_ms += frame_ms
                self._silence_ms = 0
            else:
                self._silence_ms += frame_ms
            self._set_state("listening", level)
            too_long = (time.time() - self._turn_started) > a.max_turn_s
            if (self._silence_ms >= a.vad_end_ms and self._speech_ms >= a.vad_start_ms) or too_long:
                await self._end_user_turn()
            return
        # not in a turn yet: keep a little pre-roll and wait for sustained speech
        self._pre_roll.append(model_pcm)
        if loud:
            self._speech_ms += frame_ms
        else:
            self._speech_ms = max(0, self._speech_ms - frame_ms // 2)
        if self._speech_ms >= a.vad_start_ms:
            if self._responding and a.barge_in:
                await self._interrupt()
            elif self._responding:
                return
            self._in_turn, self._silence_ms, self._turn_started = True, 0, time.time()
            for chunk in self._pre_roll:
                await self.rt.append_audio(chunk)
            self._pre_roll.clear()

    async def _end_user_turn(self) -> None:
        self._in_turn = False
        self._speech_ms = self._silence_ms = 0
        self._responding = True
        self._set_state("thinking")
        await self.rt.commit_and_respond()

    async def _interrupt(self) -> None:
        logger.debug("barge-in: cancelling response")
        try:
            await self.rt.cancel()
        except Exception:  # noqa: BLE001
            pass
        async with self._out_lock:
            self._out.clear()
        self._responding = False

    # ---------- audio out ----------
    async def _queue_out(self, pcm16_model_rate: bytes) -> None:
        out = f32_to_pcm16(resample(pcm16_to_f32(pcm16_model_rate), self.rt.session.output_rate, TG_RATE))
        async with self._out_lock:
            self._out.append(out)

    def next_audio_out(self, nbytes: int) -> bytes:
        """Transport pulls exactly ``nbytes`` of 48 kHz mono PCM16 (silence when idle). Sync, lock-free
        append/popleft on a deque is safe from the loop thread."""
        buf = bytearray()
        while len(buf) < nbytes and self._out:
            chunk = self._out.popleft()
            need = nbytes - len(buf)
            if len(chunk) > need:
                self._out.appendleft(chunk[need:])
                chunk = chunk[:need]
            buf += chunk
        if buf:
            lvl = rms(pcm16_to_f32(bytes(buf)))
            self._set_state("speaking", lvl)
        elif self.state == "speaking" and not self._responding:
            self._set_state("listening")
        if len(buf) < nbytes:
            buf += b"\x00" * (nbytes - len(buf))
        return bytes(buf)

    def has_pending_audio(self) -> bool:
        return bool(self._out)

    # ---------- realtime events ----------
    async def _on_event(self, event: dict) -> None:
        kind = event.get("type", "")
        if kind == "response.output_audio.delta":
            await self._queue_out(decode_audio_delta(event))
        elif kind == "response.output_text.delta":
            self._text_buf += event.get("delta") or ""
        elif kind == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                self._pending_tools[item.get("call_id") or item.get("id") or str(len(self._pending_tools))] = {
                    "name": item.get("name"), "args": item.get("arguments") or ""}
        elif kind == "response.function_call_arguments.done":
            call_id = event.get("call_id") or ""
            entry = self._pending_tools.setdefault(call_id, {"name": event.get("name"), "args": ""})
            entry["args"] = event.get("arguments") or entry["args"]
            entry["name"] = entry.get("name") or event.get("name")
        elif kind == "response.done":
            if self._text_buf.strip():
                self.transcript.append(f"hermes: {self._text_buf.strip()}")
            self._text_buf = ""
            pending, self._pending_tools = self._pending_tools, {}
            if pending:
                self._spawn(self._run_tools(pending))
            else:
                self._responding = False
                if not self._out:
                    self._set_state("listening")
        elif kind == "conversation.item.input_audio_transcription.completed":
            text = (event.get("transcript") or "").strip()
            if text:
                self.transcript.append(f"caller: {text}")
        elif kind == "_closed":
            logger.warning("realtime session dropped: %s", event.get("error"))
            await self.t.hangup()

    # ---------- tools ----------
    async def _run_tools(self, pending: dict[str, dict]) -> None:
        self._set_state("thinking")
        for call_id, call in pending.items():
            name = call.get("name") or ""
            try:
                args = json.loads(call.get("args") or "{}") if isinstance(call.get("args"), str) else (call.get("args") or {})
            except json.JSONDecodeError:
                args = {}
            logger.info("tool %s(%s)", name, json.dumps(args)[:200])
            self.transcript.append(f"tool: {name} {json.dumps(args, ensure_ascii=False)[:160]}")
            self._start_filler()
            try:
                if name == "ask_hermes":
                    result = await self._ask_hermes(str(args.get("request") or ""))
                elif name == "look":
                    result = await self._look(str(args.get("question") or self.cfg.omni.prompt))
                elif name == "hang_up":
                    result = "Call ended."
                    self._spawn(self._hangup_after(2.5))
                else:
                    result = f"Unknown tool {name}."
            except Exception as e:  # noqa: BLE001
                logger.exception("tool %s failed", name)
                result = f"The tool failed: {e}"
            finally:
                self._cancel_filler()
            if self._closed.is_set():
                return
            self._responding = True
            await self.rt.tool_result(call_id, result)

    async def _ask_hermes(self, request: str) -> str:
        if not request.strip():
            return "No request given."
        full = await self.hermes.ask(request, session_id=self._session_id, caller=self.t.caller_name, want_full=True)
        if full.full_text and full.full_text != full.spoken:
            self._spawn(self.t.post_text(full.full_text))
        return full.spoken

    async def _look(self, question: str) -> str:
        if self.latest_frame is None:
            return "There is no video from the caller right now."
        if self.omni is None:
            return "Vision is turned off."
        return await self.omni.describe(self.latest_frame, question)

    async def _hangup_after(self, seconds: float) -> None:
        deadline = time.time() + seconds + 8
        await asyncio.sleep(seconds)
        while self._out and time.time() < deadline:
            await asyncio.sleep(0.1)
        await self.t.hangup()

    def _start_filler(self) -> None:
        text = self.cfg.audio.filler_text
        if not text or self._tts_filler is None:
            return

        async def filler():
            await asyncio.sleep(self.cfg.audio.filler_after_s)
            try:
                pcm = await self._tts_filler(text)
                if pcm and not self._closed.is_set():
                    async with self._out_lock:
                        self._out.append(pcm)
            except Exception as e:  # noqa: BLE001
                logger.debug("filler failed: %s", e)
        self._filler_task = asyncio.create_task(filler())

    def _cancel_filler(self) -> None:
        if self._filler_task and not self._filler_task.done():
            self._filler_task.cancel()
        self._filler_task = None

    # ---------- misc ----------
    async def say(self, text: str) -> None:
        """Inject something to say (control API / Hermes tool voice_call_say)."""
        self._responding = True
        await self.rt.say_text(f"(Say this to the caller now, in your own words if needed: \"{text}\")")

    def status(self) -> dict:
        return {"ok": True, "active": not self._closed.is_set(), "state": self.state,
                "caller": {"id": self.t.caller_id, "name": self.t.caller_name, "chat": self.t.chat_id},
                "seconds": int(time.time() - self.started), "video": self.latest_frame is not None,
                "transcript": list(self.transcript)}

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
