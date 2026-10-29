"""Matrix VoIP transport: the Hermes bot account answers 1:1 audio/video calls over WebRTC (aiortc).

Signalling follows the Matrix VoIP spec (m.call.invite / answer / candidates / hangup, version "1").
The homeserver's TURN credentials (``/voip/turnServer``) are used so calls work from a phone on
mobile data to a Spark behind NAT. Reuses Hermes' MATRIX_HOMESERVER / MATRIX_ACCESS_TOKEN /
MATRIX_USER_ID from ``~/.hermes/.env`` (a second device on the same account).
"""
from __future__ import annotations

import asyncio
import fractions
import logging
import time
import uuid
from typing import Callable, Optional

import httpx
import numpy as np

from .audio import TG_RATE, pcm16_to_f32
from .config import Config, read_hermes_env

logger = logging.getLogger(__name__)

try:  # heavy optional deps: the daemon still runs (telegram only) without them
    from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack, MediaStreamError
    from av import AudioFrame, VideoFrame
    HAVE_AIORTC = True
except Exception:  # noqa: BLE001
    HAVE_AIORTC = False
    AudioStreamTrack = VideoStreamTrack = object  # type: ignore

try:
    import nio
    HAVE_NIO = True
except Exception:  # noqa: BLE001
    HAVE_NIO = False

FRAME_MS = 20
SAMPLES = TG_RATE * FRAME_MS // 1000


class BridgeAudioTrack(AudioStreamTrack):
    """Outgoing audio: 20 ms s16 mono 48 kHz frames pulled from the bridge (silence when idle)."""
    kind = "audio"

    def __init__(self, pull: Callable[[int], bytes]):
        super().__init__()
        self._pull = pull
        self._ts = 0
        self._start = None

    async def recv(self):
        if self._start is None:
            self._start = time.time()
        self._ts += SAMPLES
        wait = self._start + self._ts / TG_RATE - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        pcm = self._pull(SAMPLES * 2)
        frame = AudioFrame(format="s16", layout="mono", samples=SAMPLES)
        frame.planes[0].update(pcm)
        frame.sample_rate = TG_RATE
        frame.pts = self._ts
        frame.time_base = fractions.Fraction(1, TG_RATE)
        return frame


class AvatarVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, next_rgb: Callable[[], np.ndarray], fps: int = 15):
        super().__init__()
        self._next = next_rgb
        self._fps = fps

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        await asyncio.sleep(max(0.0, 1.0 / self._fps - 0.002))
        frame = VideoFrame.from_ndarray(self._next(), format="rgb24")
        frame.pts, frame.time_base = pts, time_base
        return frame


class MatrixCall:
    """One answered call; implements the bridge Transport protocol."""

    def __init__(self, transport: "MatrixTransport", room_id: str, call_id: str, caller_id: str, caller_name: str,
                 remote_party: str | None, video: bool):
        self.transport, self.room_id, self.call_id = transport, room_id, call_id
        self.caller_id, self.caller_name, self.chat_id = caller_id, caller_name, room_id
        self.remote_party, self.video = remote_party, video
        self.party_id = uuid.uuid4().hex[:8]
        self.pc: Optional[RTCPeerConnection] = None
        self.session = None
        self._ended = False

    async def hangup(self, reason: str = "user_hangup") -> None:
        if self._ended:
            return
        self._ended = True
        try:
            await self.transport.send_call_event(self.room_id, "m.call.hangup",
                                                 {"call_id": self.call_id, "party_id": self.party_id, "version": "1", "reason": reason})
        except Exception as e:  # noqa: BLE001
            logger.debug("hangup event failed: %s", e)
        await self.transport.end_call(self)

    async def post_text(self, text: str) -> None:
        await self.transport.send_text(self.room_id, text)


class MatrixTransport:
    def __init__(self, cfg: Config, on_incoming, on_ended, *, homeserver: str = "", token: str = "", user_id: str = "",
                 allowed_users: list[str] | None = None):
        self.cfg = cfg
        self.on_incoming, self.on_ended = on_incoming, on_ended
        self.homeserver = (homeserver or read_hermes_env("MATRIX_HOMESERVER") or "").rstrip("/")
        self.token = token or read_hermes_env("MATRIX_ACCESS_TOKEN") or ""
        self.user_id = user_id or read_hermes_env("MATRIX_USER_ID") or ""
        self.allowed_users = allowed_users or []
        self.client = None
        self.active: Optional[MatrixCall] = None
        self._pending: dict[str, dict] = {}   # call_id -> invite content waiting for candidates
        self._sync_task: Optional[asyncio.Task] = None

    def available(self) -> tuple[bool, str]:
        if not HAVE_NIO or not HAVE_AIORTC:
            return False, "matrix calls need: pip install 'spark-calld[matrix]' (matrix-nio, aiortc, av)"
        if not (self.homeserver and self.token):
            return False, "MATRIX_HOMESERVER / MATRIX_ACCESS_TOKEN not set (Hermes ~/.hermes/.env)"
        return True, "ok"

    # ---------- matrix plumbing ----------
    async def start(self) -> None:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(why)
        store = str(self.cfg.telegram.session).rsplit("/", 1)[0] + "/matrix-calld"
        import os
        os.makedirs(store, exist_ok=True)
        cfg = nio.AsyncClientConfig(store_sync_tokens=True, encryption_enabled=False)
        self.client = nio.AsyncClient(self.homeserver, self.user_id, store_path=store, config=cfg, device_id="SPARKCALLD")
        self.client.access_token = self.token
        self.client.user_id = self.user_id or self.client.user_id
        if not self.user_id:
            who = await self.client.whoami()
            self.user_id = getattr(who, "user_id", "") or ""
            self.client.user_id = self.user_id
        self.client.add_event_callback(self._on_call_event, (nio.CallInviteEvent, nio.CallCandidatesEvent, nio.CallHangupEvent))
        self.client.add_event_callback(self._on_unknown, nio.UnknownEvent)
        # skip history: one sync with a filter that returns nothing old
        await self.client.sync(timeout=0, full_state=False, sync_filter={"room": {"timeline": {"limit": 1}}})
        self._sync_task = asyncio.create_task(self.client.sync_forever(timeout=30000, full_state=False), name="matrix-sync")
        logger.info("matrix: listening for calls as %s", self.user_id)

    async def stop(self) -> None:
        if self.active:
            await self.active.hangup("user_hangup")
        if self._sync_task:
            self._sync_task.cancel()
        if self.client:
            await self.client.close()

    async def send_call_event(self, room_id: str, event_type: str, content: dict) -> None:
        await self.client.room_send(room_id, event_type, content, ignore_unverified_devices=True)

    async def send_text(self, room_id: str, text: str) -> None:
        await self.client.room_send(room_id, "m.room.message", {"msgtype": "m.text", "body": text}, ignore_unverified_devices=True)

    async def turn_servers(self) -> list:
        try:
            async with httpx.AsyncClient(timeout=10) as http:
                resp = await http.get(self.homeserver + "/_matrix/client/v3/voip/turnServer",
                                      headers={"Authorization": f"Bearer {self.token}"})
            if resp.status_code != 200:
                return []
            data = resp.json()
            uris = data.get("uris") or []
            if not uris:
                return []
            return [RTCIceServer(urls=uris, username=data.get("username"), credential=data.get("password"))]
        except Exception as e:  # noqa: BLE001
            logger.debug("no TURN from homeserver: %s", e)
            return []

    def _allowed(self, sender: str) -> bool:
        return (not self.allowed_users) or sender in self.allowed_users

    # ---------- events ----------
    async def _on_unknown(self, room, event) -> None:
        # nio has typed events for invite/candidates/hangup; reject/select_answer arrive as UnknownEvent
        if getattr(event, "type", "") in ("m.call.reject",):
            content = event.source.get("content", {})
            if self.active and content.get("call_id") == self.active.call_id:
                await self.active.hangup("remote_reject")

    async def _on_call_event(self, room, event) -> None:
        if event.sender == self.user_id:
            return
        if isinstance(event, nio.CallInviteEvent):
            await self._handle_invite(room, event)
        elif isinstance(event, nio.CallCandidatesEvent):
            if self.active and event.call_id == self.active.call_id:
                await self._add_candidates(self.active, event.candidates)
            elif event.call_id in self._pending:
                self._pending[event.call_id].setdefault("candidates", []).extend(event.candidates)
        elif isinstance(event, nio.CallHangupEvent):
            if self.active and event.call_id == self.active.call_id:
                logger.info("matrix: caller hung up")
                self.active._ended = True
                await self.end_call(self.active)

    async def _handle_invite(self, room, event) -> None:
        sender = event.sender
        if not self._allowed(sender):
            logger.info("matrix: call from %s refused (not allowed)", sender)
            await self.send_call_event(room.room_id, "m.call.hangup",
                                       {"call_id": event.call_id, "party_id": uuid.uuid4().hex[:8], "version": "1", "reason": "user_busy"})
            return
        if self.active and not self.active._ended:
            await self.send_call_event(room.room_id, "m.call.hangup",
                                       {"call_id": event.call_id, "party_id": uuid.uuid4().hex[:8], "version": "1", "reason": "user_busy"})
            return
        content = event.source.get("content", {})
        offer = (content.get("offer") or {}).get("sdp") or ""
        video = "m=video" in offer
        name = room.user_name(sender) if hasattr(room, "user_name") else sender
        call = MatrixCall(self, room.room_id, event.call_id, sender, name or sender, content.get("party_id"), video)
        self.active = call
        pending = self._pending.pop(event.call_id, {})
        try:
            await self._answer(call, offer, pending.get("candidates") or [])
        except Exception as e:  # noqa: BLE001
            logger.exception("matrix: answering failed: %s", e)
            await call.hangup("ice_failed")

    async def _answer(self, call: MatrixCall, offer_sdp: str, early_candidates: list) -> None:
        pc = RTCPeerConnection(RTCConfiguration(iceServers=(await self.turn_servers()) or [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]))
        call.pc = pc
        session = await self.on_incoming(call)      # the bridge CallSession (started)
        call.session = session
        pc.addTrack(BridgeAudioTrack(session.next_audio_out))
        if call.video and self.cfg.video.avatar and getattr(session, "avatar", None) is not None:
            pc.addTrack(AvatarVideoTrack(session.avatar.frame_rgb, self.cfg.video.fps))

        @pc.on("track")
        def on_track(track):
            asyncio.ensure_future(self._consume(track, session, call))

        @pc.on("connectionstatechange")
        async def on_state():
            logger.info("matrix: webrtc %s", pc.connectionState)
            if pc.connectionState in ("failed", "closed") and not call._ended:
                await call.hangup("ice_failed")

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
        await self._add_candidates(call, early_candidates)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)      # aiortc gathers ICE before this returns (non-trickle)
        await self.send_call_event(call.room_id, "m.call.answer", {
            "call_id": call.call_id, "party_id": call.party_id, "version": "1",
            "answer": {"type": "answer", "sdp": pc.localDescription.sdp}})
        logger.info("matrix: answered %s call from %s", "video" if call.video else "audio", call.caller_id)

    async def _add_candidates(self, call: MatrixCall, candidates: list) -> None:
        if not call.pc or not candidates:
            return
        from aiortc.sdp import candidate_from_sdp
        for c in candidates:
            sdp = (c.get("candidate") or "").strip()
            if not sdp:
                continue  # end-of-candidates marker
            try:
                cand = candidate_from_sdp(sdp.split(":", 1)[1] if sdp.startswith("candidate:") else sdp)
                cand.sdpMid = c.get("sdpMid")
                cand.sdpMLineIndex = c.get("sdpMLineIndex")
                await call.pc.addIceCandidate(cand)
            except Exception as e:  # noqa: BLE001
                logger.debug("bad candidate: %s", e)

    async def _consume(self, track, session, call: MatrixCall) -> None:
        try:
            if track.kind == "audio":
                from av.audio.resampler import AudioResampler
                rs = AudioResampler(format="s16", layout="mono", rate=TG_RATE)
                while not call._ended:
                    frame = await track.recv()
                    for out in rs.resample(frame):
                        session.on_audio_in(out.to_ndarray().tobytes(), TG_RATE, 1)
            elif track.kind == "video":
                last = 0.0
                while not call._ended:
                    frame = await track.recv()
                    now = time.time()
                    if now - last >= 0.5:        # keep ~2 fps for the vision model
                        session.latest_frame = frame.to_ndarray(format="rgb24")
                        last = now
        except MediaStreamError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.debug("track %s ended: %s", track.kind, e)

    async def end_call(self, call: MatrixCall) -> None:
        if call.pc:
            try:
                await call.pc.close()
            except Exception:  # noqa: BLE001
                pass
        if self.active is call:
            self.active = None
        await self.on_ended(call)
