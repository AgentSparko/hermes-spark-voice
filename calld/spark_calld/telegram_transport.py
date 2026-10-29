"""Telegram p2p transport (optional): a *user* account (kurigram + py-tgcalls) answers private calls.

Bot accounts cannot take Telegram calls, so this needs a second phone number logged in as the agent.
Audio in/out is raw PCM16 48 kHz through ntgcalls' external media; video frames are yuv420p.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import numpy as np

from .audio import TG_RATE, i420_to_rgb, rgb_to_i420
from .config import Config

logger = logging.getLogger(__name__)

try:
    from pyrogram import Client
    from pytgcalls import PyTgCalls, filters as fl
    from pytgcalls.types import ChatUpdate, Device, ExternalMedia, MediaStream, RecordStream, StreamFrames
    from pytgcalls.types.raw import AudioParameters, VideoParameters
    HAVE_TG = True
except Exception:  # noqa: BLE001
    HAVE_TG = False

CHUNK_MS = 10


class TelegramCall:
    def __init__(self, transport: "TelegramTransport", chat_id: int, caller_name: str):
        self.transport = transport
        self.chat_id = str(chat_id)
        self.caller_id = str(chat_id)
        self.caller_name = caller_name
        self.session = None
        self.ended = False

    async def hangup(self) -> None:
        if self.ended:
            return
        self.ended = True
        try:
            await self.transport.calls.leave_call(int(self.chat_id))
        except Exception as e:  # noqa: BLE001
            logger.debug("leave_call: %s", e)
        await self.transport.end_call(self)

    async def post_text(self, text: str) -> None:
        try:
            await self.transport.app.send_message(int(self.chat_id), text[:4000])
        except Exception as e:  # noqa: BLE001
            logger.debug("send_message failed: %s", e)


class TelegramTransport:
    def __init__(self, cfg: Config, on_incoming, on_ended):
        self.cfg, self.on_incoming, self.on_ended = cfg, on_incoming, on_ended
        self.app = None
        self.calls = None
        self.active: Optional[TelegramCall] = None
        self._pump: Optional[asyncio.Task] = None

    def available(self) -> tuple[bool, str]:
        if not HAVE_TG:
            return False, "telegram calls need py-tgcalls and kurigram in the venv"
        t = self.cfg.telegram
        if not (t.api_id and t.api_hash):
            return False, "telegram.api_id / api_hash not set (my.telegram.org)"
        return True, "ok"

    async def start(self) -> None:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(why)
        t = self.cfg.telegram
        self.app = Client(t.session, api_id=t.api_id, api_hash=t.api_hash)
        self.calls = PyTgCalls(self.app)

        @self.calls.on_update(fl.chat_update(ChatUpdate.Status.INCOMING_CALL))
        async def _incoming(_, update: ChatUpdate):
            await self._handle_incoming(update.chat_id)

        @self.calls.on_update(fl.stream_frame(devices=Device.SPEAKER | Device.CAMERA))
        async def _frames(_, update: StreamFrames):
            call = self.active
            if not call or call.ended or str(update.chat_id) != call.chat_id or call.session is None:
                return
            for f in update.frames:
                if update.device == Device.SPEAKER:
                    call.session.on_audio_in(f.frame, TG_RATE, 1)
                elif update.device == Device.CAMERA and f.info.width and f.info.height:
                    try:
                        call.session.latest_frame = i420_to_rgb(f.frame, f.info.width, f.info.height)
                    except ValueError:
                        pass

        @self.calls.on_update(fl.chat_update(ChatUpdate.Status.LEFT_CALL | ChatUpdate.Status.DISCARDED_CALL))
        async def _left(_, update: ChatUpdate):
            if self.active and str(update.chat_id) == self.active.chat_id and not self.active.ended:
                self.active.ended = True
                await self.end_call(self.active)

        await self.calls.start()
        me = await self.app.get_me()
        logger.info("telegram: logged in as %s (%s), waiting for calls", me.first_name, me.id)

    async def stop(self) -> None:
        if self.active:
            await self.active.hangup()
        if self.app:
            try:
                await self.app.stop()
            except Exception:  # noqa: BLE001
                pass

    def _allowed(self, user_id: int, is_contact: bool) -> bool:
        t = self.cfg.telegram
        return user_id in t.allowed_users or (t.allow_contacts and is_contact)

    async def _handle_incoming(self, chat_id: int) -> None:
        try:
            user = await self.app.get_users(chat_id)
            name = " ".join(x for x in (user.first_name, user.last_name) if x) or str(chat_id)
            is_contact = bool(getattr(user, "is_contact", False))
        except Exception:  # noqa: BLE001
            name, is_contact = str(chat_id), False
        if not self._allowed(chat_id, is_contact):
            logger.info("telegram: call from %s (%s) refused", name, chat_id)
            try:
                await self.calls.leave_call(chat_id)
                await self.app.send_message(chat_id, self.cfg.telegram.busy_text)
            except Exception:  # noqa: BLE001
                pass
            return
        if self.active and not self.active.ended:
            await self.calls.leave_call(chat_id)
            return
        call = TelegramCall(self, chat_id, name)
        self.active = call
        session = await self.on_incoming(call)
        call.session = session
        v = self.cfg.video
        media = ExternalMedia.AUDIO | ExternalMedia.VIDEO if v.avatar else ExternalMedia.AUDIO
        audio_params = AudioParameters(bitrate=TG_RATE, channels=1)
        video_params = VideoParameters(width=v.width, height=v.height, frame_rate=v.fps) if v.avatar else None
        await self.calls.play(chat_id, MediaStream(media, audio_params, video_params) if video_params else MediaStream(media, audio_params))
        await self.calls.record(chat_id, RecordStream(audio=True, audio_parameters=audio_params, camera=True))
        self._pump = asyncio.create_task(self._pump_out(call), name="tg-pump")

    async def _pump_out(self, call: TelegramCall) -> None:
        nbytes = TG_RATE * CHUNK_MS // 1000 * 2
        next_at = time.monotonic()
        next_video = next_at
        avatar = getattr(call.session, "avatar", None) if self.cfg.video.avatar else None
        try:
            while not call.ended:
                pcm = call.session.next_audio_out(nbytes)
                await self.calls.send_frame(int(call.chat_id), Device.MICROPHONE, pcm)
                now = time.monotonic()
                if avatar is not None and now >= next_video:
                    await self.calls.send_frame(int(call.chat_id), Device.CAMERA, avatar.frame_i420())
                    next_video = now + 1.0 / self.cfg.video.fps
                next_at += CHUNK_MS / 1000
                await asyncio.sleep(max(0.0, next_at - time.monotonic()))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.info("telegram: outgoing pump ended: %s", e)

    async def end_call(self, call: TelegramCall) -> None:
        if self._pump:
            self._pump.cancel()
        if self.active is call:
            self.active = None
        await self.on_ended(call)
