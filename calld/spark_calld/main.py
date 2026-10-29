"""spark-calld entry point: ``spark-calld run | login | doctor | config``."""
from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import signal
import sys
from pathlib import Path

import httpx
import yaml

from . import __version__
from .audio import f32_to_pcm16, resample
from .avatar import Avatar
from .bridge import CallSession
from .config import DEFAULT_PATH, Config
from .control import serve
from .hermes_client import HermesClient
from .concierge import Concierge

logger = logging.getLogger("spark_calld")


class Daemon:
    def __init__(self, cfg: Config, transports: list[str]):
        self.cfg = cfg
        self.transport_names = transports
        self.transports: list = []
        self.session: CallSession | None = None
        self.last_status: dict = {"ok": True, "active": False}
        self._stop = asyncio.Event()

    # ---- call plumbing shared by all transports ----
    async def on_incoming(self, call) -> CallSession:
        if self.session and not self.session.closed:
            await self.session.close()
        avatar = Avatar(self.cfg.video.width, self.cfg.video.height) if self.cfg.video.avatar else None

        def on_state(state: str, level: float) -> None:
            if avatar is not None:
                avatar.state, avatar.level = state, level

        concierge = Concierge([self.cfg.hermes.concierge_url, self.cfg.audio.url], language=self.cfg.audio.language)
        hermes = HermesClient(self.cfg.hermes.url, self.cfg.hermes.key, self.cfg.hermes.model, self.cfg.hermes.timeout_s, concierge=concierge)
        session = CallSession(self.cfg, call, hermes=hermes, tts_filler=self.tts_pcm, on_state=on_state)
        session.avatar = avatar  # type: ignore[attr-defined]
        self.session = session
        await session.start()
        logger.info("call started with %s (%s)", call.caller_name, call.caller_id)
        return session

    async def on_ended(self, call) -> None:
        if self.session is not None:
            self.last_status = self.session.status() | {"active": False}
            await self.session.close()
            logger.info("call with %s ended after %ss", call.caller_name, self.last_status.get("seconds"))
            self.session = None

    async def tts_pcm(self, text: str) -> bytes:
        """Short filler phrases via /tts, returned as 48 kHz mono PCM16."""
        a = self.cfg.audio
        payload = {"input": text, "response_format": "pcm", "voice": "male" if a.voice.startswith("m") else "female"}
        ref = Path(a.speaker_ref) if a.speaker_ref else None
        if ref and ref.is_file():
            payload.pop("voice", None)
            payload["speaker_ref_b64"] = base64.b64encode(ref.read_bytes()).decode("ascii")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(a.url.rstrip("/") + "/tts", json=payload)
        resp.raise_for_status()
        rate = 24000
        for part in resp.headers.get("content-type", "").split(";"):
            if part.strip().startswith("rate="):
                rate = int(part.strip()[5:])
        import numpy as np
        samples = np.frombuffer(resp.content, dtype="<f4")
        return f32_to_pcm16(resample(samples, rate, 48000))

    # ---- control API ----
    async def _status(self, _body: dict) -> dict:
        if self.session and not self.session.closed:
            return self.session.status() | {"transports": self.transport_names}
        return self.last_status | {"active": False, "transports": self.transport_names}

    async def _hangup(self, _body: dict) -> dict:
        if self.session and not self.session.closed:
            await self.session.t.hangup()
            return {"ok": True, "ended": True}
        return {"ok": True, "ended": False, "note": "no active call"}

    async def _say(self, body: dict) -> dict:
        text = str(body.get("text") or "").strip()
        if not text:
            return {"ok": False, "error": "text is required"}
        if not (self.session and not self.session.closed):
            return {"ok": False, "error": "no active call"}
        await self.session.say(text)
        return {"ok": True}

    async def run(self) -> int:
        from .matrix_transport import MatrixTransport
        from .telegram_transport import TelegramTransport
        factories = {"matrix": lambda: MatrixTransport(self.cfg, self.on_incoming, self.on_ended,
                                                       allowed_users=[str(u) for u in self.cfg.telegram.allowed_users if isinstance(u, str)]),
                     "telegram": lambda: TelegramTransport(self.cfg, self.on_incoming, self.on_ended)}
        started = 0
        for name in self.transport_names:
            t = factories[name]()
            ok, why = t.available()
            if not ok:
                logger.warning("%s transport disabled: %s", name, why)
                continue
            try:
                await t.start()
                self.transports.append(t)
                started += 1
            except Exception as e:  # noqa: BLE001
                logger.error("%s transport failed to start: %s", name, e)
        if not started:
            logger.error("no call transport could start; check `spark-calld doctor`")
            return 2
        server = await serve(self.cfg.control.host, self.cfg.control.port,
                             {("GET", "/status"): self._status, ("POST", "/hangup"): self._hangup, ("POST", "/say"): self._say})
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:
                pass
        await self._stop.wait()
        for t in self.transports:
            await t.stop()
        server.close()
        return 0


def cmd_doctor(cfg: Config, transports: list[str]) -> int:
    import json
    import urllib.request
    ok_all = True

    def check(label: str, url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:  # noqa: S310
                body = r.read().decode()
            print(f"  ok    {label:14s} {url}")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {label:14s} {url}  ({e})")
            return False

    print("llama-server endpoints:")
    ok_all &= check("lfm3-audio", cfg.audio.url.rstrip("/") + "/health")
    if cfg.omni.enabled:
        check("lfm3-omni", cfg.omni.url.rstrip("/") + "/health")
    check("concierge", cfg.hermes.concierge_url.rstrip("/") + "/health")
    print("hermes api server:")
    try:
        req = urllib.request.Request(cfg.hermes.url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {cfg.hermes.key}"})
        with urllib.request.urlopen(req, timeout=3) as r:  # noqa: S310
            print(f"  ok    hermes         {cfg.hermes.url} ({len(json.loads(r.read()).get('data', []))} models)")
    except Exception as e:  # noqa: BLE001
        ok_all = False
        print(f"  FAIL  hermes         {cfg.hermes.url}  ({e}) -> API_SERVER_ENABLED=true in ~/.hermes/.env and `hermes gateway` running?")
    from .matrix_transport import MatrixTransport
    from .telegram_transport import TelegramTransport
    print("transports:")
    for name in transports:
        t = MatrixTransport(cfg, None, None) if name == "matrix" else TelegramTransport(cfg, None, None)
        ok, why = t.available()
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:14s} {why}")
    return 0 if ok_all else 1


def cmd_login(cfg: Config) -> int:
    try:
        from pyrogram import Client
    except ImportError:
        print("kurigram is not installed in this venv", file=sys.stderr)
        return 2
    t = cfg.telegram
    if not (t.api_id and t.api_hash):
        print("Set telegram.api_id and telegram.api_hash in", DEFAULT_PATH, "(from https://my.telegram.org)", file=sys.stderr)
        return 2
    Path(t.session).parent.mkdir(parents=True, exist_ok=True)
    app = Client(t.session, api_id=t.api_id, api_hash=t.api_hash)
    with app:
        me = app.get_me()
        print(f"logged in as {me.first_name} (@{me.username}) id={me.id}; session stored at {t.session}.session")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="spark-calld", description="Telegram/Matrix call daemon for Hermes + LFM3-Audio")
    ap.add_argument("--config", default=None, help=f"config file (default {DEFAULT_PATH})")
    ap.add_argument("--transport", action="append", choices=["matrix", "telegram"], help="repeatable; default = both")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("run", help="run the daemon")
    sub.add_parser("login", help="interactive Telegram user-account login (second phone number)")
    sub.add_parser("doctor", help="check llama-servers, Hermes API and transports")
    sub.add_parser("config", help="print the effective configuration (secrets masked)")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    transports = args.transport or ["matrix", "telegram"]
    if args.cmd == "login":
        return cmd_login(cfg)
    if args.cmd == "doctor":
        return cmd_doctor(cfg, transports)
    if args.cmd == "config":
        import dataclasses
        data = dataclasses.asdict(cfg)
        data["telegram"]["api_hash"] = "***" if data["telegram"]["api_hash"] else ""
        data["hermes"]["key"] = "***" if data["hermes"]["key"] else ""
        print(yaml.safe_dump(data, sort_keys=False))
        return 0
    return asyncio.run(Daemon(cfg, transports).run())


if __name__ == "__main__":
    sys.exit(main())
