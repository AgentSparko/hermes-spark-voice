"""spark-voice: LFM3-Audio speech for Hermes on a DGX Spark (standalone plugin).

Registers the ``lfm3-audio`` STT and TTS providers, the call-daemon tools, the ``hermes spark-voice``
CLI and the ``spark-voice:telegram-voice`` skill. Nothing here touches Hermes core.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .lfm3_audio import DEFAULT_URL, LFM3AudioSTT, LFM3AudioTTS, Settings, server_alive
from .tools import make_handlers, schemas

logger = logging.getLogger(__name__)
_HERE = Path(__file__).resolve().parent


class _PluginSettings(Settings):
    def __init__(self, get_config):
        super().__init__(get_config)
        self.omni_url = str(os.environ.get("SPARK_VOICE_OMNI_URL") or get_config("omni_url", "http://127.0.0.1:8182") or "").rstrip("/")
        self.brain_url = str(os.environ.get("SPARK_VOICE_BRAIN_URL") or get_config("brain_url", "http://127.0.0.1:8183") or "").rstrip("/")
        self.calld_url = str(os.environ.get("SPARK_VOICE_CALLD_URL") or get_config("calld_url", "http://127.0.0.1:8797") or "").rstrip("/")


def _cli_setup(parser) -> None:
    sub = parser.add_subparsers(dest="spark_voice_cmd")
    sub.add_parser("status", help="Show which LFM3 servers and the call daemon are up")
    say = sub.add_parser("say", help="Synthesize text with LFM3-Audio and write a wav")
    say.add_argument("text")
    say.add_argument("--out", default="spark-voice.wav")
    say.add_argument("--voice", default=None)
    asr = sub.add_parser("transcribe", help="Transcribe an audio file with LFM3-Audio")
    asr.add_argument("file")


def _cli_handler_factory(settings):
    def run(args) -> int:
        cmd = getattr(args, "spark_voice_cmd", None) or "status"
        if cmd == "status":
            for label, url in (("audio", settings.audio_url), ("omni", settings.omni_url), ("brain", settings.brain_url)):
                print(f"{label:6s} {url:32s} {'up' if server_alive(url) else 'down'}")
            print("calld ", make_handlers(settings)["voice_call_status"]({}))
            return 0
        if cmd == "say":
            path = LFM3AudioTTS(settings).synthesize(args.text, args.out, voice=args.voice, format="wav")
            print(path)
            return 0
        if cmd == "transcribe":
            result = LFM3AudioSTT(settings).transcribe(args.file)
            print(result.get("transcript") if result.get("success") else f"error: {result.get('error')}")
            return 0 if result.get("success") else 1
        return 0
    return run


def register(ctx) -> None:
    settings = _PluginSettings(ctx.get_config)
    ctx.register_transcription_provider(LFM3AudioSTT(settings))
    ctx.register_tts_provider(LFM3AudioTTS(settings))

    handlers = make_handlers(settings)
    emojis = {"voice_call_status": "📞", "voice_call_hangup": "👋", "voice_call_say": "🗣️", "spark_voice_models": "🧠"}
    for name, schema in schemas().items():
        ctx.register_tool(name=name, toolset="spark_voice", schema=schema, handler=handlers[name],
                          emoji=emojis.get(name, ""))

    ctx.register_cli_command(name="spark-voice", help="LFM3-Audio voice stack on the Spark (status, say, transcribe)",
                             setup_fn=_cli_setup, handler_fn=_cli_handler_factory(settings))

    skill = _HERE / "skills" / "spark-voice" / "SKILL.md"
    if skill.is_file():
        ctx.register_skill("telegram-voice", skill, description="Voice on Telegram with local LFM3-Audio.")

    def _on_session_end(**_kw) -> None:
        return None  # calls are owned by the daemon; nothing to clean up in-process

    ctx.register_hook("on_session_end", _on_session_end)
    logger.info("spark-voice: lfm3-audio providers registered (audio=%s)", settings.audio_url or DEFAULT_URL)
