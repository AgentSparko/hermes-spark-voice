from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PATH = Path(os.environ.get("SPARK_CALLD_CONFIG", Path.home() / ".hermes-spark-voice" / "calld.yaml"))

DEFAULT_INSTRUCTIONS = (
    "You are Hermes, a personal assistant on a live phone call. Speak naturally and briefly, like a "
    "person: one to three sentences per turn, no lists, no markdown. You hear the caller through the "
    "call. For anything that needs facts, files, the web, the calendar, messages, or any real action, "
    "call the ask_hermes tool with a clear request and then tell the caller what came back. Use the "
    "look tool when the caller shows you something on video. Use hang_up only when the caller says "
    "goodbye or asks you to end the call."
)


@dataclass
class TelegramCfg:
    api_id: int = 0
    api_hash: str = ""
    session: str = str(Path.home() / ".hermes-spark-voice" / "telegram")
    allowed_users: list[int] = field(default_factory=list)   # Telegram user ids allowed to call
    allow_contacts: bool = False                               # also answer anyone in the account's contacts
    greeting: str = "Hi, Hermes here. How can I help?"
    busy_text: str = "Sorry, this line is not open to you."


@dataclass
class AudioCfg:
    url: str = "http://127.0.0.1:8181"
    voice: str = "female"
    speaker_ref: str = ""
    language: str = "English"
    instructions: str = DEFAULT_INSTRUCTIONS
    output_modalities: list[str] = field(default_factory=lambda: ["text", "audio"])
    max_output_tokens: int = 600
    # turn taking
    vad_threshold: float = 0.012       # RMS (0..1) above which a 20 ms frame counts as speech
    vad_start_ms: int = 200            # speech needed to open a user turn
    vad_end_ms: int = 700              # silence that closes it
    max_turn_s: float = 45.0           # commit anyway after this much speech
    barge_in: bool = True              # caller speaking cancels our reply
    filler_text: str = "One moment."   # said while a long tool call runs ("" to disable)
    filler_after_s: float = 1.5


@dataclass
class HermesCfg:
    url: str = "http://127.0.0.1:8642/v1"
    key: str = ""                       # empty = read API_SERVER_KEY from ~/.hermes/.env
    model: str = "hermes-agent"
    timeout_s: float = 240.0
    concierge_url: str = "http://127.0.0.1:8183"   # LFM3-1.7B text server for briefs/summaries (falls back to audio.url)


@dataclass
class OmniCfg:
    url: str = "http://127.0.0.1:8182"
    enabled: bool = True
    prompt: str = "Describe what you see in one or two spoken sentences."


@dataclass
class VideoCfg:
    avatar: bool = True                 # send a simple generated picture on video calls
    width: int = 640
    height: int = 360
    fps: int = 15
    pixel_format: str = "yuv420p"       # format of raw frames in/out of ntgcalls


@dataclass
class ControlCfg:
    host: str = "127.0.0.1"
    port: int = 8797
    transcript_lines: int = 40


@dataclass
class Config:
    telegram: TelegramCfg = field(default_factory=TelegramCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
    hermes: HermesCfg = field(default_factory=HermesCfg)
    omni: OmniCfg = field(default_factory=OmniCfg)
    video: VideoCfg = field(default_factory=VideoCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = Path(path or DEFAULT_PATH)
        raw = yaml.safe_load(path.read_text()) if path.is_file() else {}
        return cls.from_dict(raw or {})

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        cfg = cls()
        for section, klass in (("telegram", TelegramCfg), ("audio", AudioCfg), ("hermes", HermesCfg),
                               ("omni", OmniCfg), ("video", VideoCfg), ("control", ControlCfg)):
            data = raw.get(section) or {}
            known = {k: v for k, v in data.items() if k in klass.__dataclass_fields__}
            setattr(cfg, section, klass(**known))
        cfg.log_level = str(raw.get("log_level", cfg.log_level))
        cfg.telegram.session = os.path.expanduser(cfg.telegram.session)
        cfg.audio.speaker_ref = os.path.expanduser(cfg.audio.speaker_ref or "")
        if not cfg.hermes.key:
            cfg.hermes.key = read_hermes_env("API_SERVER_KEY") or ""
        env_id = os.environ.get("TELEGRAM_API_ID")
        env_hash = os.environ.get("TELEGRAM_API_HASH")
        if env_id:
            cfg.telegram.api_id = int(env_id)
        if env_hash:
            cfg.telegram.api_hash = env_hash
        return cfg


def read_hermes_env(key: str) -> str | None:
    env = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / ".env"
    if not env.is_file():
        return None
    for line in env.read_text().splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None
