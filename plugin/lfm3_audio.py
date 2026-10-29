"""LFM3-Audio-2B over llama-server: speech-to-text and text-to-speech providers for Hermes.

Everything goes through two llama-server endpoints:

* ``POST /v1/chat/completions`` with an ``input_audio`` part and the model card's ASR prompt
  (``Perform ASR in <language>.``) for transcription.
* ``POST /tts`` for synthesis (``voice`` female/male or ``speaker_ref_b64`` for cloning).

The providers never raise out of ``transcribe``/``synthesize``: Hermes expects a result dict or a
clear error so the gateway can fall back to text.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Optional

from agent.transcription_provider import TranscriptionProvider
from agent.tts_provider import TTSProvider

from .concierge import Concierge

logger = logging.getLogger(__name__)

PROVIDER_NAME = "lfm3-audio"
DEFAULT_URL = "http://127.0.0.1:8181"
ASR_PROMPT = "Perform ASR in {language}."
TTS_LANGS = {"english": "en", "german": "de", "french": "fr", "spanish": "es", "italian": "it",
             "portuguese": "pt", "japanese": "ja", "korean": "ko", "chinese": "zh", "arabic": "ar",
             "finnish": "fi", "swedish": "sv", "dutch": "nl", "polish": "pl", "russian": "ru"}
_LLAMA_AUDIO_FORMATS = {".wav", ".mp3", ".flac"}


class Settings:
    """Plugin settings with env overrides (``SPARK_VOICE_AUDIO_URL`` etc.) so the providers also work
    from a bare ``config.yaml`` without the plugin's ``config_schema`` values."""

    def __init__(self, get_config=None):
        def pick(key: str, default):
            env = os.environ.get("SPARK_VOICE_" + key.upper())
            if env not in (None, ""):
                return type(default)(env) if default is not None and not isinstance(default, str) else env
            if get_config is not None:
                try:
                    value = get_config(key, default)
                except Exception:  # pragma: no cover - defensive, config layer failures
                    value = default
                return default if value in (None, "") else value
            return default

        self.audio_url = str(pick("audio_url", DEFAULT_URL)).rstrip("/")
        self.voice = str(pick("voice", "female"))
        self.speaker_ref = os.path.expanduser(str(pick("speaker_ref", "") or ""))
        self.language = str(pick("language", "English"))
        self.timeout = int(pick("timeout", 180))
        self.brain_url = str(pick("brain_url", "http://127.0.0.1:8183") or "").rstrip("/")
        self.front_desk = str(pick("front_desk", "on")).lower() not in {"off", "false", "0", "no"}
        self.summary_threshold = int(pick("summary_threshold", 350))

    def concierge(self) -> Concierge:
        return Concierge([self.brain_url, self.audio_url], language=self.language, timeout=min(self.timeout, 90))


def _post_json(url: str, payload: dict, timeout: int) -> tuple[bytes, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - loopback llama-server
        return resp.read(), resp.headers.get("Content-Type", "")


def _get_json(url: str, timeout: int = 5) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def server_alive(base_url: str, timeout: int = 3) -> bool:
    try:
        health = _get_json(base_url.rstrip("/") + "/health", timeout=timeout)
    except Exception:
        return False
    return str(health.get("status", "")).lower() in {"ok", "no slot available"}


def _to_llama_audio(path: str) -> tuple[str, Optional[str]]:
    """llama-server decodes mp3/wav/flac; Telegram voice notes are ogg/opus so convert with ffmpeg.
    Returns (path_to_send, temp_path_to_delete)."""
    suffix = Path(path).suffix.lower()
    if suffix in _LLAMA_AUDIO_FORMATS:
        return path, None
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(f"ffmpeg is needed to convert {suffix or 'this'} audio for LFM3-Audio")
    fd, out = tempfile.mkstemp(suffix=".wav", prefix="lfm3-asr-")
    os.close(fd)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "16000", out],
                   check=True, timeout=120)
    return out, out


class LFM3AudioSTT(TranscriptionProvider):
    """``stt.provider: lfm3-audio`` - transcribe a file with LFM3-Audio-2B on llama-server."""

    def __init__(self, settings: Settings):
        self._s = settings

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        return True  # cheap: never probe the network here (Hermes calls this on every resolve)

    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": "LFM3-Audio-2B", "display": "LFM3-Audio-2B (llama-server)"}]

    def transcribe(self, file_path: str, *, model: Optional[str] = None,
                   language: Optional[str] = None, **extra: Any) -> dict[str, Any]:
        tmp = None
        try:
            send_path, tmp = _to_llama_audio(file_path)
            with open(send_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
            lang = _language_name(language) or self._s.language
            payload = {
                "model": model or "LFM3-Audio-2B",
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": ASR_PROMPT.format(language=lang)},
                    {"role": "user", "content": [{"type": "input_audio",
                                                  "input_audio": {"data": b64, "format": "wav"}}]},
                ],
            }
            raw, _ = _post_json(self._s.audio_url + "/v1/chat/completions", payload, self._s.timeout)
            body = json.loads(raw.decode("utf-8"))
            text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            text = _strip_think(text).strip()
            if not text:
                return {"success": False, "error": "LFM3-Audio returned an empty transcript", "provider": self.name}
            if self._s.front_desk:
                text = self._s.concierge().brief(text)
            return {"success": True, "transcript": text, "provider": self.name, "language": lang}
        except urllib.error.URLError as e:
            return {"success": False, "provider": self.name,
                    "error": f"LFM3-Audio server unreachable at {self._s.audio_url}: {e.reason}"}
        except Exception as e:  # noqa: BLE001 - provider contract: never raise
            logger.warning("lfm3-audio transcription failed: %s", e)
            return {"success": False, "provider": self.name, "error": str(e)}
        finally:
            if tmp:
                with _suppress():
                    os.unlink(tmp)


class LFM3AudioTTS(TTSProvider):
    """``tts.provider: lfm3-audio`` - synthesize with LFM3-Audio-2B via llama-server ``/tts``."""

    def __init__(self, settings: Settings):
        self._s = settings

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def voice_compatible(self) -> bool:
        return True  # the gateway turns our wav into an opus voice bubble on Telegram

    def is_available(self) -> bool:
        return True

    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": "LFM3-Audio-2B", "display": "LFM3-Audio-2B (llama-server)"}]

    def list_voices(self) -> list[dict[str, Any]]:
        voices = [{"id": "female", "display": "Female (LFM3-Audio)", "gender": "female"},
                  {"id": "male", "display": "Male (LFM3-Audio)", "gender": "male"}]
        if self._s.speaker_ref and os.path.isfile(self._s.speaker_ref):
            voices.insert(0, {"id": "clone", "display": f"Clone of {Path(self._s.speaker_ref).name}"})
        return voices

    def default_voice(self) -> Optional[str]:
        return "clone" if self._s.speaker_ref and os.path.isfile(self._s.speaker_ref) else self._s.voice

    def _tts_payload(self, text: str, voice: Optional[str], language: Optional[str]) -> dict:
        payload: dict[str, Any] = {"input": text, "response_format": "wav", "repeat_penalty": 1.05}
        lang = TTS_LANGS.get((language or self._s.language).lower())
        if lang:
            payload["lang"] = lang
        chosen = (voice or self.default_voice() or "female").lower()
        ref = self._s.speaker_ref
        if chosen == "clone" or (os.path.isfile(chosen) and chosen.endswith(".wav")):
            ref = chosen if os.path.isfile(chosen) else ref
            if ref and os.path.isfile(ref):
                with open(ref, "rb") as fh:
                    payload["speaker_ref_b64"] = base64.b64encode(fh.read()).decode("ascii")
                return payload
            chosen = self._s.voice
        payload["voice"] = "male" if chosen.startswith("m") else "female"
        return payload

    def synthesize(self, text: str, output_path: str, *, voice: Optional[str] = None,
                   model: Optional[str] = None, speed: Optional[float] = None,
                   format: str = "mp3", **extra: Any) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("nothing to say")
        if self._s.front_desk and not extra.get("verbatim"):
            text = self._s.concierge().spoken_summary(text, threshold_chars=self._s.summary_threshold) or text
        payload = self._tts_payload(text, voice, extra.get("language"))
        try:
            wav, _ = _post_json(self._s.audio_url + "/tts", payload, self._s.timeout)
        except urllib.error.URLError as e:
            raise RuntimeError(f"LFM3-Audio server unreachable at {self._s.audio_url}: {e.reason}") from e
        if not wav or len(wav) < 64:
            raise RuntimeError("LFM3-Audio returned no audio")
        fmt = (format or "wav").lower()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "wav" or not shutil.which("ffmpeg"):
            if fmt != "wav":
                logger.warning("ffmpeg missing: writing wav instead of %s", fmt)
            out.write_bytes(wav)
            return str(out)
        fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="lfm3-tts-")
        os.close(fd)
        try:
            Path(tmp).write_bytes(wav)
            args = ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp]
            if speed and abs(speed - 1.0) > 0.01:
                args += ["-filter:a", f"atempo={max(0.5, min(2.0, speed)):.2f}"]
            if fmt in {"opus", "ogg"}:
                args += ["-c:a", "libopus", "-b:a", "48k", "-application", "voip"]
            args.append(str(out))
            subprocess.run(args, check=True, timeout=120)
        finally:
            with _suppress():
                os.unlink(tmp)
        return str(out)

    def stream(self, text: str, *, voice: Optional[str] = None, model: Optional[str] = None,
               format: str = "opus", **extra: Any) -> Iterator[bytes]:
        suffix = ".ogg" if format in {"opus", "ogg"} else f".{format}"
        fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="lfm3-stream-")
        os.close(fd)
        try:
            self.synthesize(text, tmp, voice=voice, model=model, format=format, **extra)
            with open(tmp, "rb") as fh:
                while chunk := fh.read(16384):
                    yield chunk
        finally:
            with _suppress():
                os.unlink(tmp)


def _language_name(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    code = code.strip().lower()
    for name, short in TTS_LANGS.items():
        if code in (short, name):
            return name.capitalize()
    return code.capitalize() if len(code) > 2 else None


def _strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return text


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True
