"""Tools the Hermes agent can call: call daemon control and model status (all loopback HTTP)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .lfm3_audio import server_alive

_SCHEMAS = {
    "voice_call_status": {
        "name": "voice_call_status",
        "description": "State of the Telegram call daemon (spark-calld): logged-in account, whether a call is "
                       "active, with whom, for how long, and the last few things said on the call.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "voice_call_hangup": {
        "name": "voice_call_hangup",
        "description": "End the active Telegram audio/video call.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "voice_call_say": {
        "name": "voice_call_say",
        "description": "Speak a sentence into the active Telegram call (LFM3-Audio voice). Use it to report the "
                       "result of a longer task to the person on the call.",
        "parameters": {"type": "object",
                       "properties": {"text": {"type": "string", "description": "What to say (one or two sentences)."}},
                       "required": ["text"]},
    },
    "spark_voice_models": {
        "name": "spark_voice_models",
        "description": "Which local LFM3 llama-servers are up (audio, omni, brain) and what they report in /v1/models.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def schemas() -> dict[str, dict]:
    return _SCHEMAS


def _call(url: str, method: str = "GET", payload: dict | None = None, timeout: int = 20) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - loopback
            return json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"ok": False, "error": f"call daemon not reachable at {url}: {e}"}


def make_handlers(settings) -> dict[str, Any]:
    calld = settings.calld_url.rstrip("/")

    def status(args: dict, **_kw) -> str:
        return json.dumps(_call(calld + "/status"), ensure_ascii=False)

    def hangup(args: dict, **_kw) -> str:
        return json.dumps(_call(calld + "/hangup", "POST", {}), ensure_ascii=False)

    def say(args: dict, **_kw) -> str:
        text = str(args.get("text") or "").strip()
        if not text:
            return json.dumps({"ok": False, "error": "text is required"})
        return json.dumps(_call(calld + "/say", "POST", {"text": text}, timeout=120), ensure_ascii=False)

    def models(args: dict, **_kw) -> str:
        out = {}
        for label, url in (("audio", settings.audio_url), ("omni", settings.omni_url), ("brain", settings.brain_url)):
            if not url:
                continue
            info: dict[str, Any] = {"url": url, "up": server_alive(url)}
            if info["up"]:
                try:
                    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=5) as resp:  # noqa: S310
                        data = json.loads(resp.read().decode())
                    info["models"] = [{"id": m.get("id"), "capabilities": (m.get("meta") or {}).get("capabilities")
                                       or m.get("capabilities")} for m in data.get("data", [])]
                except Exception as e:  # noqa: BLE001
                    info["models_error"] = str(e)
            out[label] = info
        out["calld"] = _call(calld + "/status", timeout=5)
        return json.dumps(out, ensure_ascii=False)

    return {"voice_call_status": status, "voice_call_hangup": hangup, "voice_call_say": say,
            "spark_voice_models": models}
