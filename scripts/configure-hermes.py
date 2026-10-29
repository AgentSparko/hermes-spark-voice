#!/usr/bin/env python3
"""Point an existing Hermes install at the local LFM3 stack. Idempotent and conservative: only the
keys listed below are written, everything else in ~/.hermes/config.yaml is kept. Pass --brain lfm3 to
make LFM3-1.7B the agent model (default keeps whatever provider you already use, e.g. Qwen).
"""
from __future__ import annotations

import argparse
import os
import secrets
import shutil
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:  # the installer pip-installs pyyaml into the daemon venv and runs us with it
    print("pyyaml missing: run with the venv python", file=sys.stderr)
    sys.exit(2)

HERMES = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def load(path: Path) -> dict:
    if path.is_file():
        return yaml.safe_load(path.read_text()) or {}
    return {}


def deep_set(cfg: dict, dotted: str, value) -> None:
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise SystemExit(f"config key {k!r} is not a mapping; refusing to overwrite")
    node[keys[-1]] = value


def env_upsert(path: Path, key: str, value: str, *, keep_existing: bool = True) -> str:
    lines = path.read_text().splitlines() if path.is_file() else []
    for i, line in enumerate(lines):
        if line.startswith(key + "="):
            if keep_existing and line.split("=", 1)[1].strip():
                return line.split("=", 1)[1].strip()
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    os.chmod(path, 0o600)
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-url", default="http://127.0.0.1:8181")
    ap.add_argument("--omni-url", default="http://127.0.0.1:8182")
    ap.add_argument("--brain-url", default="http://127.0.0.1:8183")
    ap.add_argument("--brain", choices=["keep", "lfm3"], default="keep",
                    help="keep = leave model.provider alone (Qwen); lfm3 = use LFM3-1.7B via llama-server")
    ap.add_argument("--vision", choices=["keep", "omni"], default="omni",
                    help="omni = route vision_analyze / image understanding to LFM3-Omni")
    ap.add_argument("--voice", default="female")
    ap.add_argument("--speaker-ref", default="")
    ap.add_argument("--language", default="English")
    ap.add_argument("--no-auto-tts", action="store_true")
    ap.add_argument("--api-server-port", type=int, default=8642)
    ap.add_argument("--print-api-key", action="store_true")
    args = ap.parse_args()

    cfg_path = HERMES / "config.yaml"
    cfg = load(cfg_path)
    if cfg_path.is_file():
        shutil.copy2(cfg_path, cfg_path.with_suffix(f".yaml.bak-{time.strftime('%Y%m%d%H%M%S')}"))

    deep_set(cfg, "stt.provider", "lfm3-audio")
    deep_set(cfg, "tts.provider", "lfm3-audio")
    deep_set(cfg, "voice.auto_tts", not args.no_auto_tts)
    deep_set(cfg, "voice.voice_chat_mode", "chained")
    # the whole reply must reach the provider in ONE call so the front desk can summarize it;
    # Hermes otherwise splits long replies into 4000-char chunks before synthesis
    deep_set(cfg, "tts.lfm3-audio.max_text_length", 40000)
    deep_set(cfg, "plugins.entries.spark-voice.enabled", True)
    for key, value in (("audio_url", args.audio_url), ("omni_url", args.omni_url), ("brain_url", args.brain_url),
                       ("voice", args.voice), ("speaker_ref", args.speaker_ref), ("language", args.language)):
        deep_set(cfg, f"plugins.entries.spark-voice.config.{key}", value)
    if args.vision == "omni":
        deep_set(cfg, "auxiliary.vision.provider", "custom")
        deep_set(cfg, "auxiliary.vision.base_url", args.omni_url.rstrip("/") + "/v1")
        deep_set(cfg, "auxiliary.vision.model", "LFM3-Omni-2B")
        deep_set(cfg, "auxiliary.vision.api_key", "local")
    if args.brain == "lfm3":
        deep_set(cfg, "model.provider", "custom")
        deep_set(cfg, "model.default", "LFM3-1.7B")
        deep_set(cfg, "model.base_url", args.brain_url.rstrip("/") + "/v1")
        deep_set(cfg, "model.api_key", "local")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    env = HERMES / ".env"
    env_upsert(env, "API_SERVER_ENABLED", "true", keep_existing=False)
    env_upsert(env, "API_SERVER_HOST", "127.0.0.1")
    env_upsert(env, "API_SERVER_PORT", str(args.api_server_port))
    key = env_upsert(env, "API_SERVER_KEY", secrets.token_urlsafe(24))
    if args.print_api_key:
        print(key)
    print(f"[configure] wrote {cfg_path} and enabled the Hermes API server on 127.0.0.1:{args.api_server_port}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
