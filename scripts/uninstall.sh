#!/usr/bin/env bash
set -euo pipefail
PREFIX="${SPARK_VOICE_HOME:-$HOME/.hermes-spark-voice}"
systemctl --user disable --now lfm3-audio lfm3-omni lfm3-brain spark-calld 2>/dev/null || true
rm -f "$HOME/.config/systemd/user/"{lfm3-audio,lfm3-omni,lfm3-brain,spark-calld}.service
systemctl --user daemon-reload 2>/dev/null || true
rm -rf "$HOME/.hermes/plugins/spark-voice"
if [[ "${KEEP_MODELS:-}" == 1 ]]; then
  rm -rf "${PREFIX:?}/venv" "${PREFIX:?}/src" "${PREFIX:?}/bin"
else
  rm -rf "${PREFIX:?}"
fi
echo "removed. Hermes config.yaml still points stt/tts at lfm3-audio: restore a ~/.hermes/config.yaml.bak-* copy or set the providers back."
