#!/usr/bin/env bash
# hermes-spark-voice one-line installer
#   curl -fsSL https://raw.githubusercontent.com/AgentSparko/hermes-spark-voice/main/install.sh | bash
# Options (env): SPARK_VOICE_MODELS="audio omni brain"  BRAIN=keep|lfm3  TRANSPORTS="matrix telegram"
#                NO_SERVICES=1 (skip systemd)  SPARK_VOICE_REF=main  SPARK_VOICE_HOME=~/.hermes-spark-voice
set -euo pipefail
REPO="${SPARK_VOICE_REPO:-https://github.com/AgentSparko/hermes-spark-voice.git}"
REF="${SPARK_VOICE_REF:-main}"
PREFIX="${SPARK_VOICE_HOME:-$HOME/.hermes-spark-voice}"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
BRAIN="${BRAIN:-keep}"
TRANSPORTS="${TRANSPORTS:-matrix telegram}"
export SPARK_VOICE_HOME="$PREFIX"

log(){ printf '\033[1;32m[spark-voice]\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[spark-voice] %s\033[0m\n' "$*"; }
die(){ printf '\033[1;31m[spark-voice] %s\033[0m\n' "$*" >&2; exit 1; }

# 0) prerequisites
for bin in git curl python3 ffmpeg; do command -v "$bin" >/dev/null || die "$bin is required (sudo apt install -y git curl python3 python3-venv ffmpeg cmake build-essential libcurl4-openssl-dev)"; done
python3 - <<'PY' || die "python3 >= 3.10 is required"
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
mkdir -p "$PREFIX"

# 1) get the repo (this script may be running from curl, so clone/update ourselves)
SRC="$PREFIX/src/hermes-spark-voice"
if [[ -d "$SRC/.git" ]]; then
  # best effort: an offline or diverged cache must not stop the install, and each
  # step is attempted on its own so a failed fetch cannot skip the checkout
  git -C "$SRC" fetch -q origin || true
  git -C "$SRC" checkout -q "$REF" || true
  git -C "$SRC" pull -q --ff-only origin "$REF" || true
else
  mkdir -p "$PREFIX/src" && git clone -q --branch "$REF" "$REPO" "$SRC"
fi
log "source at $SRC ($(git -C "$SRC" rev-parse --short HEAD))"

# 2) Hermes
if ! command -v hermes >/dev/null 2>&1; then
  warn "Hermes is not installed; installing with the official script"
  curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v hermes >/dev/null || die "hermes not on PATH after install; open a new shell and re-run"

# 3) daemon venv (also gives us pyyaml + hf for the scripts)
if [[ ! -x "$PREFIX/venv/bin/python" ]]; then python3 -m venv "$PREFIX/venv"; fi
"$PREFIX/venv/bin/pip" install -q --upgrade pip wheel
extras=""
[[ " $TRANSPORTS " == *" matrix "* ]] && extras="[matrix]"
"$PREFIX/venv/bin/pip" install -q -e "$SRC/calld$extras" || { warn "optional call deps failed; retrying without matrix extras"; "$PREFIX/venv/bin/pip" install -q -e "$SRC/calld"; }
export PATH="$PREFIX/venv/bin:$PATH"
log "spark-calld $("$PREFIX/venv/bin/spark-calld" --version)"

# 4) llama.cpp + models
bash "$SRC/spark/llama.sh"
bash "$SRC/spark/models.sh"

# 5) Hermes plugin
PLUG="$HERMES_HOME/plugins/spark-voice"
mkdir -p "$HERMES_HOME/plugins"
rm -rf "$PLUG" && cp -r "$SRC/plugin" "$PLUG"
hermes plugins enable spark-voice >/dev/null 2>&1 || true
log "plugin installed at $PLUG"

# 6) config: Hermes (stt/tts/voice/api server) + env + calld.yaml
"$PREFIX/venv/bin/python" "$SRC/scripts/configure-hermes.py" --brain "$BRAIN" ${SPARK_VOICE_SPEAKER_REF:+--speaker-ref "$SPARK_VOICE_SPEAKER_REF"} ${SPARK_VOICE_LANGUAGE:+--language "$SPARK_VOICE_LANGUAGE"}
if [[ ! -f "$PREFIX/env" ]]; then sed "s|%h|$HOME|g" "$SRC/spark/env.example" > "$PREFIX/env"; fi
if [[ ! -f "$PREFIX/calld.yaml" ]]; then
  cat > "$PREFIX/calld.yaml" <<YAML
# spark-calld configuration (see README "Calls"). Matrix calls reuse Hermes' MATRIX_* from ~/.hermes/.env.
telegram:
  api_id: 0            # from https://my.telegram.org (only for Telegram p2p calls, needs a user account)
  api_hash: ""
  allowed_users: []    # Telegram numeric user ids (or @matrix:ids as strings) allowed to call
  allow_contacts: false
audio:
  url: http://127.0.0.1:8181
  voice: female
  speaker_ref: ""      # 5-15 s wav for voice cloning
  language: English
hermes:
  url: http://127.0.0.1:8642/v1
  concierge_url: http://127.0.0.1:8183
omni:
  url: http://127.0.0.1:8182
  enabled: true
video:
  avatar: true
YAML
  chmod 600 "$PREFIX/calld.yaml"
fi

# 7) services
if [[ -z "${NO_SERVICES:-}" ]] && command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  mkdir -p "$HOME/.config/systemd/user"
  cp "$SRC/spark/units/"*.service "$HOME/.config/systemd/user/"
  systemctl --user daemon-reload
  units=(lfm3-audio)
  [[ -e "$PREFIX/models/omni.gguf" ]] && units+=(lfm3-omni)
  [[ -e "$PREFIX/models/brain.gguf" ]] && units+=(lfm3-brain)
  units+=(spark-calld)
  systemctl --user enable --now "${units[@]}"
  loginctl enable-linger "$USER" 2>/dev/null || true
  log "services: ${units[*]} (systemctl --user status lfm3-audio)"
else
  warn "systemd user session not available: start by hand, e.g. $PREFIX/bin/llama-server -m $PREFIX/models/audio.gguf --mmproj $PREFIX/models/audio-mmproj.gguf --port 8181 --jinja"
fi

cat <<EOT

$(log "done.")
  1. Voice notes: restart the gateway ->  hermes gateway restart   (or: systemctl --user restart hermes-gateway)
     Send a Telegram voice note; the reply comes back as a voice bubble (LFM3 front desk + your Qwen brain).
  2. Calls over Matrix (bot account): nothing else to do if Hermes already has MATRIX_* in ~/.hermes/.env;
     call the bot from Element (audio or video).
  3. Calls over Telegram (needs a second Telegram account): edit $PREFIX/calld.yaml (api_id/api_hash/allowed_users),
     run  $PREFIX/venv/bin/spark-calld login , then  systemctl --user restart spark-calld
  4. Check:  $PREFIX/venv/bin/spark-calld doctor   and   hermes spark-voice status
EOT
