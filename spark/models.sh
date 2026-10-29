#!/usr/bin/env bash
# Download the LFM3 GGUFs used by the voice stack into $PREFIX/models (resumable, idempotent).
# Set SPARK_VOICE_MODELS="audio" to skip omni/brain; AUDIO_QUANT/OMNI_QUANT/BRAIN_QUANT pick quants.
set -euo pipefail
PREFIX="${SPARK_VOICE_HOME:-$HOME/.hermes-spark-voice}"
MODELS_DIR="$PREFIX/models"
WANT="${SPARK_VOICE_MODELS:-audio omni brain}"
AUDIO_QUANT="${AUDIO_QUANT:-Q8_0}"; OMNI_QUANT="${OMNI_QUANT:-Q4_K_M}"; BRAIN_QUANT="${BRAIN_QUANT:-Q4_K_M}"
MMPROJ_QUANT="${MMPROJ_QUANT:-F16}"
mkdir -p "$MODELS_DIR"
log(){ printf '\033[1;34m[models]\033[0m %s\n' "$*"; }

# hf CLI (huggingface_hub) in the daemon venv, falling back to curl of resolve URLs
fetch(){ # repo file
  local repo="$1" file="$2" dest="$MODELS_DIR/$2"
  if [[ -s "$dest" ]]; then log "have $file"; return 0; fi
  log "downloading $repo/$file"
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo" "$file" --local-dir "$MODELS_DIR" >/dev/null
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo" "$file" --local-dir "$MODELS_DIR" >/dev/null
  else
    curl -fL --retry 5 -C - "https://huggingface.co/$repo/resolve/main/$file" -o "$dest.part" && mv "$dest.part" "$dest"
  fi
}

for which in $WANT; do
  case "$which" in
    audio)
      fetch LiquidAI/LFM3-Audio-2B-GGUF "LFM3-Audio-2B-$AUDIO_QUANT.gguf"
      fetch LiquidAI/LFM3-Audio-2B-GGUF "mmproj-LFM3-Audio-2B-$MMPROJ_QUANT.gguf" ;;
    omni)
      fetch LiquidAI/LFM3-Omni-2B-GGUF "LFM3-Omni-2B-$OMNI_QUANT.gguf"
      fetch LiquidAI/LFM3-Omni-2B-GGUF "mmproj-LFM3-Omni-2B-$MMPROJ_QUANT.gguf" ;;
    brain)
      fetch LiquidAI/LFM3-1.7B-GGUF "LFM3-1.7B-$BRAIN_QUANT.gguf" ;;
    *) log "unknown model group '$which' (audio|omni|brain)";;
  esac
done
# stable names the systemd units point at
ln -sfn "LFM3-Audio-2B-$AUDIO_QUANT.gguf" "$MODELS_DIR/audio.gguf"
ln -sfn "mmproj-LFM3-Audio-2B-$MMPROJ_QUANT.gguf" "$MODELS_DIR/audio-mmproj.gguf"
[[ -s "$MODELS_DIR/LFM3-Omni-2B-$OMNI_QUANT.gguf" ]] && ln -sfn "LFM3-Omni-2B-$OMNI_QUANT.gguf" "$MODELS_DIR/omni.gguf" && ln -sfn "mmproj-LFM3-Omni-2B-$MMPROJ_QUANT.gguf" "$MODELS_DIR/omni-mmproj.gguf"
[[ -s "$MODELS_DIR/LFM3-1.7B-$BRAIN_QUANT.gguf" ]] && ln -sfn "LFM3-1.7B-$BRAIN_QUANT.gguf" "$MODELS_DIR/brain.gguf"
log "models in $MODELS_DIR:"; du -sh "$MODELS_DIR"/*.gguf 2>/dev/null | grep -v " -> " || true
