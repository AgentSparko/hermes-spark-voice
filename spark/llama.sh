#!/usr/bin/env bash
# Install llama.cpp with the LFM3 audio/omni/realtime support (b11960 or newer) into $PREFIX.
# On a DGX Spark (aarch64 + CUDA 13, sm_121) we build from source with CUDA; elsewhere we try a
# release binary first and fall back to a CPU build. Idempotent: re-running updates in place.
set -euo pipefail
PREFIX="${SPARK_VOICE_HOME:-$HOME/.hermes-spark-voice}"
SRC="$PREFIX/src/llama.cpp"
MIN_TAG="${LLAMA_MIN_TAG:-b11960}"
PIN="${LLAMA_TAG:-}"            # pin a tag, e.g. LLAMA_TAG=b12010; default = latest release
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
mkdir -p "$PREFIX/bin" "$PREFIX/src"

log(){ printf '\033[1;34m[llama]\033[0m %s\n' "$*"; }
die(){ printf '\033[1;31m[llama] %s\033[0m\n' "$*" >&2; exit 1; }

have_new_enough(){ # $1 = llama-server path
  local v; v="$("$1" --version 2>&1 | grep -oE 'b[0-9]+' | head -1 || true)"
  [[ -n "$v" && "${v#b}" -ge "${MIN_TAG#b}" ]]
}

if [[ -x "$PREFIX/bin/llama-server" ]] && have_new_enough "$PREFIX/bin/llama-server" && [[ -z "${LLAMA_FORCE:-}" ]]; then
  log "llama-server already installed: $("$PREFIX/bin/llama-server" --version 2>&1 | head -1)"; exit 0
fi

tag="$PIN"
if [[ -z "$tag" ]]; then
  tag="$(curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | grep -oE '"tag_name": *"[^"]+"' | cut -d'"' -f4 || true)"
  [[ -n "$tag" ]] || tag="master"
fi
log "target llama.cpp $tag (minimum $MIN_TAG)"

arch="$(uname -m)"; os="$(uname -s)"
use_cuda=0
if command -v nvcc >/dev/null 2>&1 || [[ -x /usr/local/cuda/bin/nvcc ]]; then use_cuda=1; export PATH="/usr/local/cuda/bin:$PATH"; fi

# 1) prebuilt release (x86_64 Linux CUDA / macOS arm64) when it exists for this platform
try_release(){
  [[ "$tag" != "master" ]] || return 1
  local asset=""
  case "$os-$arch-$use_cuda" in
    Linux-x86_64-1) asset="llama-$tag-bin-ubuntu-cuda-x64.tar.gz" ;;   # name may differ across releases: fall through to build
    Linux-x86_64-0) asset="llama-$tag-bin-ubuntu-x64.tar.gz" ;;
    Darwin-arm64-*) asset="llama-$tag-bin-macos-arm64.tar.gz" ;;
    *) return 1 ;;
  esac
  local url="https://github.com/ggml-org/llama.cpp/releases/download/$tag/$asset"
  local tmp; tmp="$(mktemp -d)"
  if curl -fsSL "$url" -o "$tmp/l.tgz" 2>/dev/null; then
    tar -xzf "$tmp/l.tgz" -C "$tmp"
    local bin; bin="$(find "$tmp" -type f -name llama-server | head -1)"
    [[ -n "$bin" ]] || return 1
    cp -f "$(dirname "$bin")"/* "$PREFIX/bin/" 2>/dev/null || true
    chmod +x "$PREFIX/bin/llama-server"
    have_new_enough "$PREFIX/bin/llama-server" && { log "installed release binary $asset"; return 0; }
  fi
  return 1
}

build_from_source(){
  command -v cmake >/dev/null || die "cmake missing (sudo apt install -y cmake build-essential git libcurl4-openssl-dev)"
  if [[ -d "$SRC/.git" ]]; then git -C "$SRC" fetch -q --tags origin; else git clone -q https://github.com/ggml-org/llama.cpp.git "$SRC"; fi
  if [[ "$tag" == "master" ]]; then git -C "$SRC" checkout -q origin/master; else git -C "$SRC" checkout -q "$tag"; fi
  local flags=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON)
  if [[ "$use_cuda" == 1 ]]; then
    # DGX Spark GB10 = sm_121; let CMake pick native when it can, else name it.
    flags+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS:-native}")
    log "building with CUDA (archs=${CUDA_ARCHS:-native})"
  elif [[ "$os" == Darwin ]]; then
    flags+=(-DGGML_METAL=ON)
  else
    log "no nvcc found: CPU build (slow; install the CUDA toolkit for the Spark)"
  fi
  cmake -S "$SRC" -B "$SRC/build" "${flags[@]}" >"$PREFIX/llama-cmake.log" 2>&1 || { tail -30 "$PREFIX/llama-cmake.log"; die "cmake configure failed (see $PREFIX/llama-cmake.log)"; }
  cmake --build "$SRC/build" --config Release -j"$JOBS" --target llama-server llama-cli >"$PREFIX/llama-build.log" 2>&1 || { tail -30 "$PREFIX/llama-build.log"; die "build failed (see $PREFIX/llama-build.log)"; }
  cp -f "$SRC/build/bin/"* "$PREFIX/bin/" 2>/dev/null || true
  have_new_enough "$PREFIX/bin/llama-server" || die "built llama-server is older than $MIN_TAG (tag $tag); set LLAMA_TAG to a newer release"
  log "built $("$PREFIX/bin/llama-server" --version 2>&1 | head -1)"
}

try_release || build_from_source
