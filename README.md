# hermes-spark-voice

Talk to your [Hermes](https://github.com/NousResearch/hermes-agent) agent by voice — Telegram voice
notes and live audio/video calls — with everything running locally on one NVIDIA DGX Spark, next to
the big model you already serve (Qwen3.5-Flash-Next or anything OpenAI-compatible).

```
curl -fsSL https://raw.githubusercontent.com/AgentSparko/hermes-spark-voice/main/install.sh | bash
```

## How it works

```
 you (Telegram voice note / Matrix or Telegram call)
   │ speech
   ▼
 LFM3-Audio-2B  ── llama-server :8181 ──  hears, speaks, keeps the conversation natural (fast, ~3 GB)
   │ "front desk": turns what you said into a brief / task list
   ▼
 Hermes agent  ── your Qwen brain, all Hermes tools, memory, skills ──  does the real work (slow, thorough)
   │ long written answer  ──────────────────────────────►  posted in the chat as text
   ▼
 LFM3 front desk summarizes it in 2-4 plain sentences  ──►  spoken back to you
 LFM3-Omni-2B   ── llama-server :8182 ──  looks at your camera / photos when asked (optional, ~2.3 GB)
 LFM3-1.7B      ── llama-server :8183 ──  text front desk (briefs, summaries); optional Hermes brain
```

* **Voice notes** use Hermes' own gateway path: the plugin registers `lfm3-audio` as the STT and TTS
  provider, so a voice note comes in, is transcribed and shaped into a brief by LFM3, answered by your
  normal Hermes brain, and the answer is spoken back as a voice bubble (short summary) while the full
  text is sent as usual. `/voice on|tts|off` per chat as in Hermes.
* **Calls** are handled by `spark-calld`, a small daemon that streams the call audio into llama.cpp's
  `/v1/realtime` (LFM3-Audio speech-to-speech with tool calls). The model talks to you directly and calls
  `ask_hermes` (through the Hermes API server) for anything real, `look` (LFM3-Omni on your video) and
  `hang_up`. Two transports:
  * **Matrix** (recommended): the Hermes bot account answers 1:1 audio/video calls from Element over
    WebRTC. No extra account; reuses `MATRIX_*` from `~/.hermes/.env`.
  * **Telegram p2p**: Telegram bots cannot take calls, so this logs in a *second* Telegram account (a
    spare number) as the agent; you call it from your own account.

Nothing leaves the machine. All model servers bind to 127.0.0.1.

## Requirements

* DGX Spark (or any Linux/CUDA box; macOS Apple Silicon works for the audio part) with ~6 GB of free
  unified memory for the default audio + omni set — see [spark/memory.md](spark/memory.md) for the
  budget next to Qwen and how to free more by trimming Qwen's KV pool (concurrency is unaffected).
* `git curl python3 python3-venv ffmpeg cmake build-essential libcurl4-openssl-dev` and the CUDA toolkit
  (`nvcc`) for the llama.cpp build. The installer builds llama.cpp ≥ b11960 (first release with LFM3
  audio/omni and `/v1/realtime`) from source on arm64.
* Hermes installed and already talking to your Telegram bot (the installer installs Hermes if missing).

## What the installer does

1. Clones this repo to `~/.hermes-spark-voice/src`, creates a venv with `spark-calld`.
2. Builds/installs `llama-server` and downloads `LFM3-Audio-2B` (Q8_0 + mmproj), `LFM3-Omni-2B`
   (Q4_K_M) and `LFM3-1.7B` (Q4_K_M) from Hugging Face. `SPARK_VOICE_MODELS="audio"` for audio only.
3. Copies the Hermes plugin to `~/.hermes/plugins/spark-voice` and enables it.
4. Writes `stt.provider`/`tts.provider: lfm3-audio`, `voice.auto_tts: true`, routes `auxiliary.vision`
   to LFM3-Omni, enables the Hermes API server on 127.0.0.1:8642 (key generated into `~/.hermes/.env`).
   `BRAIN=lfm3` additionally makes LFM3-1.7B the main model; the default keeps your current brain.
   A timestamped backup of `config.yaml` is kept.
5. Installs and starts systemd user units `lfm3-audio`, `lfm3-omni`, `lfm3-brain`, `spark-calld`.

Re-running is safe; it updates in place.

## After install

```bash
hermes gateway restart              # pick up the providers
hermes spark-voice status           # servers + call daemon
hermes spark-voice say "hello" --out /tmp/hi.wav
~/.hermes-spark-voice/venv/bin/spark-calld doctor
```

Send a voice note to your bot. Say something long-winded: the transcript echo shows the brief LFM3 made
for Qwen; the voice reply is the spoken summary; the text reply is the full answer.

### Calls

**Matrix**: enable the Hermes Matrix platform (`MATRIX_HOMESERVER`, `MATRIX_ACCESS_TOKEN`,
`MATRIX_USER_ID` in `~/.hermes/.env`), open a DM with the bot in Element and press call. Allowed callers
default to everyone who can DM the bot; restrict with `telegram.allowed_users: ["@you:server"]` in
`~/.hermes-spark-voice/calld.yaml` (string ids are Matrix, integers are Telegram). Unencrypted DM rooms
only for now (call signalling in E2EE rooms is not read by the daemon).

**Telegram**: create an API id/hash at https://my.telegram.org for the *agent's* account, put them and
your numeric user id into `calld.yaml`, run `spark-calld login` once (phone + code), restart the unit.
Call that account from Telegram — audio or video.

On a video call the daemon shows a simple animated avatar and keeps the latest frame of your camera for
the `look` tool ("what am I holding?"). Set `video.avatar: false` to send audio only.

### Voice cloning

Put a clean 5-15 s mono WAV of the voice you want at e.g. `~/.hermes-spark-voice/voice.wav` and set
`plugins.entries.spark-voice.config.speaker_ref` (Hermes) and `audio.speaker_ref` (calld). LFM3-Audio's
own `female`/`male` voices are used otherwise.

## Front desk behaviour

`plugins.entries.spark-voice.config.front_desk: "off"` turns the shaping off (verbatim STT/TTS).
`summary_threshold` (chars, default 350) decides when an answer is summarized before being spoken.
The text server used for briefs/summaries is `brain_url` (LFM3-1.7B); if it is not running, LFM3-Audio
in text mode is used; if neither answers, the plugin falls back to markdown stripping.

## Layout

| Path | What |
|---|---|
| `plugin/` | Hermes plugin `spark-voice`: `lfm3-audio` STT/TTS providers, tools `voice_call_status/hangup/say`, `spark_voice_models`, CLI `hermes spark-voice`, skill `spark-voice:telegram-voice` |
| `calld/` | `spark-calld` daemon: realtime bridge, Matrix (aiortc) and Telegram (py-tgcalls) transports, control API on 127.0.0.1:8797 |
| `spark/` | `llama.sh` (build), `models.sh` (download), systemd units, `memory.md` |
| `scripts/configure-hermes.py` | idempotent Hermes config patcher |

## Uninstall

```bash
systemctl --user disable --now lfm3-audio lfm3-omni lfm3-brain spark-calld
rm -rf ~/.hermes-spark-voice ~/.hermes/plugins/spark-voice ~/.config/systemd/user/{lfm3-*,spark-calld}.service
# then restore ~/.hermes/config.yaml from the .bak-* copy if you want the old stt/tts providers back
```

## Development

```bash
python -m pytest -q plugin/tests calld/tests
hermes plugins doctor plugin --ci
```

MIT.
