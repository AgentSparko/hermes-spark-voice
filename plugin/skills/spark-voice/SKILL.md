---
name: telegram-voice
description: Voice on Telegram with local LFM3-Audio on the Spark.
version: 0.1.0
author: AgentSparko
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [voice, telegram, lfm3, local]
    category: communication
---

# Voice on Telegram (LFM3-Audio, local)

Hermes hears and speaks on Telegram without any cloud service. Voice notes are transcribed by
LFM3-Audio-2B (`stt.provider: lfm3-audio`) and replies are spoken by the same model
(`tts.provider: lfm3-audio`). Live audio/video calls are handled by the `spark-calld` daemon,
which talks to you with LFM3-Audio and delegates real work to this agent.

## When to Use

- The user sends a voice note and expects a spoken answer: nothing to do, the gateway handles it
  when `voice.auto_tts` is on. Answer briefly: spoken replies over 40 seconds are tiring.
- The user asks "are you on a call?", "hang up", "tell him on the call that ...": use the
  `voice_call_status`, `voice_call_hangup` and `voice_call_say` tools.
- Voice replies stop working: run `spark_voice_models` and read which server is down.

## Prerequisites

- `llama-server` with LFM3-Audio-2B on `audio_url` (default http://127.0.0.1:8181), installed by
  `hermes-spark-voice/install.sh` as the `lfm3-audio` systemd user unit.
- `ffmpeg` on PATH (ogg/opus voice notes are converted to wav for the model).

## Quick Reference

| Need | Do |
|------|----|
| Is voice up? | `spark_voice_models` |
| Who is on the call? | `voice_call_status` |
| Say something on the call | `voice_call_say` with one or two sentences |
| End the call | `voice_call_hangup` |
| Per-chat voice replies | `/voice on`, `/voice tts`, `/voice off` in the chat |

## Procedure: a task that arrived from a call

1. `spark-calld` forwards the spoken request through the Hermes API server; you receive it as a
   normal message whose session id starts with `call:`.
2. Do the work with your usual tools. Keep the final answer short and speakable: no markdown, no
   lists, no URLs read out letter by letter (say "I sent the link to the chat" and send it there).
3. If the work takes more than ~20 seconds, call `voice_call_say` with a one-line progress note.

## Pitfalls

- LFM3-Audio speaks what it is given: emoji, code blocks and tables sound like noise. Strip them.
- One synthesis call handles a few sentences well; for long text the gateway already splits.
- `speaker_ref` (voice cloning) needs a clean 5-15 s mono WAV; a noisy reference degrades every reply.

## Verification

`hermes spark-voice status` prints `audio ... up`; `hermes spark-voice say "hello" --out /tmp/h.wav`
produces a playable file; a voice note in Telegram comes back as a voice bubble.
