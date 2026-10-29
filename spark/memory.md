# Memory budget on a 128 GB DGX Spark next to Qwen3.5-Flash-Next

Unified memory: whatever Qwen's server (vLLM/SGLang/TRT) reserves is gone for good, so size the
LFM3 sidecars against the ~10 GB you currently have free, and reclaim more by trimming Qwen's
KV-cache pool, not its concurrency.

| Service | Weights | KV / activations | Total (approx.) | Needed for |
|---|---|---|---|---|
| `lfm3-audio` LFM3-Audio-2B Q8_0 + mmproj F16, ctx 8k x2 | 2.4 GB + 0.3 GB | ~0.8 GB | **~3.5 GB** | voice notes, calls (always on) |
| `lfm3-omni` LFM3-Omni-2B Q4_K_M + mmproj F16, ctx 8k | 1.3 GB + 0.4 GB | ~0.6 GB | **~2.3 GB** | video calls / photos (on demand) |
| `lfm3-brain` LFM3-1.7B Q4_K_M, ctx 32k x2 | 1.1 GB | ~1.2 GB | **~2.4 GB** | only if Hermes should not use Qwen |

Default install = audio + omni = ~6 GB, leaving ~4 GB of your 10 GB. If you also want the LFM3 brain,
free ~3 GB by lowering Qwen's context pool: with 1.6 M tokens of pool, dropping to ~1.2 M frees
several GB on most engines (vLLM: `--gpu-memory-utilization`, `--max-num-batched-tokens`;
SGLang: `--mem-fraction-static`). Concurrency is unaffected: the pool is shared by all requests.

Knobs (edit `~/.hermes-spark-voice/env`, then `systemctl --user restart lfm3-audio`):
- `LFM3_AUDIO_CTX=4096` and `LFM3_AUDIO_PARALLEL=1` shave ~0.5 GB if you never overlap a call with voice notes.
- `AUDIO_QUANT=Q4_K_M` at install time halves the audio weights (speech quality drops a little).
- `systemctl --user stop lfm3-omni` when you are not on video calls; the Hermes tool `spark_voice_models` tells you what is up.
