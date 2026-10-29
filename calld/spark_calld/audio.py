"""PCM helpers (numpy only). Telegram gives/takes PCM16 at 48 kHz; the model speaks PCM16 at its own rate."""
from __future__ import annotations

import math

import numpy as np

TG_RATE = 48000


def pcm16_to_f32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def f32_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def downmix(x: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 1:
        return x
    n = (len(x) // channels) * channels
    return x[:n].reshape(-1, channels).mean(axis=1)


def upmix(x: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 1:
        return x
    return np.repeat(x[:, None], channels, axis=1).reshape(-1)


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear-interpolation resampler: plenty for speech between 16/24/48 kHz."""
    if src == dst or len(x) == 0:
        return x
    n_out = int(round(len(x) * dst / src))
    if n_out <= 0:
        return x[:0]
    pos = np.linspace(0, len(x) - 1, n_out, dtype=np.float64)
    return np.interp(pos, np.arange(len(x)), x).astype(np.float32)


def rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(np.square(x)))))


def frame_bytes(rate: int, channels: int, ms: int = 10) -> int:
    return rate * ms // 1000 * channels * 2


class Resampler:
    """Streaming wrapper: feed PCM16 bytes at (src_rate, src_channels), get PCM16 mono at dst_rate."""

    def __init__(self, src_rate: int, src_channels: int, dst_rate: int):
        self.src_rate, self.src_channels, self.dst_rate = src_rate, src_channels, dst_rate
        self._tail = b""

    def feed(self, data: bytes) -> bytes:
        data = self._tail + data
        step = 2 * self.src_channels
        keep = len(data) % step
        if keep:
            data, self._tail = data[:-keep], data[-keep:]
        else:
            self._tail = b""
        if not data:
            return b""
        x = downmix(pcm16_to_f32(data), self.src_channels)
        return f32_to_pcm16(resample(x, self.src_rate, self.dst_rate))


def to_telegram(pcm16_mono: bytes, src_rate: int, channels: int = 2) -> bytes:
    x = resample(pcm16_to_f32(pcm16_mono), src_rate, TG_RATE)
    return f32_to_pcm16(upmix(x, channels))


def rgb_to_i420(rgb: np.ndarray) -> bytes:
    """HxWx3 uint8 RGB -> planar I420 (yuv420p) bytes, BT.601 limited range."""
    h, w, _ = rgb.shape
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    y = 16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    u = 128 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0
    v = 128 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0
    u = u.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    v = v.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    return b"".join(np.clip(p, 16, 240).astype(np.uint8).tobytes() for p in (y, u, v))


def i420_to_rgb(data: bytes, w: int, h: int) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    y = buf[: w * h].reshape(h, w).astype(np.float32)
    u = buf[w * h: w * h + (w // 2) * (h // 2)].reshape(h // 2, w // 2).astype(np.float32)
    v = buf[w * h + (w // 2) * (h // 2): w * h + 2 * (w // 2) * (h // 2)].reshape(h // 2, w // 2).astype(np.float32)
    u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)[:h, :w] - 128
    v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)[:h, :w] - 128
    c = (y - 16) * 1.164
    rgb = np.stack([c + 1.596 * v, c - 0.392 * u - 0.813 * v, c + 2.017 * u], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)
