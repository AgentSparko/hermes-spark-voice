import numpy as np

from spark_calld import audio


def test_pcm_roundtrip_and_resample():
    x = np.sin(np.linspace(0, 20 * np.pi, 4800)).astype(np.float32) * 0.5
    pcm = audio.f32_to_pcm16(x)
    back = audio.pcm16_to_f32(pcm)
    assert np.allclose(x, back, atol=1e-3)
    y = audio.resample(x, 48000, 16000)
    assert len(y) == 1600
    assert abs(audio.rms(y) - audio.rms(x)) < 0.02


def test_streaming_resampler_keeps_odd_tail():
    rs = audio.Resampler(48000, 2, 16000)
    stereo = audio.f32_to_pcm16(np.zeros(96, dtype=np.float32))  # 48 stereo frames
    out = rs.feed(stereo[:-3])  # cut in the middle of a sample pair
    out += rs.feed(stereo[-3:])
    assert len(out) == 2 * 16  # 48 frames @48k -> 16 @16k


def test_to_telegram_upmixes():
    mono = audio.f32_to_pcm16(np.ones(240, dtype=np.float32) * 0.1)  # 10 ms @24k
    out = audio.to_telegram(mono, 24000, channels=2)
    assert len(out) == 480 * 2 * 2


def test_i420_roundtrip():
    rgb = np.zeros((36, 64, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    i420 = audio.rgb_to_i420(rgb)
    assert len(i420) == 64 * 36 * 3 // 2
    back = audio.i420_to_rgb(i420, 64, 36)
    assert back.shape == rgb.shape and abs(int(back[0, 0, 0]) - 200) < 12 and back[0, 0, 1] < 40
