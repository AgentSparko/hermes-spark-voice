import base64
import json
import wave
from pathlib import Path

import pytest

from plugin import concierge, lfm3_audio as lfm3, tools  # noqa: E402


def _settings(**over):
    cfg = {"audio_url": "http://127.0.0.1:1", "voice": "female", "speaker_ref": "", "language": "English", "timeout": 5,
           "front_desk": "off", "brain_url": "http://127.0.0.1:1"}
    cfg.update(over)
    return lfm3.Settings(lambda k, d=None: cfg.get(k, d))


def _wav(path: Path, seconds: float = 0.2, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


class TestSTT:
    def test_posts_asr_prompt_and_returns_transcript(self, tmp_path, monkeypatch):
        seen = {}

        def fake_post(url, payload, timeout):
            seen["url"], seen["payload"] = url, payload
            return json.dumps({"choices": [{"message": {"content": "<think></think> hello there "}}]}).encode(), "application/json"

        monkeypatch.setattr(lfm3, "_post_json", fake_post)
        stt = lfm3.LFM3AudioSTT(_settings())
        out = stt.transcribe(str(_wav(tmp_path / "a.wav")), language="fi")
        assert out["success"] and out["transcript"] == "hello there"
        assert seen["url"].endswith("/v1/chat/completions")
        assert seen["payload"]["messages"][0]["content"] == "Perform ASR in Finnish."
        part = seen["payload"]["messages"][1]["content"][0]
        assert part["type"] == "input_audio" and base64.b64decode(part["input_audio"]["data"])[:4] == b"RIFF"

    def test_server_down_is_an_error_not_an_exception(self, tmp_path, monkeypatch):
        import urllib.error

        def boom(*a, **k):
            raise urllib.error.URLError("refused")

        monkeypatch.setattr(lfm3, "_post_json", boom)
        out = lfm3.LFM3AudioSTT(_settings()).transcribe(str(_wav(tmp_path / "a.wav")))
        assert out["success"] is False and "unreachable" in out["error"]

    def test_ogg_is_converted_when_ffmpeg_present(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(lfm3.shutil, "which", lambda n: "/usr/bin/ffmpeg")

        def fake_run(args, **kw):
            calls.append(args)
            _wav(Path(args[-1]))
            return None

        monkeypatch.setattr(lfm3.subprocess, "run", fake_run)
        monkeypatch.setattr(lfm3, "_post_json", lambda *a, **k: (json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode(), ""))
        ogg = tmp_path / "note.ogg"
        ogg.write_bytes(b"OggS")
        out = lfm3.LFM3AudioSTT(_settings()).transcribe(str(ogg))
        assert out["success"] and calls and "-ar" in calls[0]
        assert not list(tmp_path.glob("lfm3-asr-*"))


class TestTTS:
    def test_payload_voice_and_lang(self):
        tts = lfm3.LFM3AudioTTS(_settings(voice="male", language="German"))
        p = tts._tts_payload("Hallo", None, None)
        assert p == {"input": "Hallo", "response_format": "wav", "repeat_penalty": 1.05, "lang": "de", "voice": "male"}
        assert tts._tts_payload("x", "female", None)["voice"] == "female"

    def test_speaker_ref_is_sent_as_b64(self, tmp_path):
        ref = _wav(tmp_path / "me.wav")
        tts = lfm3.LFM3AudioTTS(_settings(speaker_ref=str(ref)))
        assert tts.default_voice() == "clone"
        p = tts._tts_payload("hi", None, None)
        assert "voice" not in p and base64.b64decode(p["speaker_ref_b64"])[:4] == b"RIFF"
        assert tts.list_voices()[0]["id"] == "clone"

    def test_synthesize_writes_wav(self, tmp_path, monkeypatch):
        wav_bytes = _wav(tmp_path / "src.wav").read_bytes()
        monkeypatch.setattr(lfm3, "_post_json", lambda url, payload, timeout: (wav_bytes, "audio/wav"))
        out = lfm3.LFM3AudioTTS(_settings()).synthesize("hello", str(tmp_path / "o" / "out.wav"), format="wav")
        assert Path(out).read_bytes()[:4] == b"RIFF"

    def test_synthesize_converts_with_ffmpeg(self, tmp_path, monkeypatch):
        wav_bytes = _wav(tmp_path / "src.wav").read_bytes()
        monkeypatch.setattr(lfm3, "_post_json", lambda url, payload, timeout: (wav_bytes, "audio/wav"))
        monkeypatch.setattr(lfm3.shutil, "which", lambda n: "/usr/bin/ffmpeg")
        seen = {}

        def fake_run(args, **kw):
            seen["args"] = args
            Path(args[-1]).write_bytes(b"OggS")

        monkeypatch.setattr(lfm3.subprocess, "run", fake_run)
        out = lfm3.LFM3AudioTTS(_settings()).synthesize("hello", str(tmp_path / "out.ogg"), format="opus", speed=1.3)
        assert Path(out).read_bytes() == b"OggS"
        assert "libopus" in seen["args"] and "atempo=1.30" in seen["args"]

    def test_empty_text_rejected(self):
        with pytest.raises(ValueError):
            lfm3.LFM3AudioTTS(_settings()).synthesize("  ", "/tmp/x.wav")

    def test_voice_compatible(self):
        assert lfm3.LFM3AudioTTS(_settings()).voice_compatible is True


class TestTools:
    def test_schemas_have_names(self):
        for name, schema in tools.schemas().items():
            assert schema["name"] == name and schema["parameters"]["type"] == "object"

    def test_handlers_report_daemon_down(self):
        s = _settings()
        s.calld_url, s.omni_url, s.brain_url = "http://127.0.0.1:1", "", ""
        h = tools.make_handlers(s)
        assert json.loads(h["voice_call_status"]({}))["ok"] is False
        assert json.loads(h["voice_call_say"]({}))["error"] == "text is required"
        models = json.loads(h["spark_voice_models"]({}))
        assert models["audio"]["up"] is False and "omni" not in models


class TestFrontDesk:
    def test_short_transcript_is_untouched_and_long_one_gets_a_brief(self, monkeypatch):
        c = concierge.Concierge(["http://127.0.0.1:1"])
        assert c.brief("call mom") == "call mom"
        monkeypatch.setattr(c, "_chat", lambda system, user, max_tokens: "Goal: x\n1. do y")
        long = " ".join(["word"] * 20)
        out = c.brief(long)
        assert out.startswith(long) and "[Front desk brief for the agent]" in out and "1. do y" in out

    def test_brief_falls_back_when_server_down(self):
        c = concierge.Concierge(["http://127.0.0.1:1"], timeout=1)
        long = " ".join(["word"] * 20)
        assert c.brief(long) == long

    def test_summary_only_for_long_or_structured(self, monkeypatch):
        c = concierge.Concierge(["http://127.0.0.1:1"])
        calls = []
        monkeypatch.setattr(c, "_chat", lambda s, u, m: calls.append(u) or "Done. Two things matter.")
        assert c.spoken_summary("All good, **done**.") == "All good, done." and not calls
        assert c.spoken_summary("x" * 400) == "Done. Two things matter." and calls
        assert c.spoken_summary("a\n- b\n- c") == "Done. Two things matter."

    def test_summary_fallback_is_plain(self):
        c = concierge.Concierge(["http://127.0.0.1:1"], timeout=1)
        out = c.spoken_summary("Hello ```code``` see https://x.y/z **bold** " + "more words. " * 60)
        assert "```" not in out and "http" not in out and "*" not in out and len(out) <= 500

    def test_tts_uses_summary_when_front_desk_on(self, tmp_path, monkeypatch):
        wav_bytes = _wav(tmp_path / "src.wav").read_bytes()
        seen = {}

        def fake_post(url, payload, timeout):
            seen["payload"] = payload
            return wav_bytes, "audio/wav"

        monkeypatch.setattr(lfm3, "_post_json", fake_post)
        monkeypatch.setattr(concierge.Concierge, "_chat", lambda self, s, u, m: "Short spoken version.")
        tts = lfm3.LFM3AudioTTS(_settings(front_desk="on"))
        tts.synthesize("- a\n- b\n" + "long " * 200, str(tmp_path / "o.wav"), format="wav")
        assert seen["payload"]["input"] == "Short spoken version."
        tts.synthesize("- a\n- b", str(tmp_path / "o2.wav"), format="wav", verbatim=True)
        assert seen["payload"]["input"] == "- a\n- b"


class TestHumanVoice:
    def test_strip_reasoning_blocks_and_labels(self):
        s = concierge.strip_reasoning
        assert s("<think>long thoughts</think>\nDone, pushed.") == "Done, pushed."
        assert s("Reasoning: I should check the file first.\n\n**Answer**\nIt is fixed.") == "It is fixed."
        assert s("reasoning\nWe looked at it.\nIt works.") == "We looked at it.\nIt works."
        assert s("<reasoning>partial, unterminated") == ""
        assert s("plain reply") == "plain reply"

    def test_humanize_hides_technical_tokens(self):
        out = concierge.humanize("Bound 127.0.0.1:8181 and localhost:8642, edited ~/.hermes/config.yaml and plugin/lfm3_audio.py, "
                                 "called make_handlers() in commit 2a3746b9dc, build b11960 v0.8.0, flag --jinja, see https://x.y/z")
        for bad in ("127.0.0.1", "8181", "localhost", "config.yaml", "lfm3_audio", "make_handlers", "2a3746b", "b11960", "0.8.0", "--jinja", "http"):
            assert bad not in out, (bad, out)
        assert "the local address" in out and "the file" in out and "the commit" in out

    def test_plain_caps_at_a_sentence(self):
        long = "This is sentence number one. " * 40
        out = concierge.plain(long)
        assert len(out) <= concierge.SPOKEN_MAX_CHARS and out.endswith(".")
        assert concierge.plain("I committed the two PRs locally, let me know if I should push and merge.") == \
            "I committed the two PRs locally, let me know if I should push and merge."

    def test_technical_short_answer_is_still_rephrased(self, monkeypatch):
        c = concierge.Concierge(["http://127.0.0.1:1"])
        monkeypatch.setattr(c, "_chat", lambda s, u, m: "I fixed the config, you can restart the gateway.")
        assert c.spoken_summary("Patched ~/.hermes/config.yaml (stt.provider).") == "I fixed the config, you can restart the gateway."
        assert "max_words" not in concierge.SUMMARY_SYSTEM.format(language="English", max_words=60)
        assert "I committed the two PRs locally" in concierge.SUMMARY_SYSTEM

    def test_summary_fallback_without_server_is_short_and_clean(self):
        c = concierge.Concierge(["http://127.0.0.1:1"], timeout=1)
        answer = "<think>x</think>Reasoning: steps.\n\nI edited /etc/hosts and ran setup_env() on 10.0.0.2:22. " + "More detail here. " * 80
        out = c.spoken_summary(answer)
        assert len(out) <= concierge.SPOKEN_MAX_CHARS and "10.0.0.2" not in out and "setup_env" not in out and "Reasoning" not in out
