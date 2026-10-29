"""Run the plugin tests without a Hermes checkout: stub the two ABC modules the providers inherit."""
import abc
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import agent.tts_provider  # noqa: F401
except Exception:
    agent = types.ModuleType("agent")

    class _Base(abc.ABC):
        def is_available(self):
            return True

        def list_models(self):
            return []

    class TTSProvider(_Base):
        streams_pcm = False
        stream_sample_rate = None

        def list_voices(self):
            return []

        def default_voice(self):
            return None

        @property
        def voice_compatible(self):
            return False

    class TranscriptionProvider(_Base):
        pass

    tts_mod = types.ModuleType("agent.tts_provider")
    tts_mod.TTSProvider = TTSProvider
    tr_mod = types.ModuleType("agent.transcription_provider")
    tr_mod.TranscriptionProvider = TranscriptionProvider
    sys.modules["agent"] = agent
    sys.modules["agent.tts_provider"] = tts_mod
    sys.modules["agent.transcription_provider"] = tr_mod
