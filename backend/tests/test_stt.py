"""
Deterministic, offline coverage for stt.py's control flow (model-load
failure, the successful transcribe+detect-language path, and a mid-call
exception), using monkeypatch to stand in for the real Whisper model
instead of downloading pretrained weights. The real-model test at the
bottom is skipped unless `whisper` and its cached weights are actually
available (this sandbox has neither -- see stt.py's module docstring).
"""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import app.services.stt as stt

SR = 16000


def _fake_waveform(duration=1.0):
    t = np.linspace(0, duration, int(SR * duration), dtype=np.float32)
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _install_fake_whisper(monkeypatch, *, detect_probs, transcript, decode_error=None):
    """Stand in for the `whisper` package so the function's internal
    `import whisper` succeeds without the real (unavailable) package."""
    fake_model = SimpleNamespace(device="cpu", detect_language=lambda mel: (None, detect_probs))

    def decode(model, mel, options):
        if decode_error:
            raise decode_error
        return SimpleNamespace(text=transcript)

    fake_whisper = ModuleType("whisper")
    fake_whisper.pad_or_trim = lambda y: y
    fake_whisper.log_mel_spectrogram = lambda audio: SimpleNamespace(to=lambda device: "mel")
    fake_whisper.DecodingOptions = lambda **kwargs: kwargs
    fake_whisper.decode = decode

    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    monkeypatch.setattr(stt, "_model", fake_model)
    monkeypatch.setattr(stt, "_load_error", None)


def test_model_unavailable_returns_error_result(monkeypatch):
    # patch _get_model itself (not just _model/_load_error) so the real
    # loader -- which would try to `import whisper` and clobber
    # `_load_error` with its own message -- never runs.
    monkeypatch.setattr(stt, "_get_model", lambda: None)
    monkeypatch.setattr(stt, "_load_error", "no network")
    result = stt.transcribe_and_detect_language(_fake_waveform(), SR)
    assert result.available is False
    assert result.language == "unknown"
    assert result.transcript == ""
    assert result.error == "no network"


def test_model_unavailable_without_load_error_uses_default_message(monkeypatch):
    monkeypatch.setattr(stt, "_get_model", lambda: None)
    monkeypatch.setattr(stt, "_load_error", None)
    result = stt.transcribe_and_detect_language(_fake_waveform(), SR)
    assert result.available is False
    assert result.error == "Whisper model not loaded"


def test_successful_transcription_reports_detected_language_and_text(monkeypatch):
    _install_fake_whisper(
        monkeypatch, detect_probs={"en": 0.82, "hi": 0.1, "kn": 0.08}, transcript="  send me the otp  "
    )
    result = stt.transcribe_and_detect_language(_fake_waveform(), SR)
    assert result.available is True
    assert result.language == "en"
    assert result.confidence == 0.82
    assert result.transcript == "send me the otp"  # stripped


def test_picks_the_highest_probability_language(monkeypatch):
    _install_fake_whisper(monkeypatch, detect_probs={"en": 0.2, "hi": 0.7, "ta": 0.1}, transcript="...")
    result = stt.transcribe_and_detect_language(_fake_waveform(), SR)
    assert result.language == "hi"
    assert result.confidence == 0.7


def test_resamples_non_16k_audio_without_crashing(monkeypatch):
    _install_fake_whisper(monkeypatch, detect_probs={"en": 0.9}, transcript="hello")
    result = stt.transcribe_and_detect_language(_fake_waveform(), sample_rate=44100)
    assert result.available is True
    assert result.language == "en"


def test_decode_exception_is_caught_and_reported(monkeypatch):
    _install_fake_whisper(
        monkeypatch, detect_probs={"en": 0.9}, transcript="", decode_error=RuntimeError("boom")
    )
    result = stt.transcribe_and_detect_language(_fake_waveform(), SR)
    assert result.available is False
    assert result.language == "unknown"
    assert "boom" in result.error


# --- Real-model integration: only runs where the weights actually exist --

_WHISPER_AVAILABLE = importlib.util.find_spec("whisper") is not None
_WHISPER_CACHED = (Path.home() / ".cache" / "whisper" / f"{stt.WHISPER_MODEL_NAME}.pt").exists()


@pytest.mark.skipif(
    not (_WHISPER_AVAILABLE and _WHISPER_CACHED),
    reason="whisper package + cached model weights not available in this environment",
)
def test_real_whisper_model_transcribes_without_crashing():
    result = stt.transcribe_and_detect_language(_fake_waveform(duration=2.0), SR)
    assert result.available is True
    assert isinstance(result.transcript, str)
    assert 0.0 <= result.confidence <= 1.0
