"""
Calibration-focused tests for deepfake_detection.py / replay_detection.py:
that thresholds are configurable and documented as defaults (not fabricated
as "validated"), and that replay detection handles short/edge-case audio
without an overconfident false result. General direction-of-effect tests
for genuine vs. synthetic/replayed audio already live in
test_signal_detectors.py; this file covers the calibration behavior itself.
"""
import importlib
import os

import numpy as np
from scipy.signal import butter, lfilter

import app.services.deepfake_detection as deepfake_detection
import app.services.replay_detection as replay_detection
from app.services.deepfake_detection import detect_deepfake
from app.services.replay_detection import detect_replay

SR = 16000


def _live_like_signal(duration=3.0, seed=0):
    t = np.linspace(0, duration, int(SR * duration))
    rng = np.random.default_rng(seed)
    f0_jitter = 150 + np.cumsum(rng.standard_normal(len(t)) * 0.5)
    phase = 2 * np.pi * np.cumsum(f0_jitter) / SR
    sig = (0.5 * np.sin(phase) + 0.15 * np.sin(3 * phase) + 0.05 * np.sin(7 * phase)
           + 0.03 * rng.standard_normal(len(t)))
    return sig.astype(np.float32)


def _replay_like_signal(live_like):
    b, a = butter(4, 3000 / (SR / 2), btype="low")
    filtered = lfilter(b, a, live_like)
    return (np.tanh(filtered * 3) / 3).astype(np.float32)


def _ai_like_signal(duration=3.0):
    t = np.linspace(0, duration, int(SR * duration))
    phase = 2 * np.pi * 150 * t
    sig = 0.5 * np.sin(phase) + 0.15 * np.sin(3 * phase) + 0.05 * np.sin(7 * phase)
    return sig.astype(np.float32)


# --- Both modules document their calibration status honestly -----------

def test_deepfake_calibration_is_labeled_default_not_validated():
    assert deepfake_detection.CALIBRATION_SOURCE == "default-heuristic"


def test_replay_calibration_is_labeled_default_not_validated():
    assert replay_detection.CALIBRATION_SOURCE == "default-heuristic"


# --- Thresholds are configurable via env vars, not hardcoded magic numbers --

def test_deepfake_thresholds_are_configurable_via_env(monkeypatch):
    monkeypatch.setenv("DF_JITTER_BASELINE", "0.5")
    monkeypatch.setenv("DF_JITTER_WEIGHT", "999")
    importlib.reload(deepfake_detection)
    try:
        assert deepfake_detection.DF_JITTER_BASELINE == 0.5
        assert deepfake_detection.DF_JITTER_WEIGHT == 999.0
    finally:
        monkeypatch.delenv("DF_JITTER_BASELINE", raising=False)
        monkeypatch.delenv("DF_JITTER_WEIGHT", raising=False)
        importlib.reload(deepfake_detection)  # restore defaults for later tests


def test_replay_thresholds_are_configurable_via_env(monkeypatch):
    monkeypatch.setenv("RD_HF_RATIO_BASELINE", "0.5")
    monkeypatch.setenv("RD_MIN_FRAMES", "1")
    importlib.reload(replay_detection)
    try:
        assert replay_detection.RD_HF_RATIO_BASELINE == 0.5
        assert replay_detection.RD_MIN_FRAMES == 1
    finally:
        monkeypatch.delenv("RD_HF_RATIO_BASELINE", raising=False)
        monkeypatch.delenv("RD_MIN_FRAMES", raising=False)
        importlib.reload(replay_detection)  # restore defaults for later tests


# --- Replay: false-positive guard on too-short audio --------------------

def test_replay_rejects_too_short_audio_instead_of_guessing():
    too_short = _live_like_signal(duration=0.05)  # ~50ms, well under RD_MIN_FRAMES
    r = detect_replay(too_short, SR)
    assert r.available is False
    assert "short" in r.error


def test_replay_accepts_audio_at_minimum_duration():
    long_enough = _live_like_signal(duration=1.0)
    r = detect_replay(long_enough, SR)
    assert r.available is True


# --- Genuine / synthetic(AI-like) / replayed / edge-case audio ----------

def test_genuine_like_audio_scores_low_on_both_detectors():
    genuine = _live_like_signal()
    d = detect_deepfake(genuine, SR)
    r = detect_replay(genuine, SR)
    assert d.available and r.available
    assert d.ai_generated_probability < 0.5
    assert r.replay_probability < 0.5


def test_synthetic_perfectly_periodic_audio_scores_higher_as_ai_generated():
    ai_like = _ai_like_signal()
    d = detect_deepfake(ai_like, SR)
    assert d.available
    assert d.ai_generated_probability > 0.5


def test_replayed_like_audio_scores_higher_as_replay():
    genuine = _live_like_signal()
    replayed = _replay_like_signal(genuine)
    r = detect_replay(replayed, SR)
    assert r.available
    assert r.replay_probability > 0.5


def test_silence_does_not_crash_and_stays_in_bounds():
    silence = np.zeros(SR * 2, dtype=np.float32)
    d = detect_deepfake(silence, SR)
    r = detect_replay(silence, SR)
    if d.available:
        assert 0.0 <= d.ai_generated_probability <= 1.0
    if r.available:
        assert 0.0 <= r.replay_probability <= 1.0


def test_empty_waveform_is_unavailable_not_a_fabricated_score():
    empty = np.array([], dtype=np.float32)
    d = detect_deepfake(empty, SR)
    r = detect_replay(empty, SR)
    assert d.available is False and d.ai_generated_probability == 0.0
    assert r.available is False and r.replay_probability == 0.0
