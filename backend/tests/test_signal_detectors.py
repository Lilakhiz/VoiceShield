"""
Validates that the replay-detection and deepfake-detection signal
heuristics respond in the physically-correct direction on synthetic
test signals (no network / pretrained-checkpoint download required,
so this runs anywhere including CI and offline dev machines).
"""
import numpy as np
from scipy.signal import butter, lfilter

from app.services.replay_detection import detect_replay
from app.services.deepfake_detection import detect_deepfake

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


def test_replay_detector_flags_bandlimited_compressed_audio_higher():
    live = _live_like_signal()
    replay = _replay_like_signal(live)
    r_live = detect_replay(live, SR)
    r_replay = detect_replay(replay, SR)
    assert r_live.available and r_replay.available
    assert r_replay.replay_probability > r_live.replay_probability


def test_deepfake_detector_flags_zero_jitter_tone_higher():
    live = _live_like_signal()
    ai = _ai_like_signal()
    d_live = detect_deepfake(live, SR)
    d_ai = detect_deepfake(ai, SR)
    assert d_live.available and d_ai.available
    assert d_ai.ai_generated_probability > d_live.ai_generated_probability


def test_detectors_never_return_available_true_on_empty_audio():
    empty = np.array([], dtype=np.float32)
    r = detect_replay(empty, SR)
    d = detect_deepfake(empty, SR)
    assert r.available is False
    assert d.available is False
