"""
Replay / recorded-audio detection.

Approach: classic anti-spoofing signal-processing cues used in the
ASVspoof "physical access / replay" literature, computed directly from
the waveform via librosa. No pretrained weights are required, so this
module works fully offline and deterministically -- it is a real
feature-based detector, not a placeholder.

Cues combined (in order of how much each drives the final score, based
on validation against synthetic low-pass/dynamic-range-compressed test
signals simulating a loudspeaker-mic replay channel -- see
`test_replay_detection.py`):
  1. High-frequency roll-off energy ratio (dominant cue): replayed audio
     played through a loudspeaker and re-recorded loses high-frequency
     content compared to a live microphone capture, because both
     consumer speakers and the re-recording mic roll off well before
     Nyquist. This is the most robust, physically-grounded cue.
  2. Spectral rolloff frequency (the frequency below which 85% of the
     energy is concentrated): a second, independent way of measuring
     the same "missing highs" effect, using a standard librosa feature.
  3. Long-term variance of the spectral centroid: live speech has more
     natural micro-variation; loudspeaker playback + re-recording tends
     to compress dynamic range and smooth this out.

These are combined into a single 0..1 "replay_probability" via a
logistic function on a weighted sum of features, each expressed
relative to a "typical live speech" baseline.

CALIBRATION STATUS: DEFAULT (uncalibrated). Baselines/weights below are
heuristic starting points, not values fit against a labeled corpus --
no ASVspoof-scale replay/spoofing dataset (e.g. ASVspoof2019 PA) was
available to evaluate against in this environment, so nothing here
should be read as "validated". They're configurable via env vars (see
RD_* constants below) and CALIBRATION_SOURCE records whether they've
since been replaced with empirically-derived values. Run
`scripts/calibrate_audio_thresholds.py` against a labeled corpus to
derive those values.
"""
from __future__ import annotations
import os
import numpy as np
import librosa

from app.models.schemas import ReplayResult

CALIBRATION_SOURCE = "default-heuristic"  # e.g. "asvspoof2019-pa" once real calibration is done

# Natural live speech at 16kHz typically has ~12% of STFT energy above
# 6kHz; a loudspeaker+re-recording channel rolls this off.
RD_HF_RATIO_BASELINE = float(os.environ.get("RD_HF_RATIO_BASELINE", "0.12"))
RD_HF_RATIO_WEIGHT = float(os.environ.get("RD_HF_RATIO_WEIGHT", "6.0"))  # dominant cue

# Spectral rolloff (85% energy point) normalized by Nyquist; ~0.35 is
# typical for natural live speech at 16kHz.
RD_ROLLOFF_BASELINE = float(os.environ.get("RD_ROLLOFF_BASELINE", "0.35"))
RD_ROLLOFF_WEIGHT = float(os.environ.get("RD_ROLLOFF_WEIGHT", "3.0"))  # corroborating cue

# Spectral-centroid variance, normalized by Nyquist^2; penalized directly
# (no baseline) as a light tie-breaker -- lower natural variation is
# consistent with playback/re-recording compressing dynamic range.
RD_CENTROID_VAR_WEIGHT = float(os.environ.get("RD_CENTROID_VAR_WEIGHT", "75.0"))

# Below this many analysis frames (~0.25s of audio at the STFT settings
# used here), the spectral features above are too noisy to trust -- a
# short clip could otherwise swing hf_ratio/rolloff to an extreme value
# from a single frame and produce an overconfident score in either
# direction. Report `available=False` instead of guessing.
RD_MIN_FRAMES = int(os.environ.get("RD_MIN_FRAMES", "16"))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _hf_ratio(stft: np.ndarray, freqs: np.ndarray, cutoff_hz: float = 6000) -> float:
    """Fraction of STFT energy at/above `cutoff_hz`. Factored out (rather
    than inlined in detect_replay) so scripts/calibrate_audio_thresholds.py
    can compute it directly against a labeled corpus."""
    hf_mask = freqs >= cutoff_hz
    hf_energy = stft[hf_mask, :].sum()
    total_energy = stft.sum() + 1e-9
    return float(hf_energy / total_energy)


def _rolloff_norm(y: np.ndarray, sample_rate: int) -> float:
    """Mean spectral-rolloff frequency (85% energy point), normalized by Nyquist."""
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate, roll_percent=0.85)[0]
    return float(np.mean(rolloff)) / (sample_rate / 2)


def _centroid_var_norm(y: np.ndarray, sample_rate: int) -> float:
    """Spectral-centroid variance, normalized by Nyquist^2 for cross-sample-rate comparability."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)[0]
    centroid_var = float(np.var(centroid)) if centroid.size > 1 else 0.0
    return centroid_var / (sample_rate / 2) ** 2


def detect_replay(waveform: np.ndarray, sample_rate: int) -> ReplayResult:
    try:
        if waveform.size == 0:
            return ReplayResult(replay_probability=0.0, live_probability=1.0,
                                 method="spectral-heuristic", available=False,
                                 error="empty waveform")

        y = waveform.astype(np.float32)
        if y.ndim > 1:
            y = librosa.to_mono(y)

        n_frames = 1 + len(y) // 256  # matches hop_length used below
        if n_frames < RD_MIN_FRAMES:
            return ReplayResult(replay_probability=0.0, live_probability=1.0,
                                 method="spectral-heuristic", available=False,
                                 error="audio too short for reliable replay analysis")

        # 1. High-frequency roll-off ratio (energy above 6kHz vs total)
        stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
        hf_ratio = _hf_ratio(stft, freqs)

        # 2. Spectral rolloff frequency (85% energy point), normalized by Nyquist.
        #    A lower rolloff means energy is concentrated in lower frequencies,
        #    consistent with a band-limited playback+re-recording channel.
        rolloff_norm = _rolloff_norm(y, sample_rate)

        # 3. Spectral centroid variance (dynamic range / naturalness cue),
        #    normalized by Nyquist^2 so it's comparable across sample rates.
        centroid_var_norm = _centroid_var_norm(y, sample_rate)

        # See CALIBRATION STATUS above for where these weights/baselines
        # come from and how to recalibrate them.
        score = (
            RD_HF_RATIO_WEIGHT * (RD_HF_RATIO_BASELINE - hf_ratio)
            + RD_ROLLOFF_WEIGHT * (RD_ROLLOFF_BASELINE - rolloff_norm)
            - RD_CENTROID_VAR_WEIGHT * centroid_var_norm
        )
        replay_probability = float(np.clip(_sigmoid(score), 0.0, 1.0))
        live_probability = float(1.0 - replay_probability)

        return ReplayResult(
            replay_probability=round(replay_probability, 4),
            live_probability=round(live_probability, 4),
            method="spectral-heuristic(hf_ratio+spectral_rolloff+centroid_var)",
        )
    except Exception as e:  # pragma: no cover - defensive, never fabricate a score on failure
        return ReplayResult(replay_probability=0.0, live_probability=0.0,
                             method="spectral-heuristic", available=False, error=str(e))
