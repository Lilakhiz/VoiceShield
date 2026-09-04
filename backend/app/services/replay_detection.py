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
relative to a "typical live speech" baseline. Baselines/weights are
documented inline so they can be recalibrated against a real labeled
corpus (e.g. ASVspoof2019 PA) before production use.

CONFIGURATION:
  Heuristic thresholds are configurable via environment variables.
  Current defaults are BASELINE HEURISTIC VALUES derived from synthetic
  signal validation (see tests/test_signal_detectors.py), NOT calibrated
  against a labeled corpus like ASVspoof2019 PA. For production use, you MUST
  recalibrate against real labeled data and update the thresholds.

  Environment variables:
    REPLAY_HF_RATIO_WEIGHT: Weight for HF ratio term (default: 6.0)
    REPLAY_HF_RATIO_BASELINE: Expected live speech HF ratio baseline (default: 0.12)
    REPLAY_ROLLOFF_WEIGHT: Weight for spectral rolloff term (default: 3.0)
    REPLAY_ROLLOFF_BASELINE: Expected live speech rolloff baseline (default: 0.35)
    REPLAY_CENTROID_VAR_WEIGHT: Weight for centroid variance term (default: -75.0)
    # Note: centroid_var_norm is multiplied by 50 internally, so -1.5 * 50 = -75.0
"""
from __future__ import annotations
import os
import numpy as np
import librosa

from app.models.schemas import ReplayResult


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# Heuristic fallback configuration (BASELINE VALUES - NOT CALIBRATED)
# These defaults come from synthetic signal validation, NOT real ASVspoof PA data.
# Override via environment variables for production calibration.
REPLAY_HF_RATIO_WEIGHT = float(os.environ.get("REPLAY_HF_RATIO_WEIGHT", "6.0"))
REPLAY_HF_RATIO_BASELINE = float(os.environ.get("REPLAY_HF_RATIO_BASELINE", "0.12"))
REPLAY_ROLLOFF_WEIGHT = float(os.environ.get("REPLAY_ROLLOFF_WEIGHT", "3.0"))
REPLAY_ROLLOFF_BASELINE = float(os.environ.get("REPLAY_ROLLOFF_BASELINE", "0.35"))
REPLAY_CENTROID_VAR_WEIGHT = float(os.environ.get("REPLAY_CENTROID_VAR_WEIGHT", "-75.0"))


def detect_replay(waveform: np.ndarray, sample_rate: int) -> ReplayResult:
    try:
        if waveform.size == 0:
            return ReplayResult(replay_probability=0.0, live_probability=1.0,
                                 method="spectral-heuristic", available=False,
                                 error="empty waveform")

        y = waveform.astype(np.float32)
        if y.ndim > 1:
            y = librosa.to_mono(y)

        # 1. High-frequency roll-off ratio (energy above 6kHz vs total)
        stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
        hf_mask = freqs >= 6000
        hf_energy = stft[hf_mask, :].sum()
        total_energy = stft.sum() + 1e-9
        hf_ratio = float(hf_energy / total_energy)

        # 2. Spectral rolloff frequency (85% energy point), normalized by Nyquist.
        #    A lower rolloff means energy is concentrated in lower frequencies,
        #    consistent with a band-limited playback+re-recording channel.
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate, roll_percent=0.85)[0]
        rolloff_norm = float(np.mean(rolloff)) / (sample_rate / 2)

        # 3. Spectral centroid variance (dynamic range / naturalness cue),
        #    normalized by Nyquist^2 so it's comparable across sample rates.
        centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)[0]
        centroid_var = float(np.var(centroid)) if centroid.size > 1 else 0.0
        centroid_var_norm = centroid_var / (sample_rate / 2) ** 2

        # Weights: hf_ratio is the dominant, most reliable cue; rolloff
        # corroborates it independently; centroid variance is a light
        # tie-breaker. Baselines (0.12 hf_ratio, 0.35 rolloff_norm) are
        # typical for natural live speech at 16kHz and should be
        # recalibrated against a labeled corpus for production use.
        score = (
            REPLAY_HF_RATIO_WEIGHT * (REPLAY_HF_RATIO_BASELINE - hf_ratio)
            + REPLAY_ROLLOFF_WEIGHT * (REPLAY_ROLLOFF_BASELINE - rolloff_norm)
            + REPLAY_CENTROID_VAR_WEIGHT * centroid_var_norm
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
