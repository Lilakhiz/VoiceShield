"""
AI-generated / deepfake voice detection.

Two tiers, tried in order, and the result always reports which one fired:

  1. PRETRAINED MODEL (preferred): if a local checkpoint for a
     spoofing-detection model (e.g. AASIST/RawNet2 trained on ASVspoof2019-LA)
     is present at MODEL_PATH, load and use it. This gives the best accuracy.
  2. SIGNAL-FEATURE FALLBACK: if no checkpoint is available (e.g. offline
     dev machine, or the model failed to download), fall back to a
     deterministic acoustic-artifact detector that looks at cues known to
     differ between natural and TTS/vocoder-generated speech:
        - Pitch (F0) micro-jitter: natural voices have irregular jitter;
          many neural vocoders produce unnaturally smooth F0 contours.
        - Phase coherence / harmonic-to-noise ratio: vocoder artifacts
          often show unnaturally high HNR (over-smoothed harmonics).
        - Spectral flatness in the 0-4kHz "formant" band: TTS systems
          sometimes over-regularize formant structure.

  Both branches produce a real, computed number -- never a random draw.
  If everything fails, `available=False` is returned and the risk engine
  is expected to skip this signal rather than substitute a fabricated one.

  The fallback's thresholds/weights are configurable (env vars, see
  DF_* constants below) and CALIBRATION_SOURCE records whether they're
  empirically validated or still the uncalibrated defaults -- see the
  "Signal-heuristic calibration" comment block for details.
"""
from __future__ import annotations
import os
import numpy as np
import librosa

from app.models.schemas import DeepfakeResult

MODEL_PATH = os.environ.get("DEEPFAKE_MODEL_PATH", "")
_pretrained_model = None
_pretrained_load_attempted = False

# --- Signal-heuristic calibration ---------------------------------------
# CALIBRATION STATUS: DEFAULT (uncalibrated). These baselines/weights are
# heuristic starting points grounded in general speech-science literature
# (typical natural-speech F0 jitter, HNR and formant-band-flatness ranges),
# not values fit against a labeled spoofing corpus. No ASVspoof-scale
# dataset (e.g. ASVspoof2019-LA) was available to download/evaluate against
# in this environment, so nothing here should be read as "validated".
# Run `scripts/calibrate_audio_thresholds.py` against a labeled bona-fide
# vs. spoof/TTS corpus to derive empirical values, then update the
# defaults below and change CALIBRATION_SOURCE to name the dataset used.
CALIBRATION_SOURCE = "default-heuristic"  # e.g. "asvspoof2019-la" once real calibration is done

# Natural speech F0 jitter is typically ~0.01-0.03; vocoders tend to
# produce unnaturally smooth (lower-jitter) pitch contours.
DF_JITTER_BASELINE = float(os.environ.get("DF_JITTER_BASELINE", "0.01"))
DF_JITTER_WEIGHT = float(os.environ.get("DF_JITTER_WEIGHT", "150.0"))

# Natural speech log1p(HNR) is typically ~1.0-2.0; vocoder artifacts often
# show unnaturally high harmonic-to-noise ratio (over-smoothed harmonics).
DF_HNR_LOG_BASELINE = float(os.environ.get("DF_HNR_LOG_BASELINE", "1.5"))
DF_HNR_WEIGHT = float(os.environ.get("DF_HNR_WEIGHT", "0.8"))

# TTS systems sometimes over-regularize formant structure, lowering
# spectral flatness in the 0-4kHz band relative to natural speech.
DF_FLATNESS_BASELINE = float(os.environ.get("DF_FLATNESS_BASELINE", "0.25"))
DF_FLATNESS_WEIGHT = float(os.environ.get("DF_FLATNESS_WEIGHT", "2.0"))


def _try_load_pretrained():
    """Attempt to load a local AASIST/RawNet2-style checkpoint if configured.
    This is intentionally lazy + defensive: on machines without the
    checkpoint (e.g. this sandbox with no HF Hub access) it simply reports
    unavailable and the caller falls back to the signal-based detector.
    """
    global _pretrained_model, _pretrained_load_attempted
    if _pretrained_load_attempted:
        return _pretrained_model
    _pretrained_load_attempted = True
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        return None
    try:
        import torch
        _pretrained_model = torch.jit.load(MODEL_PATH, map_location="cpu")
        _pretrained_model.eval()
        return _pretrained_model
    except Exception:
        _pretrained_model = None
        return None


def _f0_jitter(y: np.ndarray, sr: int) -> float:
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    f0 = f0[~np.isnan(f0)]
    if f0.size < 4:
        return 0.0
    diffs = np.abs(np.diff(f0))
    jitter = float(np.mean(diffs) / (np.mean(f0) + 1e-9))
    return jitter


def _harmonic_to_noise_ratio(y: np.ndarray) -> float:
    harmonic, percussive = librosa.effects.hpss(y)
    h_energy = float(np.sum(harmonic ** 2))
    n_energy = float(np.sum((y - harmonic) ** 2)) + 1e-9
    return h_energy / n_energy


def _formant_band_flatness(y: np.ndarray, sr: int) -> float:
    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    band_mask = freqs <= 4000
    band = stft[band_mask, :] + 1e-9
    geo_mean = np.exp(np.mean(np.log(band), axis=0))
    arith_mean = np.mean(band, axis=0)
    return float(np.mean(geo_mean / (arith_mean + 1e-9)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def detect_deepfake(waveform: np.ndarray, sample_rate: int) -> DeepfakeResult:
    if waveform.size == 0:
        return DeepfakeResult(ai_generated_probability=0.0, method="none",
                               available=False, error="empty waveform")

    y = waveform.astype(np.float32)
    if y.ndim > 1:
        y = librosa.to_mono(y)

    model = _try_load_pretrained()
    if model is not None:
        try:
            import torch
            with torch.no_grad():
                x = torch.tensor(y).unsqueeze(0)
                logits = model(x)
                prob = torch.softmax(logits, dim=-1)[0, 1].item()
            return DeepfakeResult(ai_generated_probability=round(float(prob), 4),
                                   method="pretrained-checkpoint")
        except Exception as e:
            # fall through to signal-based fallback rather than crash the call
            pass

    try:
        jitter = _f0_jitter(y, sample_rate)          # natural speech: higher jitter
        hnr = _harmonic_to_noise_ratio(y)             # vocoder speech: higher HNR
        flatness = _formant_band_flatness(y, sample_rate)

        # Combination of the three cues, each relative to its baseline (see
        # CALIBRATION STATUS above): low jitter, high HNR, and low formant
        # flatness all push toward "AI-generated".
        score = (
            DF_JITTER_WEIGHT * (DF_JITTER_BASELINE - jitter)
            + DF_HNR_WEIGHT * (np.log1p(hnr) - DF_HNR_LOG_BASELINE)
            + DF_FLATNESS_WEIGHT * (DF_FLATNESS_BASELINE - flatness)
        )
        prob = float(np.clip(_sigmoid(score), 0.0, 1.0))
        return DeepfakeResult(
            ai_generated_probability=round(prob, 4),
            method="signal-heuristic(f0_jitter+hnr+formant_flatness)",
        )
    except Exception as e:  # pragma: no cover
        return DeepfakeResult(ai_generated_probability=0.0, method="signal-heuristic",
                               available=False, error=str(e))
