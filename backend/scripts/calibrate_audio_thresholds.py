"""
Calibration structure for deepfake_detection.py / replay_detection.py.

No labeled spoofing corpus (e.g. ASVspoof2019 LA/PA) was available in the
development environment, so the thresholds those modules ship with are
documented defaults, not empirically validated values (see the
CALIBRATION STATUS comment in each module). This script is how to turn
real labeled audio into calibrated values, once such a corpus is available:

    python scripts/calibrate_audio_thresholds.py deepfake \
        --genuine-dir /path/to/asvspoof/bonafide \
        --attack-dir  /path/to/asvspoof/spoof

    python scripts/calibrate_audio_thresholds.py replay \
        --genuine-dir /path/to/asvspoof/PA/bonafide \
        --attack-dir  /path/to/asvspoof/PA/replay

For each cue it reports the genuine/attack distributions and the
single-feature equal-error-rate (EER) achieved by thresholding that cue
alone, so a human can judge which cues are actually worth their current
weight before hand-editing the DF_*/RD_* env-var defaults. It intentionally
does not auto-fit and overwrite weights: a silent black-box refit is worse
than an explicit, reviewed threshold change on a security-relevant detector.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.deepfake_detection import _f0_jitter, _harmonic_to_noise_ratio, _formant_band_flatness
from app.services.replay_detection import _hf_ratio, _rolloff_norm, _centroid_var_norm


def _load_wavs(directory: str, sr: int = 16000) -> list[np.ndarray]:
    paths = sorted(glob.glob(os.path.join(directory, "**", "*.wav"), recursive=True))
    return [librosa.load(p, sr=sr)[0] for p in paths]


def equal_error_rate(genuine_scores: np.ndarray, attack_scores: np.ndarray) -> tuple[float, float]:
    """Sweep every observed score as a decision threshold (score >= t -> "attack")
    and return (EER, threshold_at_EER). Brute-force but exact and dependency-free."""
    thresholds = np.unique(np.concatenate([genuine_scores, attack_scores]))
    best_eer, best_t = 1.0, thresholds[0]
    for t in thresholds:
        false_positive_rate = float(np.mean(genuine_scores >= t))   # genuine misflagged as attack
        false_negative_rate = float(np.mean(attack_scores < t))     # attack missed
        eer = (false_positive_rate + false_negative_rate) / 2
        if eer < best_eer:
            best_eer, best_t = eer, t
    return best_eer, float(best_t)


def _report_cue(name: str, genuine: np.ndarray, attack: np.ndarray) -> None:
    eer, threshold = equal_error_rate(genuine, attack)
    print(f"  {name}:")
    print(f"    genuine  mean={genuine.mean():.4f} median={np.median(genuine):.4f} std={genuine.std():.4f}")
    print(f"    attack   mean={attack.mean():.4f} median={np.median(attack):.4f} std={attack.std():.4f}")
    print(f"    single-feature EER={eer:.1%} at threshold={threshold:.4f}")


def calibrate_deepfake(genuine_dir: str, attack_dir: str) -> None:
    genuine = _load_wavs(genuine_dir)
    attack = _load_wavs(attack_dir)
    sr = 16000
    print(f"Loaded {len(genuine)} genuine, {len(attack)} attack (AI/TTS) clips.\n")

    jitter_g = np.array([_f0_jitter(y, sr) for y in genuine])
    jitter_a = np.array([_f0_jitter(y, sr) for y in attack])
    _report_cue("F0 jitter (lower -> more AI-like)", jitter_g, jitter_a)

    hnr_g = np.array([np.log1p(_harmonic_to_noise_ratio(y)) for y in genuine])
    hnr_a = np.array([np.log1p(_harmonic_to_noise_ratio(y)) for y in attack])
    _report_cue("log1p(HNR) (higher -> more AI-like)", hnr_g, hnr_a)

    flat_g = np.array([_formant_band_flatness(y, sr) for y in genuine])
    flat_a = np.array([_formant_band_flatness(y, sr) for y in attack])
    _report_cue("formant-band flatness (lower -> more AI-like)", flat_g, flat_a)

    print("\nSuggested DF_*_BASELINE values (midpoint of genuine/attack medians):")
    print(f"  DF_JITTER_BASELINE   ~= {(np.median(jitter_g) + np.median(jitter_a)) / 2:.4f}")
    print(f"  DF_HNR_LOG_BASELINE  ~= {(np.median(hnr_g) + np.median(hnr_a)) / 2:.4f}")
    print(f"  DF_FLATNESS_BASELINE ~= {(np.median(flat_g) + np.median(flat_a)) / 2:.4f}")


def calibrate_replay(genuine_dir: str, attack_dir: str) -> None:
    genuine = _load_wavs(genuine_dir)
    attack = _load_wavs(attack_dir)
    sr = 16000
    print(f"Loaded {len(genuine)} genuine (live), {len(attack)} attack (replay) clips.\n")

    def features(ys):
        hf, ro, cv = [], [], []
        for y in ys:
            stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
            hf.append(_hf_ratio(stft, freqs))
            ro.append(_rolloff_norm(y, sr))
            cv.append(_centroid_var_norm(y, sr))
        return np.array(hf), np.array(ro), np.array(cv)

    hf_g, ro_g, cv_g = features(genuine)
    hf_a, ro_a, cv_a = features(attack)
    _report_cue("HF-ratio (lower -> more replay-like)", hf_g, hf_a)
    _report_cue("rolloff_norm (lower -> more replay-like)", ro_g, ro_a)
    _report_cue("centroid_var_norm (lower -> more replay-like)", cv_g, cv_a)

    print("\nSuggested RD_*_BASELINE values (midpoint of genuine/attack medians):")
    print(f"  RD_HF_RATIO_BASELINE ~= {(np.median(hf_g) + np.median(hf_a)) / 2:.4f}")
    print(f"  RD_ROLLOFF_BASELINE  ~= {(np.median(ro_g) + np.median(ro_a)) / 2:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("detector", choices=["deepfake", "replay"])
    parser.add_argument("--genuine-dir", required=True, help="Directory of genuine/live/bona-fide .wav files")
    parser.add_argument("--attack-dir", required=True, help="Directory of AI/spoofed/replayed .wav files")
    args = parser.parse_args()

    if args.detector == "deepfake":
        calibrate_deepfake(args.genuine_dir, args.attack_dir)
    else:
        calibrate_replay(args.genuine_dir, args.attack_dir)


if __name__ == "__main__":
    main()
