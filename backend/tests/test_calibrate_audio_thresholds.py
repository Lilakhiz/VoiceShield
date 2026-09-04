"""
Validates the one piece of non-trivial logic in the calibration script
(scripts/calibrate_audio_thresholds.py): the brute-force equal-error-rate
sweep used to report how well each acoustic cue separates genuine vs.
attack audio.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from calibrate_audio_thresholds import equal_error_rate


def test_perfectly_separable_scores_have_zero_eer():
    genuine = np.array([0.0, 0.1, 0.2, 0.3])
    attack = np.array([0.8, 0.9, 1.0, 1.1])
    eer, threshold = equal_error_rate(genuine, attack)
    assert eer == 0.0
    assert 0.3 < threshold <= 0.8


def test_identical_distributions_have_high_eer():
    rng = np.random.default_rng(0)
    genuine = rng.normal(0, 1, 200)
    attack = rng.normal(0, 1, 200)
    eer, _ = equal_error_rate(genuine, attack)
    assert eer > 0.3  # indistinguishable distributions -> can't do much better than chance


def test_eer_is_between_zero_and_one():
    genuine = np.array([0.2, 0.5, 0.5, 0.9])
    attack = np.array([0.1, 0.4, 0.6, 0.95])
    eer, _ = equal_error_rate(genuine, attack)
    assert 0.0 <= eer <= 1.0
