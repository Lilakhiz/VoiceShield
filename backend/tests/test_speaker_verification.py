"""
Deterministic coverage for speaker_verification.py's enrollment and
verification logic. The real SpeechBrain ECAPA-TDNN model needs a
network download of pretrained weights (see the module docstring), so
`_embed` is monkeypatched to return fixed vectors -- these tests pin
down enrollment averaging, cosine-similarity math, and the
verified/not-verified threshold decision without any ML dependency.
"""
import numpy as np
import pytest

import app.services.speaker_verification as sv

SR = 16000


@pytest.fixture(autouse=True)
def _reset_enrolled_speakers():
    """Enrollment is module-level global state; keep tests isolated."""
    sv._enrolled_speakers.clear()
    yield
    sv._enrolled_speakers.clear()


def _waveform():
    return np.zeros(SR, dtype=np.float32)  # content is irrelevant; _embed is mocked


def _mock_embed(monkeypatch, vector):
    monkeypatch.setattr(sv, "_embed", lambda waveform, sample_rate: np.array(vector, dtype=np.float32))


def test_cosine_similarity_identical_vectors_is_one():
    a = np.array([1.0, 0.0])
    assert sv._cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert sv._cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_enroll_speaker_stores_embedding(monkeypatch):
    _mock_embed(monkeypatch, [1.0, 0.0])
    assert sv.enroll_speaker("alice", _waveform(), SR) is True
    assert sv.list_enrolled_speakers() == ["alice"]


def test_enroll_speaker_returns_false_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(sv, "_embed", lambda waveform, sample_rate: None)
    assert sv.enroll_speaker("alice", _waveform(), SR) is False
    assert sv.list_enrolled_speakers() == []


def test_repeated_enrollment_averages_the_embedding(monkeypatch):
    _mock_embed(monkeypatch, [1.0, 0.0])
    sv.enroll_speaker("alice", _waveform(), SR)
    _mock_embed(monkeypatch, [0.0, 1.0])
    sv.enroll_speaker("alice", _waveform(), SR)
    np.testing.assert_allclose(sv._enrolled_speakers["alice"], [0.5, 0.5])


def test_verify_speaker_matches_claimed_identity_above_threshold(monkeypatch):
    _mock_embed(monkeypatch, [1.0, 0.0])
    sv.enroll_speaker("alice", _waveform(), SR)
    result = sv.verify_speaker(_waveform(), SR, claimed_speaker_id="alice")
    assert result.verified is True
    assert result.matched_speaker_id == "alice"
    assert result.similarity == pytest.approx(1.0)


def test_verify_speaker_rejects_claimed_identity_below_threshold(monkeypatch):
    _mock_embed(monkeypatch, [1.0, 0.0])
    sv.enroll_speaker("alice", _waveform(), SR)
    _mock_embed(monkeypatch, [0.0, 1.0])  # probe doesn't match alice's voiceprint
    result = sv.verify_speaker(_waveform(), SR, claimed_speaker_id="alice")
    assert result.verified is False
    assert result.matched_speaker_id is None


def test_verify_speaker_without_claim_finds_best_match_across_enrolled(monkeypatch):
    _mock_embed(monkeypatch, [1.0, 0.0])
    sv.enroll_speaker("alice", _waveform(), SR)
    _mock_embed(monkeypatch, [0.0, 1.0])
    sv.enroll_speaker("bob", _waveform(), SR)

    _mock_embed(monkeypatch, [0.9, 0.1])  # closer to alice
    result = sv.verify_speaker(_waveform(), SR)
    assert result.matched_speaker_id == "alice"
    assert result.verified is True


def test_verify_speaker_reports_error_when_no_speakers_enrolled(monkeypatch):
    _mock_embed(monkeypatch, [1.0, 0.0])
    result = sv.verify_speaker(_waveform(), SR, claimed_speaker_id="alice")
    assert result.available is True
    assert result.verified is False
    assert result.error == "no enrolled speakers"


def test_verify_speaker_unavailable_when_embedding_fails(monkeypatch):
    monkeypatch.setattr(sv, "_embed", lambda waveform, sample_rate: None)
    result = sv.verify_speaker(_waveform(), SR, claimed_speaker_id="alice")
    assert result.available is False
    assert result.error is not None
