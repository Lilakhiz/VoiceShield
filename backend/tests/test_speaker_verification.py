"""
Tests for speaker enrollment and verification.
Uses synthetic embeddings so tests run without network/model downloads.
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.services.speaker_verification import (
    enroll_speaker,
    verify_speaker,
    list_enrolled_speakers,
    get_enrollment_count,
    get_enrollment_embedding,
    _enrolled_speakers,
    _cosine_similarity,
)


def _reset_enrolled_speakers():
    """Reset the in-memory enrollment store for test isolation."""
    _enrolled_speakers.clear()


def _make_embedding(values: list[float]) -> np.ndarray:
    """Create a normalized embedding from a list of values."""
    arr = np.array(values, dtype=np.float32)
    return arr / (np.linalg.norm(arr) + 1e-9)


class TestCosineSimilarity:
    """Tests for cosine similarity utility."""
    
    def test_identical_vectors(self):
        a = _make_embedding([1, 2, 3])
        b = _make_embedding([1, 2, 3])
        assert abs(_cosine_similarity(a, b) - 1.0) < 1e-6
    
    def test_orthogonal_vectors(self):
        a = _make_embedding([1, 0, 0])
        b = _make_embedding([0, 1, 0])
        assert abs(_cosine_similarity(a, b) - 0.0) < 1e-6
    
    def test_opposite_vectors(self):
        a = _make_embedding([1, 0, 0])
        b = _make_embedding([-1, 0, 0])
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6


class TestEnrollmentRunningMean:
    """Tests for proper running mean calculation during enrollment."""
    
    def setup_method(self):
        _reset_enrolled_speakers()
    
    def teardown_method(self):
        _reset_enrolled_speakers()
    
    @patch('app.services.speaker_verification._embed')
    def test_first_enrollment(self, mock_embed):
        """First enrollment: mean = embedding1."""
        emb1 = _make_embedding([3, 4])  # norm=5, normalized=[0.6, 0.8]
        mock_embed.return_value = emb1
        
        result = enroll_speaker("test_speaker", np.zeros(16000), 16000)
        
        assert result is True
        count = get_enrollment_count("test_speaker")
        mean_emb = get_enrollment_embedding("test_speaker")
        
        assert count == 1
        assert mean_emb is not None
        np.testing.assert_allclose(mean_emb, emb1)
    
    @patch('app.services.speaker_verification._embed')
    def test_two_enrollments(self, mock_embed):
        """Two enrollments: mean = (emb1 + emb2) / 2 (running mean, not re-normalized)."""
        emb1 = _make_embedding([1, 0])
        emb2 = _make_embedding([0, 1])
        
        mock_embed.side_effect = [emb1, emb2]
        
        enroll_speaker("test_speaker", np.zeros(16000), 16000)
        enroll_speaker("test_speaker", np.zeros(16000), 16000)
        
        count = get_enrollment_count("test_speaker")
        mean_emb = get_enrollment_embedding("test_speaker")
        
        assert count == 2
        # Running mean of already-normalized vectors (not re-normalized)
        expected = (emb1 + emb2) / 2
        np.testing.assert_allclose(mean_emb, expected, rtol=1e-5)
    
    @patch('app.services.speaker_verification._embed')
    def test_three_enrollments(self, mock_embed):
        """Three enrollments: mean = (emb1 + emb2 + emb3) / 3 (running mean, not re-normalized)."""
        emb1 = _make_embedding([1, 0, 0])
        emb2 = _make_embedding([0, 1, 0])
        emb3 = _make_embedding([0, 0, 1])
        
        mock_embed.side_effect = [emb1, emb2, emb3]
        
        enroll_speaker("test_speaker", np.zeros(16000), 16000)
        enroll_speaker("test_speaker", np.zeros(16000), 16000)
        enroll_speaker("test_speaker", np.zeros(16000), 16000)
        
        count = get_enrollment_count("test_speaker")
        mean_emb = get_enrollment_embedding("test_speaker")
        
        assert count == 3
        expected = (emb1 + emb2 + emb3) / 3
        np.testing.assert_allclose(mean_emb, expected, rtol=1e-5)
    
    @patch('app.services.speaker_verification._embed')
    def test_four_enrollments_equal_weight(self, mock_embed):
        """Four enrollments: verify every embedding has equal weight."""
        # Use distinct embeddings to verify equal weighting
        emb1 = _make_embedding([4, 0, 0, 0])
        emb2 = _make_embedding([0, 4, 0, 0])
        emb3 = _make_embedding([0, 0, 4, 0])
        emb4 = _make_embedding([0, 0, 0, 4])
        
        mock_embed.side_effect = [emb1, emb2, emb3, emb4]
        
        enroll_speaker("test_speaker", np.zeros(16000), 16000)
        enroll_speaker("test_speaker", np.zeros(16000), 16000)
        enroll_speaker("test_speaker", np.zeros(16000), 16000)
        enroll_speaker("test_speaker", np.zeros(16000), 16000)
        
        count = get_enrollment_count("test_speaker")
        mean_emb = get_enrollment_embedding("test_speaker")
        
        assert count == 4
        expected = (emb1 + emb2 + emb3 + emb4) / 4
        np.testing.assert_allclose(mean_emb, expected, rtol=1e-5)
        
        # Verify equal contribution: each dimension should be equal
        assert abs(mean_emb[0] - mean_emb[1]) < 1e-5
        assert abs(mean_emb[1] - mean_emb[2]) < 1e-5
        assert abs(mean_emb[2] - mean_emb[3]) < 1e-5
    
    @patch('app.services.speaker_verification._embed')
    def test_multiple_speakers_independent(self, mock_embed):
        """Enrollments for different speakers should be independent."""
        emb1 = _make_embedding([1, 0])
        emb2 = _make_embedding([0, 1])
        
        mock_embed.side_effect = [emb1, emb2, emb2]
        
        enroll_speaker("speaker_a", np.zeros(16000), 16000)
        enroll_speaker("speaker_b", np.zeros(16000), 16000)
        enroll_speaker("speaker_a", np.zeros(16000), 16000)  # speaker_a gets emb2 as second
        
        mean_a = get_enrollment_embedding("speaker_a")
        mean_b = get_enrollment_embedding("speaker_b")
        
        # speaker_a mean should be (emb1 + emb2) / 2 (running mean, not re-normalized)
        expected_a = (emb1 + emb2) / 2
        np.testing.assert_allclose(mean_a, expected_a, rtol=1e-5)
        
        # speaker_b mean should be just emb2
        np.testing.assert_allclose(mean_b, emb2)
    
    @patch('app.services.speaker_verification._embed')
    def test_enrollment_returns_false_on_model_failure(self, mock_embed):
        """enroll_speaker returns False when embedding model unavailable."""
        mock_embed.return_value = None
        
        result = enroll_speaker("test_speaker", np.zeros(16000), 16000)
        assert result is False
        assert get_enrollment_count("test_speaker") == 0


class TestVerificationWithEnrollments:
    """Tests that verification uses the enrollment mean correctly."""
    
    def setup_method(self):
        _reset_enrolled_speakers()
    
    def teardown_method(self):
        _reset_enrolled_speakers()
    
    @patch('app.services.speaker_verification._embed')
    def test_verification_matches_enrolled_speaker(self, mock_embed):
        """Verification should match when test embedding equals enrolled mean."""
        emb = _make_embedding([1, 0, 0, 0])
        mock_embed.side_effect = [emb, emb]  # First for enrollment, then for verification
        
        enroll_speaker("test_speaker", np.zeros(16000), 16000)
        
        result = verify_speaker(np.zeros(16000), 16000, claimed_speaker_id="test_speaker")
        
        assert result.available
        assert result.verified
        assert result.similarity >= 0.99  # Should be ~1.0
    
    @patch('app.services.speaker_verification._embed')
    def test_verification_rejects_different_speaker(self, mock_embed):
        """Verification should reject when test embedding differs from enrolled."""
        emb1 = _make_embedding([1, 0, 0, 0])
        emb2 = _make_embedding([0, 1, 0, 0])  # orthogonal
        
        mock_embed.side_effect = [emb1, emb2]
        
        enroll_speaker("test_speaker", np.zeros(16000), 16000)
        
        result = verify_speaker(np.zeros(16000), 16000, claimed_speaker_id="test_speaker")
        
        assert result.available
        assert not result.verified
        assert result.similarity < 0.5  # Should be ~0.0
    
    @patch('app.services.speaker_verification._embed')
    def test_verification_no_enrolled_speakers(self, mock_embed):
        """Verification should handle empty enrollment store."""
        emb = _make_embedding([1, 0])
        mock_embed.return_value = emb
        
        result = verify_speaker(np.zeros(16000), 16000, claimed_speaker_id="unknown")
        
        assert result.available
        assert not result.verified
        assert result.error == "no enrolled speakers"
    
    @patch('app.services.speaker_verification._embed')
    def test_verification_no_claimed_identity_best_match(self, mock_embed):
        """Without claimed identity, returns best match across enrolled."""
        emb1 = _make_embedding([1, 0, 0])
        emb2 = _make_embedding([0, 1, 0])
        
        mock_embed.side_effect = [emb1, emb2, emb1]
        
        enroll_speaker("speaker_a", np.zeros(16000), 16000)
        enroll_speaker("speaker_b", np.zeros(16000), 16000)
        
        result = verify_speaker(np.zeros(16000), 16000, claimed_speaker_id=None)
        
        assert result.available
        assert result.verified
        assert result.matched_speaker_id == "speaker_a"
        assert result.similarity >= 0.99
    
    @patch('app.services.speaker_verification._embed')
    def test_list_enrolled_speakers(self, mock_embed):
        """list_enrolled_speakers returns correct IDs."""
        emb = _make_embedding([1, 0])
        mock_embed.return_value = emb
        
        assert list_enrolled_speakers() == []
        
        enroll_speaker("speaker_a", np.zeros(16000), 16000)
        assert list_enrolled_speakers() == ["speaker_a"]
        
        enroll_speaker("speaker_b", np.zeros(16000), 16000)
        speakers = list_enrolled_speakers()
        assert set(speakers) == {"speaker_a", "speaker_b"}


class TestEnrollmentPersistence:
    """Verify enrollment state is maintained correctly."""
    
    def setup_method(self):
        _reset_enrolled_speakers()
    
    def teardown_method(self):
        _reset_enrolled_speakers()
    
    @patch('app.services.speaker_verification._embed')
    def test_get_enrollment_count(self, mock_embed):
        """get_enrollment_count returns correct count."""
        emb = _make_embedding([1, 0])
        mock_embed.return_value = emb
        
        assert get_enrollment_count("test") == 0
        enroll_speaker("test", np.zeros(16000), 16000)
        assert get_enrollment_count("test") == 1
        enroll_speaker("test", np.zeros(16000), 16000)
        assert get_enrollment_count("test") == 2
        enroll_speaker("test", np.zeros(16000), 16000)
        assert get_enrollment_count("test") == 3
    
    def test_get_enrollment_embedding_nonexistent(self):
        """get_enrollment_embedding returns None for unknown speaker."""
        assert get_enrollment_embedding("unknown") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])