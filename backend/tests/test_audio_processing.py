"""
Tests for audio processing: validation, VAD, buffering, and STT segmentation.
"""
import numpy as np
import pytest

from app.services.audio_processing import (
    validate_audio_chunk,
    SimpleVAD,
    AudioBuffer,
    SpeechSegment,
    AudioValidationError,
    TARGET_SAMPLE_RATE,
)


class TestAudioValidation:
    """Tests for audio chunk validation."""
    
    def test_valid_audio(self):
        """Valid audio passes validation."""
        audio = np.random.randn(1600).astype(np.float32) * 0.1
        result = validate_audio_chunk(audio, TARGET_SAMPLE_RATE, 1)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (1600,)
    
    def test_valid_audio_list_input(self):
        """Valid audio as list passes validation."""
        audio = (np.random.randn(1600) * 0.1).tolist()
        result = validate_audio_chunk(audio, TARGET_SAMPLE_RATE, 1)
        assert isinstance(result, np.ndarray)
    
    def test_empty_audio_rejected(self):
        """Empty audio is rejected."""
        audio = np.array([], dtype=np.float32)
        with pytest.raises(AudioValidationError, match="Empty audio chunk"):
            validate_audio_chunk(audio, TARGET_SAMPLE_RATE, 1)
    
    def test_oversized_audio_rejected(self):
        """Audio exceeding max chunk size is rejected."""
        audio = np.random.randn(10000).astype(np.float32)
        with pytest.raises(AudioValidationError, match="Chunk too large"):
            validate_audio_chunk(audio, TARGET_SAMPLE_RATE, 1)
    
    def test_undersized_audio_rejected(self):
        """Audio below min chunk size is rejected."""
        audio = np.random.randn(50).astype(np.float32)
        with pytest.raises(AudioValidationError, match="Chunk too small"):
            validate_audio_chunk(audio, TARGET_SAMPLE_RATE, 1)
    
    def test_nan_audio_rejected(self):
        """Audio with NaN is rejected."""
        audio = np.full(200, 0.1, dtype=np.float32)
        audio[50] = np.nan
        with pytest.raises(AudioValidationError, match="NaN or Inf"):
            validate_audio_chunk(audio, TARGET_SAMPLE_RATE, 1)
    
    def test_inf_audio_rejected(self):
        """Audio with Inf is rejected."""
        audio = np.full(200, 0.1, dtype=np.float32)
        audio[50] = np.inf
        with pytest.raises(AudioValidationError, match="NaN or Inf"):
            validate_audio_chunk(audio, TARGET_SAMPLE_RATE, 1)
    
    def test_negative_inf_audio_rejected(self):
        """Audio with -Inf is rejected."""
        audio = np.full(200, 0.1, dtype=np.float32)
        audio[50] = -np.inf
        with pytest.raises(AudioValidationError, match="NaN or Inf"):
            validate_audio_chunk(audio, TARGET_SAMPLE_RATE, 1)
    
    def test_out_of_range_audio_rejected(self):
        """Audio exceeding [-1, 1] range is rejected."""
        audio = np.full(200, 0.1, dtype=np.float32)
        audio[50] = 1.5
        with pytest.raises(AudioValidationError, match="amplitude exceeds"):
            validate_audio_chunk(audio, TARGET_SAMPLE_RATE, 1)
    
    def test_wrong_sample_rate_rejected(self):
        """Wrong sample rate is rejected."""
        audio = np.random.randn(1600).astype(np.float32) * 0.1
        with pytest.raises(AudioValidationError, match="Unsupported sample rate"):
            validate_audio_chunk(audio, 8000, 1)
    
    def test_wrong_channels_rejected(self):
        """Wrong channel count is rejected."""
        audio = np.random.randn(1600).astype(np.float32) * 0.1
        with pytest.raises(AudioValidationError, match="Unsupported channel count"):
            validate_audio_chunk(audio, TARGET_SAMPLE_RATE, 2)
    
    def test_2d_audio_rejected(self):
        """2D audio is rejected."""
        audio = np.random.randn(1600, 2).astype(np.float32) * 0.1
        with pytest.raises(AudioValidationError, match="must be 1D"):
            validate_audio_chunk(audio, TARGET_SAMPLE_RATE, 1)
    
    def test_none_audio_rejected(self):
        """None audio is rejected."""
        with pytest.raises(AudioValidationError, match="Missing audio data"):
            validate_audio_chunk(None, TARGET_SAMPLE_RATE, 1)


class TestSimpleVAD:
    """Tests for SimpleVAD voice activity detection."""
    
    def test_silence_is_not_speech(self):
        """Silence frames are not detected as speech."""
        vad = SimpleVAD()
        silence = np.zeros(480, dtype=np.float32)
        assert vad.is_speech(silence) is False
    
    def test_pure_tone_is_speech(self):
        """A pure tone (tonal) is detected as speech."""
        vad = SimpleVAD()
        t = np.linspace(0, 0.03, 480)
        tone = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        assert vad.is_speech(tone) is True
    
    def test_noise_is_not_speech(self):
        """White noise (high flatness) is not detected as speech."""
        vad = SimpleVAD()
        noise = np.random.randn(480).astype(np.float32) * 0.01
        # Low energy noise should not be speech
        assert vad.is_speech(noise) is False
    
    def test_high_energy_noise_may_be_speech(self):
        """High energy noise might be detected as speech (energy-based)."""
        vad = SimpleVAD(energy_threshold=0.001)
        noise = np.random.randn(480).astype(np.float32) * 0.1
        # High energy noise - depends on spectral flatness
        result = vad.is_speech(noise)
        # Just verify it doesn't crash
        assert isinstance(result, bool)
    
    def test_process_audio_returns_list(self):
        """process_audio returns list of booleans."""
        vad = SimpleVAD()
        audio = np.random.randn(4800).astype(np.float32) * 0.1
        results = vad.process_audio(audio)
        assert isinstance(results, list)
        assert len(results) == 10  # 4800 / 480 = 10 frames
        assert all(isinstance(r, bool) for r in results)


class TestAudioBuffer:
    """Tests for AudioBuffer speech segmentation."""
    
    def _make_speech_frame(self, duration_ms=30, freq=440, amplitude=0.3):
        """Create a speech-like frame (tonal)."""
        sr = TARGET_SAMPLE_RATE
        n = int(sr * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, n)
        return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    
    def _make_silence_frame(self, duration_ms=30):
        """Create a silence frame."""
        sr = TARGET_SAMPLE_RATE
        n = int(sr * duration_ms / 1000)
        return np.zeros(n, dtype=np.float32)
    
    def test_silence_only_yields_no_segments(self):
        """Pure silence yields no speech segments."""
        buffer = AudioBuffer()
        silence = self._make_silence_frame(300)  # 300ms
        segments = buffer.add_audio(silence)
        assert segments == []
    
    def test_short_speech_below_min_duration_yields_no_segment(self):
        """Speech shorter than minimum duration yields no segment."""
        buffer = AudioBuffer(min_speech_seconds=1.0)  # Set higher than speech+silence
        # 200ms of speech
        speech = self._make_speech_frame(200)
        segments = buffer.add_audio(speech)
        # Followed by silence to trigger endpoint
        silence = self._make_silence_frame(500)
        segments = buffer.add_audio(silence)
        # Total duration ~700ms < 1.0s minimum
        assert segments == []
    
    def test_speech_above_min_duration_yields_segment(self):
        """Speech longer than minimum duration yields a segment."""
        buffer = AudioBuffer(min_speech_seconds=0.3)
        # 500ms of speech
        speech = self._make_speech_frame(500)
        segments = buffer.add_audio(speech)
        # Followed by silence to trigger endpoint
        silence = self._make_silence_frame(500)
        segments = buffer.add_audio(silence)
        assert len(segments) == 1
        assert isinstance(segments[0], SpeechSegment)
        assert segments[0].duration >= 0.3
    
    def test_multiple_speech_segments(self):
        """Multiple separate utterances yield multiple segments."""
        buffer = AudioBuffer(min_speech_seconds=0.2, max_silence_frames=10)
        
        # First utterance
        speech1 = self._make_speech_frame(400)
        segments = buffer.add_audio(speech1)
        silence1 = self._make_silence_frame(400)
        segments = buffer.add_audio(silence1)
        assert len(segments) == 1
        
        # Second utterance
        speech2 = self._make_speech_frame(400)
        segments = buffer.add_audio(speech2)
        silence2 = self._make_silence_frame(400)
        segments = buffer.add_audio(silence2)
        assert len(segments) == 1  # Only the second segment returned now
    
    def test_max_speech_duration_forces_segment(self):
        """Speech exceeding max duration forces segment completion."""
        buffer = AudioBuffer(max_speech_seconds=1.0, min_speech_seconds=0.2)
        # 1.5 seconds of continuous speech
        speech = self._make_speech_frame(1500)
        segments = buffer.add_audio(speech)
        # Should have forced a segment at 1.0s
        assert len(segments) >= 1
    
    def test_flush_returns_remaining_speech(self):
        """Flush returns remaining speech on call end."""
        buffer = AudioBuffer(min_speech_seconds=0.2)
        speech = self._make_speech_frame(500)
        buffer.add_audio(speech)
        # No trailing silence - flush should return it
        segment = buffer.flush()
        assert segment is not None
        assert segment.duration >= 0.2
    
    def test_flush_returns_none_for_short_speech(self):
        """Flush returns None for speech below minimum duration."""
        buffer = AudioBuffer(min_speech_seconds=0.5)
        speech = self._make_speech_frame(200)
        buffer.add_audio(speech)
        segment = buffer.flush()
        assert segment is None
    
    def test_interleaved_chunks(self):
        """Audio added in small chunks works correctly."""
        buffer = AudioBuffer(min_speech_seconds=0.3)
        speech = self._make_speech_frame(600)
        # Split into 6 chunks of 100ms each
        chunks = np.array_split(speech, 6)
        segments = []
        for chunk in chunks:
            segments.extend(buffer.add_audio(chunk))
        # Add silence to trigger endpoint
        silence = self._make_silence_frame(500)
        segments.extend(buffer.add_audio(silence))
        assert len(segments) == 1


class TestSpeechSegment:
    """Tests for SpeechSegment dataclass."""
    
    def test_segment_creation(self):
        """SpeechSegment can be created with required fields."""
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        segment = SpeechSegment(
            audio=audio,
            start_time=1.5,
            duration=1.0,
            sample_rate=16000
        )
        assert segment.audio.shape == (16000,)
        assert segment.start_time == 1.5
        assert segment.duration == 1.0
        assert segment.sample_rate == 16000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])