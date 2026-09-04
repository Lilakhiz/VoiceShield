"""
Audio processing utilities: validation, VAD, buffering, and STT segmentation.
"""
from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Generator
import logging

logger = logging.getLogger(__name__)

# Audio configuration
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
MAX_CHUNK_SAMPLES = 8000  # 500ms at 16kHz
MIN_CHUNK_SAMPLES = 100   # Minimum valid chunk
MAX_BUFFER_SECONDS = 30   # Maximum buffer duration
VAD_FRAME_MS = 30         # VAD frame size in ms
VAD_FRAME_SAMPLES = int(TARGET_SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480 samples
MIN_SPEECH_FRAMES = 3     # Minimum frames to consider as speech
MAX_SILENCE_FRAMES = 15   # Max silence frames before ending segment (~450ms)
MAX_SPEECH_SECONDS = 10   # Maximum speech segment duration before forced STT
MIN_SPEECH_SECONDS = 0.5  # Minimum speech duration to trigger STT


class AudioValidationError(Exception):
    """Raised when audio validation fails."""
    pass


def validate_audio_chunk(
    pcm_data: list | np.ndarray,
    sample_rate: int,
    channels: int = 1
) -> np.ndarray:
    """
    Validate and normalize an incoming audio chunk.
    
    Args:
        pcm_data: Raw PCM samples as list or array
        sample_rate: Sample rate in Hz
        channels: Number of channels
        
    Returns:
        Validated mono float32 numpy array at target sample rate
        
    Raises:
        AudioValidationError: If validation fails
    """
    # Type check
    if pcm_data is None:
        raise AudioValidationError("Missing audio data")
    
    # Convert to numpy array
    try:
        if isinstance(pcm_data, list):
            waveform = np.asarray(pcm_data, dtype=np.float32)
        elif isinstance(pcm_data, np.ndarray):
            waveform = pcm_data.astype(np.float32)
        else:
            raise AudioValidationError(f"Invalid audio data type: {type(pcm_data)}")
    except (ValueError, TypeError) as e:
        raise AudioValidationError(f"Cannot convert audio data: {e}")
    
    # Shape validation - must be 1D
    if waveform.ndim != 1:
        raise AudioValidationError(f"Audio must be 1D, got shape {waveform.shape}")
    
    # Size validation
    n_samples = waveform.size
    if n_samples == 0:
        raise AudioValidationError("Empty audio chunk")
    if n_samples > MAX_CHUNK_SAMPLES:
        raise AudioValidationError(f"Chunk too large: {n_samples} samples (max {MAX_CHUNK_SAMPLES})")
    if n_samples < MIN_CHUNK_SAMPLES:
        raise AudioValidationError(f"Chunk too small: {n_samples} samples (min {MIN_CHUNK_SAMPLES})")
    
    # Numeric validation
    if not np.isfinite(waveform).all():
        raise AudioValidationError("Audio contains NaN or Inf values")
    
    # Range validation - Float32 PCM should be in [-1, 1]
    max_abs = np.max(np.abs(waveform))
    if max_abs > 1.0 + 1e-5:
        raise AudioValidationError(f"Audio amplitude exceeds [-1, 1]: max={max_abs:.4f}")
    
    # Sample rate validation
    if sample_rate != TARGET_SAMPLE_RATE:
        raise AudioValidationError(
            f"Unsupported sample rate: {sample_rate} Hz (expected {TARGET_SAMPLE_RATE} Hz)"
        )
    
    # Channel validation
    if channels != TARGET_CHANNELS:
        raise AudioValidationError(
            f"Unsupported channel count: {channels} (expected {TARGET_CHANNELS})"
        )
    
    return waveform


class SimpleVAD:
    """
    Simple energy-based Voice Activity Detection.
    Lightweight, no ML model required, runs on CPU.
    """
    
    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        frame_ms: int = VAD_FRAME_MS,
        energy_threshold: float = 0.005,
        spectral_flatness_threshold: float = 0.5
    ):
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.energy_threshold = energy_threshold
        self.spectral_flatness_threshold = spectral_flatness_threshold
        
    def _compute_frame_features(self, frame: np.ndarray) -> tuple[float, float]:
        """Compute energy and spectral flatness for a frame."""
        if frame.size == 0:
            return 0.0, 1.0
        
        # Energy (RMS)
        energy = float(np.sqrt(np.mean(frame ** 2) + 1e-10))
        
        # Spectral flatness (ratio of geometric mean to arithmetic mean of power spectrum)
        # High flatness = noise-like, Low flatness = tonal (speech)
        if frame.size >= 8:
            fft = np.abs(np.fft.rfft(frame))
            fft = fft + 1e-10
            geo_mean = np.exp(np.mean(np.log(fft)))
            arith_mean = np.mean(fft)
            flatness = float(geo_mean / arith_mean)
        else:
            flatness = 1.0
            
        return energy, flatness
    
    def is_speech(self, frame: np.ndarray) -> bool:
        """
        Determine if a frame contains speech.
        
        Returns True if energy is above threshold AND spectral flatness
        is below threshold (indicating tonal content like speech).
        """
        energy, flatness = self._compute_frame_features(frame)
        return energy > self.energy_threshold and flatness < self.spectral_flatness_threshold
    
    def process_audio(self, audio: np.ndarray) -> list[bool]:
        """
        Process audio and return VAD decision for each frame.
        
        Returns list of booleans, one per frame.
        """
        n_frames = len(audio) // self.frame_samples
        results = []
        
        for i in range(n_frames):
            start = i * self.frame_samples
            end = start + self.frame_samples
            frame = audio[start:end]
            if len(frame) == self.frame_samples:
                results.append(self.is_speech(frame))
            else:
                # Partial frame at end - pad and check
                padded = np.pad(frame, (0, self.frame_samples - len(frame)))
                results.append(self.is_speech(padded))
                
        return results


@dataclass
class SpeechSegment:
    """Represents a detected speech segment ready for STT."""
    audio: np.ndarray
    start_time: float  # seconds from call start
    duration: float
    sample_rate: int = TARGET_SAMPLE_RATE


class AudioBuffer:
    """
    Bounded audio buffer with VAD-based speech segmentation.
    
    Accumulates incoming audio chunks, runs VAD, and yields
    complete speech segments for STT processing.
    """
    
    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        max_buffer_seconds: float = MAX_BUFFER_SECONDS,
        vad: Optional[SimpleVAD] = None,
        min_speech_seconds: float = MIN_SPEECH_SECONDS,
        max_speech_seconds: float = MAX_SPEECH_SECONDS,
        max_silence_frames: int = MAX_SILENCE_FRAMES
    ):
        self.sample_rate = sample_rate
        self.max_samples = int(max_buffer_seconds * sample_rate)
        self.vad = vad or SimpleVAD(sample_rate)
        self.min_speech_samples = int(min_speech_seconds * sample_rate)
        self.max_speech_samples = int(max_speech_seconds * sample_rate)
        self.max_silence_frames = max_silence_frames
        
        # Ring buffer for incoming audio
        self._buffer = deque(maxlen=self.max_samples)
        self._total_samples = 0
        
        # Speech detection state
        self._in_speech = False
        self._speech_buffer = []
        self._speech_start_sample = 0
        self._silence_frames = 0
        self._frame_samples = self.vad.frame_samples
        
        # Carryover for partial frames at chunk boundaries
        self._carryover = np.array([], dtype=np.float32)
        
    def add_audio(self, audio: np.ndarray) -> list[SpeechSegment]:
        """
        Add audio to buffer and return any completed speech segments.
        
        Args:
            audio: Validated mono float32 audio at target sample rate
            
        Returns:
            List of SpeechSegment objects ready for STT
        """
        # Prepend carryover from previous chunk
        if self._carryover.size > 0:
            audio = np.concatenate([self._carryover, audio])
            self._carryover = np.array([], dtype=np.float32)
        
        # Add to ring buffer
        self._buffer.extend(audio)
        self._total_samples += len(audio)
        
        segments = []
        
        # Process in VAD frames
        n_frames = len(audio) // self._frame_samples
        for i in range(n_frames):
            start = i * self._frame_samples
            end = start + self._frame_samples
            frame = audio[start:end]
            
            is_speech = self.vad.is_speech(frame)
            
            if is_speech:
                if not self._in_speech:
                    # Speech start
                    self._in_speech = True
                    self._speech_start_sample = self._total_samples - len(audio) + start
                    self._speech_buffer = []
                    self._silence_frames = 0
                
                self._speech_buffer.append(frame)
                self._silence_frames = 0
                
                # Check max speech duration
                speech_samples = sum(len(f) for f in self._speech_buffer)
                if speech_samples >= self.max_speech_samples:
                    # Force segment completion
                    segment = self._finalize_segment()
                    if segment:
                        segments.append(segment)
                        
            else:
                if self._in_speech:
                    self._silence_frames += 1
                    self._speech_buffer.append(frame)
                    
                    # Check for speech end (silence timeout)
                    if self._silence_frames >= self.max_silence_frames:
                        segment = self._finalize_segment()
                        if segment:
                            segments.append(segment)
        
        # Save carryover for next chunk
        remaining = len(audio) % self._frame_samples
        if remaining > 0:
            self._carryover = audio[-remaining:].copy()
        
        return segments
    
    def _finalize_segment(self) -> Optional[SpeechSegment]:
        """Finalize current speech segment and return it."""
        if not self._speech_buffer:
            self._in_speech = False
            return None
            
        # Concatenate speech frames
        speech_audio = np.concatenate(self._speech_buffer)
        
        # Check minimum duration
        if len(speech_audio) < self.min_speech_samples:
            self._in_speech = False
            self._speech_buffer = []
            self._silence_frames = 0
            return None
        
        # Create segment
        duration = len(speech_audio) / self.sample_rate
        segment = SpeechSegment(
            audio=speech_audio,
            start_time=self._speech_start_sample / self.sample_rate,
            duration=duration,
            sample_rate=self.sample_rate
        )
        
        # Reset state
        self._in_speech = False
        self._speech_buffer = []
        self._silence_frames = 0
        
        return segment
    
    def flush(self) -> Optional[SpeechSegment]:
        """Flush any remaining speech in buffer on call end."""
        if self._in_speech and self._speech_buffer:
            speech_audio = np.concatenate(self._speech_buffer)
            if len(speech_audio) >= self.min_speech_samples:
                duration = len(speech_audio) / self.sample_rate
                return SpeechSegment(
                    audio=speech_audio,
                    start_time=self._speech_start_sample / self.sample_rate,
                    duration=duration,
                    sample_rate=self.sample_rate
                )
        return None
    
    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds."""
        return len(self._buffer) / self.sample_rate