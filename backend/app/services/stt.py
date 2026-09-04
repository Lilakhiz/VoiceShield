"""
Speech-to-text + language identification.

Uses OpenAI Whisper, which natively supports English, Hindi, Kannada,
Telugu, Tamil (and 90+ other languages) and returns a language
probability alongside the transcript -- so language detection and
transcription come from one real model call, not two separate mocks.

Model size is configurable via WHISPER_MODEL env var. `base` or `small`
are recommended for near-real-time CPU inference on short (~5s) chunks;
`tiny` is fastest but less accurate for Indic languages.

NOTE ON THIS SANDBOX: loading a Whisper model the first time downloads
weights from OpenAI's CDN, which is not on this sandbox's network
allow-list. This module is written to run correctly on a normal
developer machine with internet access; here it will raise a clear,
caught exception rather than silently returning fake text.
"""
from __future__ import annotations
import os
import numpy as np

from app.models.schemas import LanguageResult

WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")

_model = None
_load_error: str | None = None

# Whisper's ISO codes map directly for our target set
SUPPORTED_LANGS = {"en", "hi", "kn", "te", "ta"}

# Minimum audio duration (seconds) for reliable language detection
# Whisper needs at least ~1s for language ID; we use 1.5s for safety
MIN_AUDIO_FOR_LANG_DETECT = 1.5
# Whisper's expected mel frames for 30s audio
WHISPER_N_FRAMES = 3000


def _get_model():
    global _model, _load_error
    if _model is not None or _load_error is not None:
        return _model
    try:
        import whisper
        _model = whisper.load_model(WHISPER_MODEL_NAME)
    except Exception as e:
        _load_error = str(e)
        _model = None
    return _model


def _prepare_audio_for_whisper(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Prepare audio for Whisper without unnecessary 30-second padding.
    
    For short utterances (< 1.5s), pad to MIN_AUDIO_FOR_LANG_DETECT for
    reliable language detection. For longer audio, use as-is (Whisper
    handles variable-length input via the mel spectrogram).
    
    Args:
        waveform: Mono float32 audio at 16kHz
        sample_rate: Sample rate (must be 16000)
        
    Returns:
        Prepared audio array
    """
    y = waveform.astype(np.float32)
    if sample_rate != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=16000)
    
    # Only pad if too short for reliable language detection
    min_samples = int(MIN_AUDIO_FOR_LANG_DETECT * 16000)
    if len(y) < min_samples:
        # Pad with silence to minimum duration
        y = np.pad(y, (0, min_samples - len(y)), mode='constant')
    
    return y


def transcribe_and_detect_language(waveform: np.ndarray, sample_rate: int) -> LanguageResult:
    model = _get_model()
    if model is None:
        return LanguageResult(
            language="unknown", confidence=0.0, transcript="",
            available=False,
            error=_load_error or "Whisper model not loaded",
        )

    try:
        import whisper
        
        # Prepare audio - minimal padding only when needed
        audio = _prepare_audio_for_whisper(waveform, sample_rate)
        
        # Compute log-mel spectrogram from the prepared audio
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # Language detection on the actual speech content (not padded to 30s)
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)
        confidence = float(probs[lang])
        
        # Transcription with detected language
        options = whisper.DecodingOptions(language=lang, fp16=False)
        result = whisper.decode(model, mel, options)
        
        return LanguageResult(
            language=lang,
            confidence=round(confidence, 4),
            transcript=result.text.strip(),
        )
    except Exception as e:
        return LanguageResult(language="unknown", confidence=0.0, transcript="",
                               available=False, error=str(e))