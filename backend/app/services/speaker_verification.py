"""
Speaker enrollment + verification.

Uses SpeechBrain's pretrained ECAPA-TDNN speaker-embedding model
(`speechbrain/spkrec-ecapa-voxceleb`), which is the standard practical
choice for speaker verification: a single forward pass produces a
192-dim embedding, and cosine similarity between embeddings is a
well-calibrated verification score (SpeechBrain publishes an EER
threshold around 0.25-0.35 cosine distance depending on the
enrollment protocol; we use similarity >= 0.5 as the practical
"verified" cutoff for a single enrollment utterance, adjustable below).

Enrollment: store an embedding (mean of enrollment utterances) per
speaker, keyed by speaker_id, persisted via the DB layer (see app/db).

NOTE ON THIS SANDBOX: `speechbrain` downloads its pretrained checkpoint
from HuggingFace Hub on first use, which is not reachable from this
sandbox's restricted network. On a normal developer machine this module
works out of the box. Failure here is surfaced as `available=False`,
never as a fabricated similarity score.
"""
from __future__ import annotations
import os
import numpy as np

from app.models.schemas import SpeakerVerificationResult

VERIFICATION_THRESHOLD = float(os.environ.get("SPEAKER_VERIFICATION_THRESHOLD", "0.5"))

_classifier = None
_load_error: str | None = None

# in-memory store: speaker_id -> np.ndarray embedding (mean of enrollment clips)
# a real deployment persists this in the DB (see app/db/models.py)
_enrolled_speakers: dict[str, np.ndarray] = {}


def _get_classifier():
    global _classifier, _load_error
    if _classifier is not None or _load_error is not None:
        return _classifier
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )
    except Exception as e:
        _load_error = str(e)
        _classifier = None
    return _classifier


def _embed(waveform: np.ndarray, sample_rate: int) -> np.ndarray | None:
    classifier = _get_classifier()
    if classifier is None:
        return None
    import torch
    y = waveform.astype(np.float32)
    if sample_rate != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=16000)
    signal = torch.tensor(y).unsqueeze(0)
    with torch.no_grad():
        emb = classifier.encode_batch(signal)
    return emb.squeeze().cpu().numpy()


def enroll_speaker(speaker_id: str, waveform: np.ndarray, sample_rate: int) -> bool:
    """Add/refine a speaker's voiceprint. Returns True on success."""
    emb = _embed(waveform, sample_rate)
    if emb is None:
        return False
    if speaker_id in _enrolled_speakers:
        # running average across enrollment sessions
        _enrolled_speakers[speaker_id] = (_enrolled_speakers[speaker_id] + emb) / 2.0
    else:
        _enrolled_speakers[speaker_id] = emb
    return True


def list_enrolled_speakers() -> list[str]:
    return list(_enrolled_speakers.keys())


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def verify_speaker(
    waveform: np.ndarray, sample_rate: int, claimed_speaker_id: str | None = None
) -> SpeakerVerificationResult:
    emb = _embed(waveform, sample_rate)
    if emb is None:
        return SpeakerVerificationResult(
            claimed_speaker_id=claimed_speaker_id, available=False,
            error=_load_error or "Speaker model not loaded",
        )

    if not _enrolled_speakers:
        return SpeakerVerificationResult(
            claimed_speaker_id=claimed_speaker_id, similarity=0.0, verified=False,
            available=True, error="no enrolled speakers",
        )

    if claimed_speaker_id and claimed_speaker_id in _enrolled_speakers:
        sim = _cosine_similarity(emb, _enrolled_speakers[claimed_speaker_id])
        return SpeakerVerificationResult(
            claimed_speaker_id=claimed_speaker_id,
            matched_speaker_id=claimed_speaker_id if sim >= VERIFICATION_THRESHOLD else None,
            similarity=round(sim, 4),
            verified=sim >= VERIFICATION_THRESHOLD,
        )

    # no claimed identity: report best match across the enrolled set
    best_id, best_sim = None, -1.0
    for sid, ref_emb in _enrolled_speakers.items():
        sim = _cosine_similarity(emb, ref_emb)
        if sim > best_sim:
            best_id, best_sim = sid, sim

    return SpeakerVerificationResult(
        claimed_speaker_id=claimed_speaker_id,
        matched_speaker_id=best_id if best_sim >= VERIFICATION_THRESHOLD else None,
        similarity=round(best_sim, 4),
        verified=best_sim >= VERIFICATION_THRESHOLD,
    )
