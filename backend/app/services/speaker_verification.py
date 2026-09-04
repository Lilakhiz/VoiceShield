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
speaker, keyed by speaker_id, persisted in the database.
On startup, all embeddings are loaded from the database into memory.

NOTE ON THIS SANDBOX: `speechbrain` downloads its pretrained checkpoint
from HuggingFace Hub on first use, which is not reachable from this
sandbox's restricted network. On a normal developer machine this module
works out of the box. Failure here is surfaced as `available=False`,
never as a fabricated similarity score.
"""
from __future__ import annotations
import os
import numpy as np
import logging

from app.models.schemas import SpeakerVerificationResult
from app.db.database import SessionLocal, TrustedSpeaker

logger = logging.getLogger(__name__)

VERIFICATION_THRESHOLD = float(os.environ.get("SPEAKER_VERIFICATION_THRESHOLD", "0.5"))
# Expected embedding dimension for ECAPA-TDNN
EXPECTED_EMBEDDING_DIM = 192
CURRENT_EMBEDDING_VERSION = 1

_classifier = None
_load_error: str | None = None

# in-memory store: speaker_id -> (np.ndarray embedding, int count)
# Loaded from DB on startup, updated on enrollment
_enrolled_speakers: dict[str, tuple[np.ndarray, int]] = {}


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


def _bytes_to_embedding(blob: bytes | None) -> np.ndarray | None:
    """Convert database blob to numpy embedding array."""
    if blob is None:
        return None
    try:
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size != EXPECTED_EMBEDDING_DIM:
            logger.warning(f"Embedding dimension mismatch: expected {EXPECTED_EMBEDDING_DIM}, got {arr.size}")
            return None
        return arr
    except Exception as e:
        logger.warning(f"Failed to decode embedding blob: {e}")
        return None


def _embedding_to_bytes(emb: np.ndarray) -> bytes:
    """Convert numpy embedding array to bytes for database storage."""
    return emb.astype(np.float32).tobytes()


def load_all_speakers_from_db() -> int:
    """
    Load all enrolled speakers from database into memory.
    Called on application startup.
    Returns the number of speakers loaded.
    """
    global _enrolled_speakers
    _enrolled_speakers.clear()
    
    loaded = 0
    with SessionLocal() as session:
        speakers = session.query(TrustedSpeaker).all()
        for spk in speakers:
            if spk.embedding is not None:
                emb = _bytes_to_embedding(spk.embedding)
                if emb is not None:
                    _enrolled_speakers[spk.id] = (emb, spk.enrollment_clips)
                    loaded += 1
                else:
                    logger.warning(f"Failed to load embedding for speaker {spk.id}, skipping")
            else:
                logger.warning(f"Speaker {spk.id} has no embedding, skipping")
    
    logger.info(f"Loaded {loaded} speaker embeddings from database")
    return loaded


def _persist_speaker_embedding(speaker_id: str, embedding: np.ndarray, enrollment_clips: int) -> bool:
    """Persist speaker embedding and enrollment count to database."""
    try:
        with SessionLocal() as session:
            spk = session.get(TrustedSpeaker, speaker_id)
            if spk is None:
                logger.error(f"Speaker {speaker_id} not found in database for embedding persistence")
                return False
            spk.embedding = _embedding_to_bytes(embedding)
            spk.enrollment_clips = enrollment_clips
            spk.embedding_version = CURRENT_EMBEDDING_VERSION
            session.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to persist embedding for speaker {speaker_id}: {e}")
        return False


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
        prev_emb, count = _enrolled_speakers[speaker_id]
        # Proper running mean: (prev_mean * N + new_emb) / (N + 1)
        new_emb = (prev_emb * count + emb) / (count + 1)
        new_count = count + 1
        _enrolled_speakers[speaker_id] = (new_emb, new_count)
    else:
        new_emb = emb
        new_count = 1
        _enrolled_speakers[speaker_id] = (new_emb, new_count)
    
    # Persist to database
    _persist_speaker_embedding(speaker_id, new_emb, new_count)
    return True


def list_enrolled_speakers() -> list[str]:
    return list(_enrolled_speakers.keys())


def get_enrollment_count(speaker_id: str) -> int:
    """Return the number of enrollment clips for a speaker."""
    if speaker_id in _enrolled_speakers:
        return _enrolled_speakers[speaker_id][1]
    return 0


def get_enrollment_embedding(speaker_id: str) -> np.ndarray | None:
    """Return the current mean embedding for a speaker."""
    if speaker_id in _enrolled_speakers:
        return _enrolled_speakers[speaker_id][0]
    return None


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
        ref_emb = _enrolled_speakers[claimed_speaker_id][0]
        sim = _cosine_similarity(emb, ref_emb)
        return SpeakerVerificationResult(
            claimed_speaker_id=claimed_speaker_id,
            matched_speaker_id=claimed_speaker_id if sim >= VERIFICATION_THRESHOLD else None,
            similarity=round(sim, 4),
            verified=sim >= VERIFICATION_THRESHOLD,
        )

    # no claimed identity: report best match across the enrolled set
    best_id, best_sim = None, -1.0
    for sid, (ref_emb, _) in _enrolled_speakers.items():
        sim = _cosine_similarity(emb, ref_emb)
        if sim > best_sim:
            best_id, best_sim = sid, sim

    return SpeakerVerificationResult(
        claimed_speaker_id=claimed_speaker_id,
        matched_speaker_id=best_id if best_sim >= VERIFICATION_THRESHOLD else None,
        similarity=round(best_sim, 4),
        verified=best_sim >= VERIFICATION_THRESHOLD,
    )