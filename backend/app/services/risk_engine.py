"""
Context + Risk Engine
======================
This is the differentiator of the project: a TRUST CONTINUITY SCORE that
evolves across a call rather than a single point-in-time risk number.

Design principles:
 - Every score change must be explainable (a list of RiskReason deltas).
 - The engine never invents numbers: if an upstream signal is unavailable
   (`available=False`), it is excluded from scoring rather than defaulted
   to a fake safe/unsafe value.
 - Trust decays faster than it recovers (asymmetric trust dynamics: it is
   easy to lose trust quickly on a strong fraud signal, hard to regain it
   within the same call), matching real fraud-analyst intuition.
 - The score is a weighted combination of independent evidence streams,
   clipped to [0, 100], with an exponential moving average to avoid
   jitter from single noisy frames.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from app.models.schemas import (
    RiskLevel, RiskReason, RiskSnapshot, LanguageResult,
    SpeakerVerificationResult, DeepfakeResult, ReplayResult,
    ThreatNlpResult, SensitiveRequestType,
)

# ---- Weights (sum roughly normalized to a 0-100 scale contributions) ----
W_SPEAKER_MISMATCH = 35      # claimed identity not verified
W_DEEPFAKE = 30              # AI-generated voice probability
W_REPLAY = 20                # replayed/recorded audio probability
W_SENSITIVE_BASE = 15        # any single sensitive-request type
W_FINANCIAL_EXTRA = 10       # extra bump for money/UPI/transfer requests
W_URGENCY = 10               # manipulative urgency language
W_MULTI_SENSITIVE_STACK = 8  # extra penalty per additional sensitive type in same window

EMA_ALPHA = 0.55             # smoothing factor for new evidence vs. history
TRUST_RECOVERY_CAP = 6       # max points trust can climb per snapshot (asymmetric recovery)
TRUST_DECAY_UNCAP = 100      # trust can fall arbitrarily fast (no cap) on strong evidence


@dataclass
class CallState:
    call_id: str
    started_at: datetime
    trust_score: float = 90.0          # calls start "innocent until proven otherwise"
    history: list[RiskSnapshot] = field(default_factory=list)
    seen_sensitive_types: set = field(default_factory=set)
    claimed_speaker_id: Optional[str] = None


class RiskEngine:
    def __init__(self):
        self._calls: dict[str, CallState] = {}

    def start_call(self, call_id: str, claimed_speaker_id: Optional[str] = None) -> CallState:
        state = CallState(call_id=call_id, started_at=datetime.utcnow(),
                           claimed_speaker_id=claimed_speaker_id)
        self._calls[call_id] = state
        return state

    def get_call(self, call_id: str) -> Optional[CallState]:
        return self._calls.get(call_id)

    def evaluate(
        self,
        call_id: str,
        *,
        language: Optional[LanguageResult] = None,
        speaker: Optional[SpeakerVerificationResult] = None,
        deepfake: Optional[DeepfakeResult] = None,
        replay: Optional[ReplayResult] = None,
        threat: Optional[ThreatNlpResult] = None,
    ) -> RiskSnapshot:
        state = self._calls.get(call_id)
        if state is None:
            state = self.start_call(call_id)

        reasons: list[RiskReason] = []
        raw_risk = 0.0  # 0..100, higher = more dangerous, built up this snapshot

        # --- Speaker identity ---
        if speaker is not None and speaker.available:
            if state.claimed_speaker_id and not speaker.verified:
                # Clamp similarity to [0, 1] for risk calculation.
                # Raw similarity is preserved in detail for debuggability.
                effective_similarity = max(0.0, min(1.0, speaker.similarity))
                contribution = W_SPEAKER_MISMATCH * (1 - effective_similarity)
                raw_risk += contribution
                reasons.append(RiskReason(
                    factor="speaker_mismatch",
                    detail=f"Claimed identity not confirmed (similarity={speaker.similarity:.2f}, effective={effective_similarity:.2f})",
                    delta=-contribution,
                ))
            elif speaker.verified:
                reasons.append(RiskReason(
                    factor="speaker_verified",
                    detail=f"Voice matched enrolled speaker (similarity={speaker.similarity:.2f})",
                    delta=+4,
                ))

        # --- Deepfake / AI-generated voice ---
        if deepfake is not None and deepfake.available:
            contribution = W_DEEPFAKE * deepfake.ai_generated_probability
            raw_risk += contribution
            if deepfake.ai_generated_probability >= 0.5:
                reasons.append(RiskReason(
                    factor="ai_voice_probability",
                    detail=f"AI-generated voice probability {deepfake.ai_generated_probability:.0%} ({deepfake.method})",
                    delta=-contribution,
                ))

        # --- Replay / recorded audio ---
        if replay is not None and replay.available:
            contribution = W_REPLAY * replay.replay_probability
            raw_risk += contribution
            if replay.replay_probability >= 0.5:
                reasons.append(RiskReason(
                    factor="replay_probability",
                    detail=f"Replayed/recorded audio probability {replay.replay_probability:.0%} ({replay.method})",
                    delta=-contribution,
                ))

        # --- Sensitive / threat NLP ---
        if threat is not None and threat.available:
            new_types = [t for t in threat.detected_types if t != SensitiveRequestType.NONE]
            for t in new_types:
                is_financial = t in (
                    SensitiveRequestType.UPI_PAYMENT,
                    SensitiveRequestType.MONEY_TRANSFER,
                    SensitiveRequestType.URGENT_FINANCIAL,
                    SensitiveRequestType.CVV_CARD,
                )
                # extra penalty if this is an additional distinct sensitive
                # type stacking on top of ones already seen this call
                already_seen = t.value in state.seen_sensitive_types
                stack_penalty = W_MULTI_SENSITIVE_STACK if (already_seen is False and state.seen_sensitive_types) else 0
                contribution = W_SENSITIVE_BASE + (W_FINANCIAL_EXTRA if is_financial else 0) + stack_penalty
                raw_risk += contribution
                state.seen_sensitive_types.add(t.value)
                reasons.append(RiskReason(
                    factor=f"sensitive_request:{t.value}",
                    detail=f"Caller requested {t.value.replace('_', ' ').title()}",
                    delta=-contribution,
                ))
            if threat.urgency_score > 0.4:
                contribution = W_URGENCY * threat.urgency_score
                raw_risk += contribution
                reasons.append(RiskReason(
                    factor="urgency_language",
                    detail=f"Urgency/pressure language detected ({threat.urgency_score:.0%})",
                    delta=-contribution,
                ))

        raw_risk = max(0.0, min(100.0, raw_risk))

        # --- Update trust continuity (EMA + asymmetric recovery/decay) ---
        target_trust = 100.0 - raw_risk
        prev_trust = state.trust_score
        if target_trust < prev_trust:
            # Trust decay: no cap, EMA toward target (fast response to new risk)
            new_trust = prev_trust + EMA_ALPHA * (target_trust - prev_trust)
        else:
            # Trust recovery: capped at TRUST_RECOVERY_CAP per snapshot
            # EMA step toward target
            ema_step = EMA_ALPHA * (target_trust - prev_trust)
            # Actual recovery is the smaller of EMA step and the hard cap
            recovery = min(ema_step, TRUST_RECOVERY_CAP)
            new_trust = prev_trust + recovery
        # Hard bounds
        new_trust = max(0.0, min(100.0, new_trust))
        state.trust_score = new_trust

        risk_score = int(round(100 - new_trust))
        level = self._level_for(risk_score)

        snapshot = RiskSnapshot(
            timestamp=datetime.utcnow(),
            elapsed_seconds=(datetime.utcnow() - state.started_at).total_seconds(),
            risk_score=risk_score,
            risk_level=level,
            trust_score=int(round(new_trust)),
            reasons=reasons,
            language=language,
            speaker=speaker,
            deepfake=deepfake,
            replay=replay,
            threat=threat,
            suggest_safe_verification=risk_score >= 60,
        )
        state.history.append(snapshot)
        return snapshot

    @staticmethod
    def _level_for(score: int) -> RiskLevel:
        if score >= 80:
            return RiskLevel.CRITICAL
        if score >= 55:
            return RiskLevel.HIGH
        if score >= 30:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


# Module-level singleton used by the FastAPI app
risk_engine = RiskEngine()
