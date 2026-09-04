"""
Focused tests for the trust-continuity update in RiskEngine.evaluate:
EMA smoothing and the recovery cap are separate steps (risk_engine.py),
and this file pins down that separation's observable behavior directly,
independent of the specific signal weights.
"""
from app.services.risk_engine import RiskEngine, EMA_ALPHA, TRUST_RECOVERY_CAP
from app.models.schemas import (
    SpeakerVerificationResult, DeepfakeResult, ReplayResult,
    ThreatNlpResult, SensitiveRequestType,
)


def _clean_signals():
    return dict(
        speaker=SpeakerVerificationResult(claimed_speaker_id="dad", similarity=0.95, verified=True),
        deepfake=DeepfakeResult(ai_generated_probability=0.0, method="test"),
        replay=ReplayResult(replay_probability=0.0, live_probability=1.0, method="test"),
        threat=ThreatNlpResult(detected_types=[SensitiveRequestType.NONE], urgency_score=0.0),
    )


def _extreme_signals():
    return dict(
        speaker=SpeakerVerificationResult(claimed_speaker_id="dad", similarity=0.0, verified=False),
        deepfake=DeepfakeResult(ai_generated_probability=1.0, method="test"),
        replay=ReplayResult(replay_probability=1.0, live_probability=0.0, method="test"),
        threat=ThreatNlpResult(detected_types=[SensitiveRequestType.OTP, SensitiveRequestType.UPI_PAYMENT],
                                urgency_score=1.0),
    )


def test_sudden_risk_drops_trust_fast_and_uncapped():
    engine = RiskEngine()
    call_id = "sudden"
    engine.start_call(call_id, claimed_speaker_id="dad")
    prev = engine.get_call(call_id).trust_score  # 90.0 starting trust
    snap = engine.evaluate(call_id, **_extreme_signals())
    assert snap.trust_score < prev
    assert prev - snap.trust_score > TRUST_RECOVERY_CAP  # decay is not limited by the recovery cap


def test_recovery_is_gradual_not_instant():
    engine = RiskEngine()
    call_id = "gradual"
    engine.start_call(call_id, claimed_speaker_id="dad")
    engine.get_call(call_id).trust_score = 10.0  # simulate trust already damaged
    snap = engine.evaluate(call_id, **_clean_signals())
    assert 10 < snap.trust_score < 100  # recovers, but doesn't jump straight to the clean target


def test_recovery_capped_at_max_points_per_snapshot():
    engine = RiskEngine()
    call_id = "cap"
    engine.start_call(call_id, claimed_speaker_id="dad")
    engine.get_call(call_id).trust_score = 10.0
    snap = engine.evaluate(call_id, **_clean_signals())
    # target_trust=100 here, so EMA alone would propose 10 + EMA_ALPHA*90,
    # far above the cap -- the cap must be the binding constraint.
    assert EMA_ALPHA * (100.0 - 10.0) > TRUST_RECOVERY_CAP
    assert snap.trust_score == round(10.0 + TRUST_RECOVERY_CAP)


def test_repeated_clean_updates_climb_by_at_most_the_cap_each_time():
    engine = RiskEngine()
    call_id = "repeated"
    engine.start_call(call_id, claimed_speaker_id="dad")
    engine.get_call(call_id).trust_score = 10.0
    prev = 10
    for _ in range(5):
        snap = engine.evaluate(call_id, **_clean_signals())
        assert prev <= snap.trust_score <= prev + TRUST_RECOVERY_CAP + 1  # +1 slack for int rounding
        prev = snap.trust_score


def test_trust_stays_within_bounds_at_extremes():
    engine = RiskEngine()
    call_id = "bounds"
    engine.start_call(call_id, claimed_speaker_id="dad")
    state = engine.get_call(call_id)

    state.trust_score = 100.0
    snap = engine.evaluate(call_id, **_clean_signals())
    assert 0 <= snap.trust_score <= 100

    state.trust_score = 0.0
    snap = engine.evaluate(call_id, **_extreme_signals())
    assert 0 <= snap.trust_score <= 100
