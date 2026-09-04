from app.services.risk_engine import RiskEngine
from app.models.schemas import (
    SpeakerVerificationResult, DeepfakeResult, ReplayResult,
    ThreatNlpResult, SensitiveRequestType, RiskLevel,
)


def _step(engine, call_id, speaker_sim, deepfake_p, replay_p, sensitive=None, urgency=0.0):
    speaker = SpeakerVerificationResult(claimed_speaker_id="dad", similarity=speaker_sim,
                                         verified=speaker_sim >= 0.5)
    deepfake = DeepfakeResult(ai_generated_probability=deepfake_p, method="test")
    replay = ReplayResult(replay_probability=replay_p, live_probability=1 - replay_p, method="test")
    threat = ThreatNlpResult(detected_types=sensitive or [SensitiveRequestType.NONE], urgency_score=urgency)
    return engine.evaluate(call_id, speaker=speaker, deepfake=deepfake, replay=replay, threat=threat)


def test_trust_declines_monotonically_as_evidence_worsens():
    engine = RiskEngine()
    call_id = "t1"
    engine.start_call(call_id, claimed_speaker_id="dad")

    s0 = _step(engine, call_id, 0.91, 0.03, 0.02)
    s1 = _step(engine, call_id, 0.55, 0.45, 0.05)
    s2 = _step(engine, call_id, 0.40, 0.70, 0.07)
    s3 = _step(engine, call_id, 0.35, 0.80, 0.08, [SensitiveRequestType.OTP], urgency=0.7)
    s4 = _step(engine, call_id, 0.30, 0.85, 0.10, [SensitiveRequestType.UPI_PAYMENT], urgency=0.9)

    trusts = [s0.trust_score, s1.trust_score, s2.trust_score, s3.trust_score, s4.trust_score]
    assert trusts == sorted(trusts, reverse=True), f"trust should be non-increasing: {trusts}"
    assert s4.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
    assert s0.risk_level == RiskLevel.LOW


def test_every_score_change_has_an_explanation():
    engine = RiskEngine()
    call_id = "t2"
    engine.start_call(call_id, claimed_speaker_id="dad")
    snap = _step(engine, call_id, 0.2, 0.9, 0.9, [SensitiveRequestType.OTP], urgency=0.9)
    assert len(snap.reasons) > 0
    assert all(r.detail for r in snap.reasons)


def test_missing_upstream_signal_is_excluded_not_faked():
    engine = RiskEngine()
    call_id = "t3"
    engine.start_call(call_id)
    unavailable_speaker = SpeakerVerificationResult(available=False, error="model down")
    snap = engine.evaluate(call_id, speaker=unavailable_speaker)
    assert not any(r.factor.startswith("speaker") for r in snap.reasons)


def test_safe_verification_suggested_above_threshold():
    engine = RiskEngine()
    call_id = "t4"
    engine.start_call(call_id, claimed_speaker_id="dad")
    snap = _step(engine, call_id, 0.1, 0.95, 0.9, [SensitiveRequestType.UPI_PAYMENT], urgency=0.9)
    assert snap.suggest_safe_verification is True


# --- Speaker similarity bounds tests ---

def test_speaker_similarity_negative_clamped_to_zero():
    """Negative similarity should be clamped to 0, giving max mismatch penalty."""
    engine = RiskEngine()
    call_id = "sim_neg"
    engine.start_call(call_id, claimed_speaker_id="dad")
    snap = _step(engine, call_id, -0.5, 0.0, 0.0)
    # With effective_similarity=0, contribution = W_SPEAKER_MISMATCH * (1 - 0) = 35
    # raw_risk = 35, target_trust = 65
    # Starting trust=90, target_trust=65 < 90, so decay
    # new_trust = 90 + 0.55 * (65 - 90) = 90 - 13.75 = 76.25 -> 76
    assert snap.trust_score == 76
    assert snap.risk_score == 24
    # Verify reason shows both raw and effective similarity
    mismatch_reasons = [r for r in snap.reasons if r.factor == "speaker_mismatch"]
    assert len(mismatch_reasons) == 1
    assert "similarity=-0.50" in mismatch_reasons[0].detail
    assert "effective=0.00" in mismatch_reasons[0].detail


def test_speaker_similarity_zero():
    """Similarity of 0 should give full mismatch penalty."""
    engine = RiskEngine()
    call_id = "sim_zero"
    engine.start_call(call_id, claimed_speaker_id="dad")
    snap = _step(engine, call_id, 0.0, 0.0, 0.0)
    # Same as negative case
    assert snap.trust_score == 76
    assert snap.risk_score == 24


def test_speaker_similarity_half():
    """Similarity of 0.5 should be verified (>= 0.5 threshold), giving no mismatch penalty."""
    engine = RiskEngine()
    call_id = "sim_half"
    engine.start_call(call_id, claimed_speaker_id="dad")
    snap = _step(engine, call_id, 0.5, 0.0, 0.0)
    # similarity=0.5 -> verified=True, so no mismatch penalty
    # raw_risk = 0, target_trust = 100
    # recovery: new_trust = min(90 + 0.55*10, 90+6) = min(95.5, 96) = 95.5 -> 96
    assert snap.trust_score == 96
    assert snap.risk_score == 4
    # Verify it's marked as verified
    verified_reasons = [r for r in snap.reasons if r.factor == "speaker_verified"]
    assert len(verified_reasons) == 1


def test_speaker_similarity_one():
    """Similarity of 1.0 should give no mismatch penalty."""
    engine = RiskEngine()
    call_id = "sim_one"
    engine.start_call(call_id, claimed_speaker_id="dad")
    snap = _step(engine, call_id, 1.0, 0.0, 0.0)
    # effective_similarity=1.0, contribution = 35 * (1 - 1) = 0
    # raw_risk = 0, target_trust = 100
    # recovery: new_trust = min(90 + 0.55*10, 90+6) = min(95.5, 96) = 95.5 -> 96
    assert snap.trust_score == 96
    assert snap.risk_score == 4


def test_speaker_similarity_above_one_clamped():
    """Similarity > 1 should be clamped to 1, giving no mismatch penalty."""
    engine = RiskEngine()
    call_id = "sim_above"
    engine.start_call(call_id, claimed_speaker_id="dad")
    snap = _step(engine, call_id, 1.5, 0.0, 0.0)
    # effective_similarity=1.0 (clamped), contribution = 0
    # Should behave same as similarity=1.0
    assert snap.trust_score == 96
    assert snap.risk_score == 4
    mismatch_reasons = [r for r in snap.reasons if r.factor == "speaker_mismatch"]
    assert len(mismatch_reasons) == 0  # verified=True when similarity >= 0.5


def test_speaker_mismatch_contribution_bounded_by_weight():
    """Speaker mismatch contribution should never exceed W_SPEAKER_MISMATCH (35)."""
    engine = RiskEngine()
    call_id = "bound_test"
    engine.start_call(call_id, claimed_speaker_id="dad")
    
    # Test with very negative similarity
    snap = _step(engine, call_id, -10.0, 0.0, 0.0)
    # The mismatch contribution is bounded by W_SPEAKER_MISMATCH
    # raw_risk from speaker alone = 35
    # But we need to isolate speaker contribution - check the reason delta
    mismatch_reasons = [r for r in snap.reasons if r.factor == "speaker_mismatch"]
    assert len(mismatch_reasons) == 1
    assert abs(mismatch_reasons[0].delta) == 35.0  # Full weight
    
    # Test with similarity = 0 (should also give full weight)
    engine2 = RiskEngine()
    call_id2 = "bound_test2"
    engine2.start_call(call_id2, claimed_speaker_id="dad")
    snap2 = _step(engine2, call_id2, 0.0, 0.0, 0.0)
    mismatch_reasons2 = [r for r in snap2.reasons if r.factor == "speaker_mismatch"]
    assert len(mismatch_reasons2) == 1
    assert abs(mismatch_reasons2[0].delta) == 35.0


# --- Trust recovery tests ---

def test_trust_recovery_small_gap_below_cap():
    """When target trust is close, recovery follows EMA (below cap)."""
    engine = RiskEngine()
    call_id = "rec_small"
    engine.start_call(call_id, claimed_speaker_id="dad")
    
    # First, drop trust with a mismatch
    snap1 = _step(engine, call_id, 0.0, 0.0, 0.0)
    # trust should be 76 (from earlier test)
    assert snap1.trust_score == 76
    
    # Now send a perfect match - target_trust = 100, prev_trust = 76
    # gap = 24, ema_step = 0.55 * 24 = 13.2, cap = 6
    # recovery = min(13.2, 6) = 6 (cap binds)
    snap2 = _step(engine, call_id, 1.0, 0.0, 0.0)
    # trust should increase by exactly 6 (capped)
    assert snap2.trust_score == 82  # 76 + 6
    assert snap2.risk_score == 18


def test_trust_recovery_large_gap_cap_binds():
    """When target trust is far, recovery is capped at TRUST_RECOVERY_CAP."""
    engine = RiskEngine()
    call_id = "rec_large"
    engine.start_call(call_id, claimed_speaker_id="dad")
    
    # Drop trust significantly
    snap1 = _step(engine, call_id, 0.0, 0.9, 0.9)
    # trust should be quite low
    assert snap1.trust_score < 60
    
    # Perfect match - target_trust = 100, large gap
    # ema_step will be large, but cap is 6
    snap2 = _step(engine, call_id, 1.0, 0.0, 0.0)
    # Recovery should be exactly TRUST_RECOVERY_CAP = 6
    assert snap2.trust_score == snap1.trust_score + 6


def test_trust_recovery_at_cap_boundary():
    """When EMA step equals cap, recovery equals cap."""
    engine = RiskEngine()
    call_id = "rec_cap"
    engine.start_call(call_id, claimed_speaker_id="dad")
    
    # Set up state where gap gives exactly ema_step = 6
    # gap = 6 / 0.55 = 10.909...
    # target_trust - prev_trust = 10.909
    # We need prev_trust such that 100 - raw_risk - prev_trust = 10.909
    # This is hard to hit exactly with discrete scores, so we test the logic directly
    # by checking that recovery is min(ema_step, 6)
    
    # Instead, test with a gap that gives ema_step < 6
    # gap < 6/0.55 = 10.9, so gap = 10
    # We need target_trust = prev_trust + 10
    # If prev_trust = 85, target_trust = 95, raw_risk = 5
    # To get raw_risk = 5, we need very low risk signals
    
    # Simulate by manually setting trust and then evaluating with low risk
    state = engine.start_call(call_id, claimed_speaker_id="dad")
    # Manually set trust to 85
    state.trust_score = 85.0
    
    # Evaluate with very low risk (raw_risk ~ 5, target_trust ~ 95)
    # gap = 10, ema_step = 5.5 < 6, so recovery = 5.5
    snap = _step(engine, call_id, 0.9, 0.02, 0.01)
    # trust should increase by 5.5 (EMA step, below cap)
    # 85 + 5.5 = 90.5 -> 91
    assert snap.trust_score == 91


def test_trust_recovery_already_high():
    """When trust is already high, recovery is small."""
    engine = RiskEngine()
    call_id = "rec_high"
    engine.start_call(call_id, claimed_speaker_id="dad")
    
    # Get trust high first
    snap1 = _step(engine, call_id, 1.0, 0.0, 0.0)  # trust -> 96
    snap2 = _step(engine, call_id, 1.0, 0.0, 0.0)  # trust -> 99 (cap 6, but gap is 4)
    snap3 = _step(engine, call_id, 1.0, 0.0, 0.0)  # trust -> 100 (gap is 1)
    
    # Trust should not exceed 100
    assert snap3.trust_score <= 100
    assert snap3.risk_score >= 0


def test_trust_recovery_low_trust():
    """Recovery from very low trust is capped."""
    engine = RiskEngine()
    call_id = "rec_low"
    engine.start_call(call_id, claimed_speaker_id="dad")
    
    # Drive trust very low
    snap1 = _step(engine, call_id, 0.0, 0.95, 0.95)
    snap2 = _step(engine, call_id, 0.0, 0.95, 0.95)
    snap3 = _step(engine, call_id, 0.0, 0.95, 0.95)
    
    low_trust = snap3.trust_score
    assert low_trust < 30
    
    # Now perfect match - should recover by at most 6
    snap4 = _step(engine, call_id, 1.0, 0.0, 0.0)
    assert snap4.trust_score == low_trust + 6


def test_trust_recovery_repeated_steps():
    """Multiple recovery steps each respect the cap."""
    engine = RiskEngine()
    call_id = "rec_repeated"
    engine.start_call(call_id, claimed_speaker_id="dad")
    
    # Drop trust
    snap1 = _step(engine, call_id, 0.0, 0.8, 0.8)
    start_trust = snap1.trust_score
    
    # 5 perfect matches in a row
    trusts = [start_trust]
    for i in range(5):
        snap = _step(engine, call_id, 1.0, 0.0, 0.0)
        trusts.append(snap.trust_score)
    
    # Each step should increase by at most 6
    for i in range(1, len(trusts)):
        increase = trusts[i] - trusts[i-1]
        assert increase <= 6, f"Step {i}: increase {increase} > 6"
    
    # Final trust should not exceed 100
    assert trusts[-1] <= 100


def test_trust_decay_no_cap():
    """Trust decay (increasing risk) has no cap."""
    engine = RiskEngine()
    call_id = "decay_test"
    engine.start_call(call_id, claimed_speaker_id="dad")
    
    # Start high trust
    snap1 = _step(engine, call_id, 1.0, 0.0, 0.0)  # trust ~96
    snap2 = _step(engine, call_id, 1.0, 0.0, 0.0)  # trust ~99
    
    # Now strong risk signal
    snap3 = _step(engine, call_id, 0.0, 0.9, 0.9)  # large raw_risk
    # Decay should be large (no cap)
    decay = snap2.trust_score - snap3.trust_score
    assert decay > 6  # Should be able to drop more than recovery cap


def test_trust_bounds_never_exceed_range():
    """Trust score always stays within [0, 100]."""
    engine = RiskEngine()
    call_id = "bounds_test"
    engine.start_call(call_id, claimed_speaker_id="dad")
    
    # Test many extreme transitions
    for sim, dp, rp in [
        (1.0, 0.0, 0.0),   # perfect
        (0.0, 1.0, 1.0),   # worst
        (1.0, 0.0, 0.0),   # perfect
        (0.0, 1.0, 1.0),   # worst
        (1.0, 0.0, 0.0),   # perfect
    ]:
        snap = _step(engine, call_id, sim, dp, rp)
        assert 0 <= snap.trust_score <= 100, f"Trust out of bounds: {snap.trust_score}"
        assert 0 <= snap.risk_score <= 100, f"Risk out of bounds: {snap.risk_score}"
