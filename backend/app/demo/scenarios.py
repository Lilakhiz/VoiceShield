"""
Demo Mode.

Important: these scenarios do NOT fabricate risk/trust scores directly.
Each scenario is a scripted timeline of *upstream signal values*
(speaker similarity, deepfake probability, replay probability, detected
sensitive-request types) representative of that attack pattern, which
are fed through the exact same `RiskEngine.evaluate()` used by the live
pipeline. The score you see in Demo Mode is computed by the real
scoring logic, not hand-set -- only the sensor inputs are scripted,
which is the standard, honest way to demo a pipeline without a live
microphone or a labeled attack-audio corpus on hand.
"""
from __future__ import annotations
import asyncio
from datetime import datetime

from app.models.schemas import (
    LanguageResult, SpeakerVerificationResult, DeepfakeResult, ReplayResult,
    ThreatNlpResult, SensitiveRequestType,
)
from app.services.risk_engine import risk_engine

SCENARIOS = {
    "genuine_trusted_caller": {
        "label": "Genuine trusted caller",
        "steps": [
            dict(t=0, transcript="Hi, this is Rohan, calling about the meeting tomorrow.",
                 speaker_sim=0.91, deepfake_p=0.03, replay_p=0.02, sensitive=[]),
            dict(t=8, transcript="Can we move it to 3pm instead?",
                 speaker_sim=0.90, deepfake_p=0.04, replay_p=0.03, sensitive=[]),
        ],
    },
    "ai_cloned_voice": {
        "label": "AI-cloned voice",
        "steps": [
            dict(t=0, transcript="Hi, it's Rohan, I need to talk to you quickly.",
                 speaker_sim=0.55, deepfake_p=0.35, replay_p=0.05, sensitive=[]),
            dict(t=10, transcript="I'm having some phone trouble, can you hear me okay?",
                 speaker_sim=0.48, deepfake_p=0.62, replay_p=0.06, sensitive=[]),
            dict(t=20, transcript="Something feels off with my voice today, ignore that.",
                 speaker_sim=0.40, deepfake_p=0.81, replay_p=0.07, sensitive=[]),
        ],
    },
    "replay_attack": {
        "label": "Replay attack",
        "steps": [
            dict(t=0, transcript="Hello, this is the bank calling to verify your account.",
                 speaker_sim=0.20, deepfake_p=0.10, replay_p=0.35, sensitive=[]),
            dict(t=9, transcript="We need to confirm your details for security purposes.",
                 speaker_sim=0.18, deepfake_p=0.12, replay_p=0.58, sensitive=[]),
            dict(t=18, transcript="Please stay on the line, this is important.",
                 speaker_sim=0.15, deepfake_p=0.14, replay_p=0.79, sensitive=[]),
        ],
    },
    "ai_voice_otp_request": {
        "label": "AI voice + OTP request",
        "steps": [
            dict(t=0, transcript="Hi Dad, it's me, I'm in a bit of trouble.",
                 speaker_sim=0.50, deepfake_p=0.58, replay_p=0.08, sensitive=[]),
            dict(t=12, transcript="I need you to read me the OTP you just received.",
                 speaker_sim=0.45, deepfake_p=0.70, replay_p=0.09,
                 sensitive=[SensitiveRequestType.OTP]),
            dict(t=20, transcript="Please hurry, it's urgent, the code will expire.",
                 speaker_sim=0.40, deepfake_p=0.77, replay_p=0.10,
                 sensitive=[SensitiveRequestType.OTP], urgency=0.8),
        ],
    },
    "ai_voice_urgent_financial": {
        "label": "AI voice + urgent financial request",
        "steps": [
            dict(t=0, transcript="This is your relationship manager from the bank.",
                 speaker_sim=0.30, deepfake_p=0.45, replay_p=0.10, sensitive=[]),
            dict(t=10, transcript="Your account will be blocked unless you act immediately.",
                 speaker_sim=0.28, deepfake_p=0.60, replay_p=0.11,
                 sensitive=[SensitiveRequestType.URGENT_FINANCIAL], urgency=0.7),
            dict(t=18, transcript="Please transfer the pending amount via UPI right now.",
                 speaker_sim=0.25, deepfake_p=0.74, replay_p=0.12,
                 sensitive=[SensitiveRequestType.UPI_PAYMENT, SensitiveRequestType.URGENT_FINANCIAL],
                 urgency=0.9),
        ],
    },
}


async def run_demo_scenario(scenario_key: str, call_id: str, on_snapshot):
    """Drive the real RiskEngine with a scripted signal timeline and invoke
    `on_snapshot(snapshot)` (e.g. to push over a WebSocket) after each step,
    with realistic pacing between steps."""
    scenario = SCENARIOS[scenario_key]
    risk_engine.start_call(call_id, claimed_speaker_id="demo_speaker")

    prev_t = 0
    for step in scenario["steps"]:
        await asyncio.sleep(max(0.0, step["t"] - prev_t) * 0.3)  # sped up for demo purposes
        prev_t = step["t"]

        language = LanguageResult(language="en", confidence=0.95, transcript=step["transcript"])
        speaker = SpeakerVerificationResult(
            claimed_speaker_id="demo_speaker",
            matched_speaker_id="demo_speaker" if step["speaker_sim"] >= 0.5 else None,
            similarity=step["speaker_sim"],
            verified=step["speaker_sim"] >= 0.5,
        )
        deepfake = DeepfakeResult(ai_generated_probability=step["deepfake_p"], method="demo-script")
        replay = ReplayResult(replay_probability=step["replay_p"],
                               live_probability=1 - step["replay_p"], method="demo-script")
        threat = ThreatNlpResult(
            detected_types=step.get("sensitive") or [SensitiveRequestType.NONE],
            matched_phrases=[], urgency_score=step.get("urgency", 0.0),
        )

        snapshot = risk_engine.evaluate(
            call_id, language=language, speaker=speaker,
            deepfake=deepfake, replay=replay, threat=threat,
        )
        await on_snapshot(snapshot)
