"""
Shared data contracts. Every ML/NLP module returns one of these typed
objects instead of a raw float, so a missing/failed model is visible
as `available=False` rather than silently turning into a fabricated number.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class LanguageResult(BaseModel):
    language: str                 # ISO code: en, hi, kn, te, ta, ...
    confidence: float
    transcript: str
    available: bool = True
    error: Optional[str] = None


class SpeakerVerificationResult(BaseModel):
    claimed_speaker_id: Optional[str] = None
    matched_speaker_id: Optional[str] = None
    similarity: float = 0.0       # cosine similarity [-1, 1]
    verified: bool = False
    available: bool = True
    error: Optional[str] = None


class DeepfakeResult(BaseModel):
    ai_generated_probability: float  # 0..1
    method: str                      # which detector produced this number
    available: bool = True
    error: Optional[str] = None


class ReplayResult(BaseModel):
    replay_probability: float        # 0..1
    live_probability: float          # 0..1 (1 - replay, but computed independently for sanity check)
    method: str
    available: bool = True
    error: Optional[str] = None


class SensitiveRequestType(str, Enum):
    OTP = "OTP"
    PIN_PASSWORD = "PIN_PASSWORD"
    CVV_CARD = "CVV_CARD"
    UPI_PAYMENT = "UPI_PAYMENT"
    MONEY_TRANSFER = "MONEY_TRANSFER"
    CREDENTIALS = "CREDENTIALS"
    URGENT_FINANCIAL = "URGENT_FINANCIAL"
    NONE = "NONE"


class ThreatNlpResult(BaseModel):
    detected_types: list[SensitiveRequestType] = Field(default_factory=list)
    matched_phrases: list[str] = Field(default_factory=list)
    urgency_score: float = 0.0      # 0..1
    rule_based: bool = True
    ml_based: bool = False
    available: bool = True
    error: Optional[str] = None


class RiskReason(BaseModel):
    factor: str
    detail: str
    delta: float                    # signed contribution to trust score change


class RiskSnapshot(BaseModel):
    timestamp: datetime
    elapsed_seconds: float
    risk_score: int                 # 0-100 (higher = more dangerous)
    risk_level: RiskLevel
    trust_score: int                # 0-100 (higher = more trustworthy) -- mirrors risk but tracked as continuity
    reasons: list[RiskReason]
    language: Optional[LanguageResult] = None
    speaker: Optional[SpeakerVerificationResult] = None
    deepfake: Optional[DeepfakeResult] = None
    replay: Optional[ReplayResult] = None
    threat: Optional[ThreatNlpResult] = None
    suggest_safe_verification: bool = False
