"""
Sensitive-request / social-engineering detection.

Layer 1 (always on, zero dependencies): deterministic keyword/regex
rules in English, Hindi, Kannada, Telugu and Tamil covering OTP, PIN/
password, CVV/card, UPI/payment, money transfer, credentials and
urgency phrasing. This guarantees the system catches the exact
attack vocabulary called out in the problem statement even if no ML
model is available.

Layer 2 (optional, used when `transformers` + a local/cached model is
available): a zero-shot classifier (`facebook/bart-large-mnli` or a
smaller distilled equivalent) scores the transcript against the same
label set to catch paraphrased/indirect requests the keyword layer
misses (e.g. "can you tell me the six digits that just came to your
phone" instead of the literal word "OTP"). If the model can't be
loaded (no network / no cache), Layer 2 is skipped and `ml_based`
is reported as False -- the result is never guessed.
"""
from __future__ import annotations
import re
from functools import lru_cache

from app.models.schemas import ThreatNlpResult, SensitiveRequestType

# --- Layer 1: deterministic multilingual keyword rules -----------------
# Patterns are intentionally broad; false positives are cheap (they just
# add risk-engine evidence), false negatives on real attack vocabulary
# are expensive, so recall is prioritized over precision here.
# NOTE: Python's `\b` word-boundary is unreliable on Devanagari/Kannada/
# Telugu/Tamil combining scripts (it can fail to match even a plain
# substring occurrence), so `\b` is only used for Latin-script/English
# patterns; Indic-script patterns match as plain substrings instead.
_PATTERNS: dict[SensitiveRequestType, list[str]] = {
    SensitiveRequestType.OTP: [
        r"\botp\b", r"one[\s-]?time[\s-]?password", r"verification code",
        r"ओटीपी", r"वन[\s-]?टाइम[\s-]?पासवर्ड",           # Hindi
        r"ಒಟಿಪಿ", r"ಒನ್[\s-]?ಟೈಮ್[\s-]?ಪಾಸ್\w*",                 # Kannada
        r"ఓటిపి", r"వన్[\s-]?టైమ్[\s-]?పాస్\w*",                # Telugu
        r"ஓடிபி", r"ஒரு[\s-]?முறை[\s-]?கடவுச்சொல்",             # Tamil
    ],
    SensitiveRequestType.PIN_PASSWORD: [
        r"\bpin\b", r"\bpassword\b", r"पिन", r"पासवर्ड",
        r"ಪಿನ್", r"ಪಾಸ್\w*ವರ್ಡ್", r"పిన్", r"పాస్\w*వర్డ్",
        r"பின்", r"கடவுச்சொல்",
    ],
    SensitiveRequestType.CVV_CARD: [
        r"\bcvv\b", r"card number", r"debit card", r"credit card",
        r"सीवीवी", r"कार्ड नंबर", r"ಸಿವಿವಿ", r"ಕಾರ್ಡ್ ಸಂಖ್ಯೆ",
        r"సివివి", r"కార్డ్ నంబర్", r"சிவிவி", r"அட்டை எண்",
    ],
    SensitiveRequestType.UPI_PAYMENT: [
        r"\bupi\b", r"upi id", r"upi pin", r"google pay", r"phonepe", r"paytm",
        r"यूपीआई", r"ಯುಪಿಐ", r"యుపిఐ", r"யுபிஐ",
    ],
    SensitiveRequestType.MONEY_TRANSFER: [
        r"transfer (the )?money", r"send (me )?money", r"wire transfer",
        r"पैसे भेजो", r"पैसे ट्रांसफर", r"ಹಣ ಕಳುಹಿಸಿ", r"ಹಣ ವರ್ಗಾವಣೆ",
        r"డబ్బు పంపండి", r"பணம் அனுப்பு", r"பணப் பரிமாற்றம்",
    ],
    SensitiveRequestType.CREDENTIALS: [
        r"login (id|details)", r"username and password", r"account details",
        r"लॉगिन (आईडी|विवरण)", r"ಲಾಗಿನ್ ವಿವರ", r"లాగిన్ వివరాలు", r"உள்நுழைவு விவரங்கள்",
    ],
    SensitiveRequestType.URGENT_FINANCIAL: [
        r"urgent(ly)?", r"immediately", r"right now", r"act fast", r"account will be blocked",
        r"तुरंत", r"अभी", r"जल्दी", r"ತಕ್ಷಣ", r"ఇప్పుడే", r"వెంటనే", r"உடனடியாக", r"இப்போதே",
    ],
}

_URGENCY_WORDS = _PATTERNS[SensitiveRequestType.URGENT_FINANCIAL]


def _compile_all():
    return {k: [re.compile(p, re.IGNORECASE | re.UNICODE) for p in v] for k, v in _PATTERNS.items()}


_COMPILED = _compile_all()


def _rule_based_pass(text: str) -> tuple[list[SensitiveRequestType], list[str], float]:
    detected: list[SensitiveRequestType] = []
    matches: list[str] = []
    for label, patterns in _COMPILED.items():
        if label == SensitiveRequestType.URGENT_FINANCIAL:
            continue  # scored separately as urgency, not a "sensitive type" hit on its own
        for pat in patterns:
            m = pat.search(text)
            if m:
                detected.append(label)
                matches.append(m.group(0))
                break

    urgency_hits = 0
    for pat in _COMPILED[SensitiveRequestType.URGENT_FINANCIAL]:
        if pat.search(text):
            urgency_hits += 1
    urgency_score = min(1.0, urgency_hits / 3.0)

    # If urgency co-occurs with any financial ask, promote to URGENT_FINANCIAL too
    financial = {SensitiveRequestType.UPI_PAYMENT, SensitiveRequestType.MONEY_TRANSFER,
                 SensitiveRequestType.CVV_CARD}
    if urgency_hits > 0 and financial & set(detected):
        detected.append(SensitiveRequestType.URGENT_FINANCIAL)

    return detected, matches, urgency_score


@lru_cache(maxsize=1)
def _get_zero_shot_classifier():
    """Lazily load a zero-shot classifier. Returns None if transformers /
    model weights are unavailable (e.g. no network) -- callers must treat
    that as 'ML layer skipped', never as a signal to fabricate a result.
    """
    try:
        from transformers import pipeline
        return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
    except Exception:
        return None


_ZS_LABELS = [
    "asking for a one-time password or verification code",
    "asking for a PIN or password",
    "asking for card or CVV details",
    "asking for a UPI or payment transfer",
    "asking to transfer money",
    "asking for login credentials",
    "pressuring the listener to act urgently",
    "normal conversation with no financial request",
]

_ZS_LABEL_MAP = {
    _ZS_LABELS[0]: SensitiveRequestType.OTP,
    _ZS_LABELS[1]: SensitiveRequestType.PIN_PASSWORD,
    _ZS_LABELS[2]: SensitiveRequestType.CVV_CARD,
    _ZS_LABELS[3]: SensitiveRequestType.UPI_PAYMENT,
    _ZS_LABELS[4]: SensitiveRequestType.MONEY_TRANSFER,
    _ZS_LABELS[5]: SensitiveRequestType.CREDENTIALS,
}


def analyze_text(text: str, use_ml: bool = True) -> ThreatNlpResult:
    if not text or not text.strip():
        return ThreatNlpResult(detected_types=[SensitiveRequestType.NONE], available=True)

    detected, matches, urgency_score = _rule_based_pass(text)
    ml_used = False

    if use_ml:
        clf = _get_zero_shot_classifier()
        if clf is not None:
            try:
                result = clf(text, candidate_labels=_ZS_LABELS, multi_label=True)
                for label, score in zip(result["labels"], result["scores"]):
                    if score >= 0.55 and label in _ZS_LABEL_MAP:
                        t = _ZS_LABEL_MAP[label]
                        if t not in detected:
                            detected.append(t)
                ml_used = True
            except Exception:
                ml_used = False

    if not detected:
        detected = [SensitiveRequestType.NONE]

    return ThreatNlpResult(
        detected_types=detected,
        matched_phrases=matches,
        urgency_score=round(urgency_score, 3),
        rule_based=True,
        ml_based=ml_used,
    )
