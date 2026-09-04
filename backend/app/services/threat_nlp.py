"""
Sensitive-request / social-engineering detection.

Layer 1 (always on, no ML dependency): deterministic keyword/token
rules in English, Hindi, Kannada, Telugu and Tamil covering OTP, PIN/
password, CVV/card, UPI/payment, money transfer, credentials and
urgency phrasing, matched as whole words/phrases (via the `regex`
package, for Unicode word-boundary support across combining scripts)
rather than raw substrings. This guarantees the system catches the
exact attack vocabulary called out in the problem statement even if
no ML model is available.

Layer 2 (optional, used when `transformers` + a local/cached model is
available): a multilingual zero-shot NLI classifier (mDeBERTa-v3-xnli,
falling back to a smaller English-only distilbart model if that isn't
cached) scores the transcript against the same label set to catch
paraphrased/indirect requests the keyword layer misses (e.g. "can you
tell me the six digits that just came to your phone" instead of the
literal word "OTP"), in any of the supported languages. If no model
can be loaded (no network / no cache), Layer 2 is skipped and
`ml_based` is reported as False -- the result is never guessed.
"""
from __future__ import annotations
import regex as re  # drop-in for `re`, but its \b is Unicode-aware across
# combining scripts (Devanagari/Kannada/Telugu/Tamil matras & virama),
# where stdlib `re`'s \b can misfire. Already an indirect dependency via
# `transformers`; pinned directly in requirements.txt since we import it.
from functools import lru_cache

from app.models.schemas import ThreatNlpResult, SensitiveRequestType

# --- Layer 1: deterministic multilingual keyword rules -----------------
# Patterns are intentionally broad; false positives are cheap (they just
# add risk-engine evidence), false negatives on real attack vocabulary
# are expensive, so recall is prioritized over precision here. Every
# pattern is wrapped in \b...\b for real word/token matching (not
# substring) -- e.g. English "pin" must not match "spinning", and Hindi
# "पिन" must not match "पिनकोड" (pincode).
_PATTERNS: dict[SensitiveRequestType, list[str]] = {
    SensitiveRequestType.OTP: [
        r"\botp\b", r"\bone[\s-]?time[\s-]?password\b", r"\bverification code\b",
        r"\bओटीपी\b", r"\bवन[\s-]?टाइम[\s-]?पासवर्ड\b",           # Hindi
        r"\bಒಟಿಪಿ\b", r"\bಒನ್[\s-]?ಟೈಮ್[\s-]?ಪಾಸ್\w*\b",                 # Kannada
        r"\bఓటిపి\b", r"\bవన్[\s-]?టైమ్[\s-]?పాస్\w*\b",                # Telugu
        r"\bஓடிபி\b", r"\bஒரு[\s-]?முறை[\s-]?கடவுச்சொல்\b",             # Tamil
    ],
    SensitiveRequestType.PIN_PASSWORD: [
        r"\bpin\b", r"\bpassword\b", r"\bपिन\b", r"\bपासवर्ड\b",
        r"\bಪಿನ್\b", r"\bಪಾಸ್\w*ವರ್ಡ್\b", r"\bపిన్\b", r"\bపాస్\w*వర్డ్\b",
        r"\bபின்\b", r"\bகடவுச்சொல்\b",
    ],
    SensitiveRequestType.CVV_CARD: [
        r"\bcvv\b", r"\bcard number\b", r"\bdebit card\b", r"\bcredit card\b",
        r"\bसीवीवी\b", r"\bकार्ड नंबर\b", r"\bಸಿವಿವಿ\b", r"\bಕಾರ್ಡ್ ಸಂಖ್ಯೆ\b",
        r"\bసివివి\b", r"\bకార్డ్ నంబర్\b", r"\bசிவிவி\b", r"\bஅட்டை எண்\b",
    ],
    SensitiveRequestType.UPI_PAYMENT: [
        r"\bupi\b", r"\bupi id\b", r"\bupi pin\b", r"\bgoogle pay\b", r"\bphonepe\b", r"\bpaytm\b",
        r"\bयूपीआई\b", r"\bಯುಪಿಐ\b", r"\bయుపిఐ\b", r"\bயுபிஐ\b",
    ],
    SensitiveRequestType.MONEY_TRANSFER: [
        r"\btransfer (the )?money\b", r"\bsend (me )?money\b", r"\bwire transfer\b",
        r"\bपैसे भेजो\b", r"\bपैसे ट्रांसफर\b", r"\bಹಣ ಕಳುಹಿಸಿ\b", r"\bಹಣ ವರ್ಗಾವಣೆ\b",
        r"\bడబ్బు పంపండి\b", r"\bபணம் அனுப்பு\b", r"\bபணப் பரிமாற்றம்\b",
    ],
    SensitiveRequestType.CREDENTIALS: [
        r"\blogin (id|details)\b", r"\busername and password\b", r"\baccount details\b",
        r"\bलॉगिन (आईडी|विवरण)\b", r"\bಲಾಗಿನ್ ವಿವರ\b", r"\bలాగిన్ వివరాలు\b", r"\bஉள்நுழைவு விவரங்கள்\b",
    ],
    SensitiveRequestType.URGENT_FINANCIAL: [
        r"\burgent(ly)?\b", r"\bimmediately\b", r"\bright now\b", r"\bact fast\b", r"\baccount will be blocked\b",
        r"\bतुरंत\b", r"\bअभी\b", r"\bजल्दी\b", r"\bತಕ್ಷಣ\b", r"\bఇప్పుడే\b", r"\bవెంటనే\b", r"\bஉடனடியாக\b", r"\bஇப்போதே\b",
    ],
}


def _compile_all():
    return {k: [re.compile(p, re.IGNORECASE | re.UNICODE) for p in v] for k, v in _PATTERNS.items()}


_COMPILED = _compile_all()

# Explicit threat framing ("your account will be blocked") is a stronger
# manipulation signal than a bare pressure word ("right now"), so it's
# weighted higher in the urgency score below.
_HIGH_SEVERITY_URGENCY = {r"\baccount will be blocked\b"}

# Types that, combined with urgency, make up the classic vishing pattern
# ("do it NOW, read me the OTP") -- co-occurrence gets a score bonus.
_FINANCIAL_OR_CREDENTIAL = {
    SensitiveRequestType.OTP, SensitiveRequestType.PIN_PASSWORD, SensitiveRequestType.CVV_CARD,
    SensitiveRequestType.UPI_PAYMENT, SensitiveRequestType.MONEY_TRANSFER, SensitiveRequestType.CREDENTIALS,
}


def _urgency_score(text: str, detected: list[SensitiveRequestType]) -> float:
    """Context-aware urgency score in [0, 1].

    Plain keyword-hit counting lets a caller repeating one word ("urgent
    urgent urgent") spike the score on its own, and treats "right now" the
    same as an explicit threat. Instead: each distinct phrase's occurrences
    are capped (repetition still counts as pressure, but with diminishing
    returns), high-severity threat framing weighs more, and urgency stacked
    with a financial/OTP/credential ask (the actual scam pattern) gets a
    bonus on top of raw keyword weight.
    """
    signal = 0.0
    for pattern_str, pat in zip(_PATTERNS[SensitiveRequestType.URGENT_FINANCIAL],
                                 _COMPILED[SensitiveRequestType.URGENT_FINANCIAL]):
        occurrences = min(len(pat.findall(text)), 2)  # cap: same word repeated != linear score
        if occurrences:
            weight = 1.5 if pattern_str in _HIGH_SEVERITY_URGENCY else 1.0
            signal += weight * occurrences

    score = signal / 4.0  # ~3 generic hits (or 2 high-severity ones) saturates the score
    if score > 0 and _FINANCIAL_OR_CREDENTIAL & set(detected):
        score += 0.25  # urgency + a concrete ask is worse than either alone
    return min(1.0, score)


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

    urgency_score = _urgency_score(text, detected)

    # If urgency co-occurs with any financial ask, promote to URGENT_FINANCIAL too
    financial = {SensitiveRequestType.UPI_PAYMENT, SensitiveRequestType.MONEY_TRANSFER,
                 SensitiveRequestType.CVV_CARD}
    if urgency_score > 0 and financial & set(detected):
        detected.append(SensitiveRequestType.URGENT_FINANCIAL)

    return detected, matches, urgency_score


# Tried in order, first one that loads locally wins. mDeBERTa-v3-xnli is
# multilingual (trained on XNLI incl. Hindi, plus mDeBERTa's 100-language
# pretraining covers Kannada/Telugu/Tamil), so it's the primary choice for
# this project's Indic languages. distilbart-mnli is English-only but kept
# as a fallback in case only it happens to be cached in a given environment.
_ZS_MODEL_CANDIDATES = [
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "valhalla/distilbart-mnli-12-3",
]


@lru_cache(maxsize=1)
def _get_zero_shot_classifier():
    """Lazily load a zero-shot classifier. Returns None if transformers /
    model weights are unavailable (e.g. no network) -- callers must treat
    that as 'ML layer skipped', never as a signal to fabricate a result.
    """
    try:
        from transformers import pipeline
    except Exception:
        return None
    for model_name in _ZS_MODEL_CANDIDATES:
        try:
            return pipeline("zero-shot-classification", model=model_name)
        except Exception:
            continue
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
