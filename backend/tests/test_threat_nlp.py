"""
Validates the rule-based layer of threat_nlp: multilingual whole-word
matching (no false positives on substrings), and context-aware urgency
scoring (severity, repetition damping, urgency+ask combination bonus).
Uses use_ml=False so these run anywhere without downloading a model.
"""
from app.services.threat_nlp import analyze_text
from app.models.schemas import SensitiveRequestType


def test_english_pin_does_not_match_inside_spinning():
    r = analyze_text("the wheel keeps spinning fast", use_ml=False)
    assert SensitiveRequestType.PIN_PASSWORD not in r.detected_types


def test_english_pin_matches_as_a_word():
    r = analyze_text("please tell me your pin", use_ml=False)
    assert SensitiveRequestType.PIN_PASSWORD in r.detected_types


def test_hindi_pin_does_not_match_inside_pincode():
    r = analyze_text("तुम्हारा पिनकोड बताओ", use_ml=False)
    assert SensitiveRequestType.PIN_PASSWORD not in r.detected_types


def test_hindi_pin_matches_as_a_word():
    r = analyze_text("तुम्हारा पिन बताओ", use_ml=False)
    assert SensitiveRequestType.PIN_PASSWORD in r.detected_types


def test_kannada_otp_matches_as_a_word():
    r = analyze_text("ನಿಮ್ಮ ಒಟಿಪಿ ಹೇಳಿ", use_ml=False)
    assert SensitiveRequestType.OTP in r.detected_types


def test_telugu_password_matches_as_a_word():
    r = analyze_text("మీ పాస్‌వర్డ్ చెప్పండి", use_ml=False)
    assert SensitiveRequestType.PIN_PASSWORD in r.detected_types


def test_tamil_pin_does_not_match_inside_unrelated_word():
    # "பின்னணி" (background) starts with "பின்" (pin) but is a different word
    r = analyze_text("பின்னணி பாருங்கள்", use_ml=False)
    assert SensitiveRequestType.PIN_PASSWORD not in r.detected_types


def test_tamil_pin_matches_as_a_word():
    r = analyze_text("உங்கள் பின் சொல்லுங்கள்", use_ml=False)
    assert SensitiveRequestType.PIN_PASSWORD in r.detected_types


def test_repeating_the_same_urgency_word_does_not_blow_up_score():
    once = analyze_text("this is urgent", use_ml=False).urgency_score
    many = analyze_text("urgent urgent urgent urgent urgent urgent", use_ml=False).urgency_score
    assert many > once          # repetition still counts as pressure...
    assert many <= 0.6          # ...but a single repeated word can't saturate the score


def test_high_severity_threat_framing_scores_higher_than_generic_pressure():
    generic = analyze_text("please do it right now", use_ml=False).urgency_score
    threat = analyze_text("your account will be blocked", use_ml=False).urgency_score
    assert threat > generic


def test_urgency_plus_otp_request_scores_higher_than_urgency_alone():
    urgency_only = analyze_text("please act immediately", use_ml=False).urgency_score
    combo = analyze_text("act immediately and tell me the otp", use_ml=False)
    assert combo.urgency_score > urgency_only
    assert SensitiveRequestType.OTP in combo.detected_types


def test_urgency_plus_financial_ask_is_also_flagged_as_urgent_financial():
    combo = analyze_text("send money right now or else", use_ml=False)
    assert SensitiveRequestType.MONEY_TRANSFER in combo.detected_types
    assert SensitiveRequestType.URGENT_FINANCIAL in combo.detected_types


def test_no_urgency_language_scores_zero():
    r = analyze_text("let's catch up sometime this week", use_ml=False)
    assert r.urgency_score == 0.0


def test_empty_text_returns_none_type():
    r = analyze_text("", use_ml=False)
    assert r.detected_types == [SensitiveRequestType.NONE]


def test_normal_conversation_is_not_flagged():
    r = analyze_text("let's catch up sometime this week", use_ml=False)
    assert r.detected_types == [SensitiveRequestType.NONE]


def test_english_cvv_matches_as_a_word():
    r = analyze_text("please read out your cvv", use_ml=False)
    assert SensitiveRequestType.CVV_CARD in r.detected_types


def test_english_upi_matches_as_a_word():
    r = analyze_text("share your upi pin", use_ml=False)
    assert SensitiveRequestType.UPI_PAYMENT in r.detected_types


def test_english_credentials_phrase_matches():
    r = analyze_text("send me your login details", use_ml=False)
    assert SensitiveRequestType.CREDENTIALS in r.detected_types


def test_tamil_upi_matches_as_a_word():
    r = analyze_text("உங்கள் யுபிஐ ஐடி சொல்லுங்கள்", use_ml=False)
    assert SensitiveRequestType.UPI_PAYMENT in r.detected_types


def test_kannada_credentials_phrase_matches():
    r = analyze_text("ನಿಮ್ಮ ಲಾಗಿನ್ ವಿವರ ಹೇಳಿ", use_ml=False)
    assert SensitiveRequestType.CREDENTIALS in r.detected_types
