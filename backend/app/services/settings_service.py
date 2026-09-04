"""
Settings service for managing application and user settings.

Provides a centralized configuration system with:
- Global defaults
- User-specific overrides
- Type-safe value retrieval with validation
- Runtime configuration updates (where supported)
"""
from __future__ import annotations
import os
import json
import logging
from typing import Optional, Any, Dict, Union
from dataclasses import dataclass

from app.db.database import SessionLocal, Setting

logger = logging.getLogger(__name__)

# Default configuration values
# These serve as fallbacks when no database setting exists
DEFAULTS = {
    # Speaker verification
    "speaker_verification_threshold": {
        "value": "0.5",
        "type": "float",
        "description": "Cosine similarity threshold for speaker verification (0-1)",
        "min": 0.0,
        "max": 1.0,
    },
    
    # Risk engine thresholds
    "risk_high_threshold": {
        "value": "55",
        "type": "int",
        "description": "Risk score threshold for HIGH level (0-100)",
        "min": 0,
        "max": 100,
    },
    "risk_critical_threshold": {
        "value": "80",
        "type": "int",
        "description": "Risk score threshold for CRITICAL level (0-100)",
        "min": 0,
        "max": 100,
    },
    
    # Deepfake detection
    "deepfake_jitter_weight": {
        "value": "150.0",
        "type": "float",
        "description": "Weight for jitter term in deepfake heuristic",
    },
    "deepfake_jitter_baseline": {
        "value": "0.01",
        "type": "float",
        "description": "Expected natural jitter baseline",
        "min": 0.0,
        "max": 1.0,
    },
    "deepfake_hnr_weight": {
        "value": "0.8",
        "type": "float",
        "description": "Weight for HNR term in deepfake heuristic",
    },
    "deepfake_hnr_baseline": {
        "value": "1.5",
        "type": "float",
        "description": "Expected natural log(HNR) baseline",
    },
    "deepfake_flatness_weight": {
        "value": "-2.0",
        "type": "float",
        "description": "Weight for flatness term in deepfake heuristic",
    },
    "deepfake_flatness_baseline": {
        "value": "0.25",
        "type": "float",
        "description": "Expected natural flatness baseline",
        "min": 0.0,
        "max": 1.0,
    },
    
    # Replay detection
    "replay_hf_ratio_weight": {
        "value": "6.0",
        "type": "float",
        "description": "Weight for HF ratio term in replay heuristic",
    },
    "replay_hf_ratio_baseline": {
        "value": "0.12",
        "type": "float",
        "description": "Expected live speech HF ratio baseline",
        "min": 0.0,
        "max": 1.0,
    },
    "replay_rolloff_weight": {
        "value": "3.0",
        "type": "float",
        "description": "Weight for spectral rolloff term in replay heuristic",
    },
    "replay_rolloff_baseline": {
        "value": "0.35",
        "type": "float",
        "description": "Expected live speech rolloff baseline",
        "min": 0.0,
        "max": 1.0,
    },
    "replay_centroid_var_weight": {
        "value": "-75.0",
        "type": "float",
        "description": "Weight for centroid variance term in replay heuristic",
    },
    
    # VAD settings
    "vad_energy_threshold": {
        "value": "0.005",
        "type": "float",
        "description": "Energy threshold for VAD",
        "min": 0.0,
        "max": 1.0,
    },
    "vad_spectral_flatness_threshold": {
        "value": "0.5",
        "type": "float",
        "description": "Spectral flatness threshold for VAD",
        "min": 0.0,
        "max": 1.0,
    },
    "vad_min_speech_seconds": {
        "value": "0.5",
        "type": "float",
        "description": "Minimum speech duration to trigger STT",
        "min": 0.1,
        "max": 5.0,
    },
    "vad_max_silence_frames": {
        "value": "15",
        "type": "int",
        "description": "Max silence frames before ending speech segment",
        "min": 1,
        "max": 100,
    },
    
    # Audio processing
    "audio_target_sample_rate": {
        "value": "16000",
        "type": "int",
        "description": "Target sample rate for audio processing",
    },
    "audio_chunk_size": {
        "value": "1600",
        "type": "int",
        "description": "Audio chunk size in samples (100ms at 16kHz)",
        "min": 100,
        "max": 8000,
    },
    
    # Risk engine
    "risk_ema_alpha": {
        "value": "0.55",
        "type": "float",
        "description": "EMA smoothing factor for risk engine (0-1)",
        "min": 0.0,
        "max": 1.0,
    },
    "risk_trust_recovery_cap": {
        "value": "6",
        "type": "int",
        "description": "Max trust recovery per snapshot",
        "min": 0,
        "max": 100,
    },
    "risk_suggest_verification_threshold": {
        "value": "60",
        "type": "int",
        "description": "Risk score threshold for suggesting safe verification",
        "min": 0,
        "max": 100,
    },
    
    # WebSocket
    "ws_max_message_size": {
        "value": "1048576",
        "type": "int",
        "description": "Max WebSocket message size in bytes",
    },
    "ws_audio_queue_max_size": {
        "value": "50",
        "type": "int",
        "description": "Max queued audio messages",
    },
    "ws_buffered_amount_threshold": {
        "value": "65536",
        "type": "int",
        "description": "WebSocket buffered amount threshold for backpressure",
    },
}


@dataclass
class SettingInfo:
    """Information about a setting including metadata."""
    key: str
    value: Any
    value_type: str
    description: Optional[str]
    min_value: Optional[Union[float, int]]
    max_value: Optional[Union[float, int]]
    is_global: bool
    requires_restart: bool


def _coerce_value(value: str, value_type: str) -> Any:
    """Convert string value to appropriate Python type."""
    if value_type == "int":
        return int(value)
    elif value_type == "float":
        return float(value)
    elif value_type == "bool":
        return value.lower() in ("true", "1", "yes", "on")
    elif value_type == "json":
        return json.loads(value)
    return value


def _validate_value(value: Any, setting_meta: Dict) -> tuple[bool, Optional[str]]:
    """Validate a value against setting metadata."""
    min_val = setting_meta.get("min")
    max_val = setting_meta.get("max")
    
    if min_val is not None and value < min_val:
        return False, f"Value {value} below minimum {min_val}"
    if max_val is not None and value > max_val:
        return False, f"Value {value} above maximum {max_val}"
    return True, None


def get_setting(key: str, user_id: Optional[int] = None) -> Any:
    """
    Get a setting value with fallback to defaults.
    
    Priority: user-specific > global > default
    """
    # Try user-specific first
    if user_id is not None:
        with SessionLocal() as session:
            setting = session.query(Setting).filter(
                Setting.key == key,
                Setting.user_id == user_id
            ).first()
            if setting:
                return _coerce_value(setting.value, setting.value_type)
    
    # Try global
    with SessionLocal() as session:
        setting = session.query(Setting).filter(
            Setting.key == key,
            Setting.is_global == True
        ).first()
        if setting:
            return _coerce_value(setting.value, setting.value_type)
    
    # Fall back to default
    if key in DEFAULTS:
        default_meta = DEFAULTS[key]
        return _coerce_value(default_meta["value"], default_meta["type"])
    
    return None


def get_all_settings(user_id: Optional[int] = None) -> Dict[str, Any]:
    """Get all settings as a flat dictionary."""
    result = {}
    
    # Start with defaults
    for key, meta in DEFAULTS.items():
        result[key] = _coerce_value(meta["value"], meta["type"])
    
    # Override with database settings
    with SessionLocal() as session:
        # Global settings
        globals = session.query(Setting).filter(Setting.is_global == True).all()
        for s in globals:
            if s.key in DEFAULTS:
                result[s.key] = _coerce_value(s.value, s.value_type)
        
        # User-specific settings
        if user_id is not None:
            user_settings = session.query(Setting).filter(
                Setting.user_id == user_id
            ).all()
            for s in user_settings:
                if s.key in DEFAULTS:
                    result[s.key] = _coerce_value(s.value, s.value_type)
    
    return result


def get_setting_info(key: str, user_id: Optional[int] = None) -> Optional[SettingInfo]:
    """Get detailed information about a setting."""
    if key not in DEFAULTS:
        return None
    
    meta = DEFAULTS[key]
    value = get_setting(key, user_id)
    
    # Check if there's a user-specific override
    is_global = True
    if user_id is not None:
        with SessionLocal() as session:
            user_setting = session.query(Setting).filter(
                Setting.key == key,
                Setting.user_id == user_id
            ).first()
            if user_setting:
                is_global = False
    
    return SettingInfo(
        key=key,
        value=value,
        value_type=meta["type"],
        description=meta.get("description"),
        min_value=meta.get("min"),
        max_value=meta.get("max"),
        is_global=is_global,
        requires_restart=key in ("audio_target_sample_rate", "audio_chunk_size"),
    )


def set_setting(
    key: str, 
    value: Any, 
    user_id: Optional[int] = None,
    is_global: bool = True
) -> tuple[bool, Optional[str]]:
    """
    Set a setting value.
    Returns (success, error_message).
    """
    if key not in DEFAULTS:
        return False, f"Unknown setting: {key}"
    
    meta = DEFAULTS[key]
    expected_type = meta["type"]
    
    # Validate value
    valid, error = _validate_value(value, meta)
    if not valid:
        return False, error
    
    # Type check
    try:
        if expected_type == "int":
            value = int(value)
        elif expected_type == "float":
            value = float(value)
        elif expected_type == "bool":
            value = bool(value)
    except (ValueError, TypeError):
        return False, f"Value must be of type {expected_type}"
    
    # Convert to string for storage
    str_value = str(value)
    
    with SessionLocal() as session:
        # Find existing setting
        query = session.query(Setting).filter(Setting.key == key)
        if is_global:
            query = query.filter(Setting.is_global == True)
        else:
            query = query.filter(Setting.user_id == user_id)
        
        existing = query.first()
        
        if existing:
            existing.value = str_value
            existing.updated_at = datetime.utcnow()
        else:
            new_setting = Setting(
                key=key,
                value=str_value,
                value_type=expected_type,
                description=meta.get("description"),
                is_global=is_global,
                user_id=user_id if not is_global else None,
            )
            session.add(new_setting)
        
        session.commit()
    
    return True, None


def initialize_defaults() -> int:
    """Initialize database with default settings if not present."""
    count = 0
    with SessionLocal() as session:
        for key, meta in DEFAULTS.items():
            existing = session.query(Setting).filter(
                Setting.key == key,
                Setting.is_global == True
            ).first()
            if not existing:
                session.add(Setting(
                    key=key,
                    value=meta["value"],
                    value_type=meta["type"],
                    description=meta.get("description"),
                    is_global=True,
                ))
                count += 1
        session.commit()
    return count


# Import here to avoid circular imports
from datetime import datetime