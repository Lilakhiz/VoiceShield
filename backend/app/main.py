from __future__ import annotations
import io
import json
import os
import uuid
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Depends, Query, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.db.database import init_db, SessionLocal, TrustedSpeaker, Call, Snapshot, User
from app.services import stt, speaker_verification, deepfake_detection, replay_detection, threat_nlp
from app.services.risk_engine import risk_engine
from app.services.audio_processing import (
    validate_audio_chunk, AudioBuffer, SimpleVAD, SpeechSegment,
    AudioValidationError, TARGET_SAMPLE_RATE
)
from app.demo.scenarios import SCENARIOS, run_demo_scenario
from app.auth import (
    create_access_token, create_refresh_token, get_current_user_id,
    get_current_user_ws, require_admin, create_demo_user_if_not_exists,
    DEMO_MODE, verify_password, decode_token
)
from app.logging_config import setup_logging, get_logger, log_call_event
from app.state import get_rate_limiter, DistributedRateLimiter

# Setup structured logging
setup_logging(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    json_format=os.environ.get("LOG_JSON", "false").lower() == "true",
    log_file=os.environ.get("LOG_FILE")
)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Create demo user if in demo mode
    try:
        create_demo_user_if_not_exists()
        logger.info("Demo user created or already exists")
    except Exception as e:
        logger.warning(f"Failed to create demo user: {e}")
    # Initialize default settings
    try:
        from app.services.settings_service import initialize_defaults
        count = initialize_defaults()
        if count > 0:
            logger.info(f"Initialized {count} default settings")
    except Exception as e:
        logger.warning(f"Failed to initialize default settings: {e}")
    # Load speaker embeddings from database into memory
    try:
        from app.services.speaker_verification import load_all_speakers_from_db
        loaded = load_all_speakers_from_db()
        logger.info(f"Loaded {loaded} speaker embeddings on startup")
    except Exception as e:
        logger.warning(f"Failed to load speaker embeddings on startup: {e}")
    logger.info("Application startup complete")
    yield
    logger.info("Application shutdown")


app = FastAPI(title="SIH26104 - Real-Time Voice-Cloning Impersonation Detection",
              lifespan=lifespan)

# CORS configuration - configurable for production
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Security scheme for OpenAPI
security = HTTPBearer(auto_error=False)

from fastapi import Request

# Rate limiting dependencies
def get_rate_limit_dependency(max_requests: int = 60, window_seconds: int = 60):
    """Create a rate limit dependency for API endpoints."""
    limiter = get_rate_limiter("api")
    
    async def rate_limit(request: Request):
        # Get client identifier (IP or user_id)
        client_ip = request.client.host if request.client else "unknown"
        user_id = getattr(request.state, "user_id", None)
        identifier = f"user:{user_id}" if user_id else f"ip:{client_ip}"
        
        allowed, info = await limiter.is_allowed(identifier, max_requests, window_seconds)
        
        # Add rate limit headers
        request.state.rate_limit_info = info
        
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": str(info["remaining"]),
                    "X-RateLimit-Reset": str(info["reset"]),
                },
            )
    
    return rate_limit


# Default rate limiter for API endpoints (60 requests per minute)
default_rate_limit = get_rate_limit_dependency(60, 60)

# Stricter rate limit for auth endpoints (10 requests per minute)
auth_rate_limit = get_rate_limit_dependency(10, 60)

# Very strict rate limit for enrollment (5 requests per minute)
enroll_rate_limit = get_rate_limit_dependency(5, 60)


# ----------------------------- Authentication -----------------------------

class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


@app.post("/api/auth/login", response_model=TokenResponse, dependencies=[Depends(auth_rate_limit)])
async def login(request: TokenRequest):
    """Authenticate user and return JWT tokens."""
    if DEMO_MODE:
        # In demo mode, accept demo/demo
        if request.username == "demo" and request.password == "demo":
            access_token = create_access_token({"sub": "1", "username": "demo"})
            refresh_token = create_refresh_token({"sub": "1", "username": "demo"})
            return TokenResponse(access_token=access_token, refresh_token=refresh_token)
    
    with SessionLocal() as session:
        user = session.query(User).filter(User.username == request.username).first()
        if not user or not verify_password(request.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled",
            )
        # Update last login
        user.last_login = datetime.utcnow()
        session.commit()
        
        access_token = create_access_token({"sub": str(user.id), "username": user.username})
        refresh_token = create_refresh_token({"sub": str(user.id), "username": user.username})
        return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@app.post("/api/auth/refresh", response_model=TokenResponse, dependencies=[Depends(auth_rate_limit)])
async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Refresh access token using refresh token."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token required",
        )
    
    payload = decode_token(credentials.credentials)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    
    user_id = payload.get("sub")
    username = payload.get("username")
    
    access_token = create_access_token({"sub": user_id, "username": username})
    refresh_token = create_refresh_token({"sub": user_id, "username": username})
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@app.get("/api/auth/me")
async def get_current_user_info(user_id: int = Depends(get_current_user_id)):
    """Get current user information."""
    if DEMO_MODE and user_id == 1:
        return {"id": 1, "username": "demo", "email": "demo@voiceguard.local", "is_admin": True}
    
    with SessionLocal() as session:
        user = session.get(User, user_id)
        if not user:
            raise HTTPException(404, "User not found")
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "is_admin": user.is_admin,
        }


# ----------------------------- Settings -----------------------------

class SettingUpdate(BaseModel):
    key: str
    value: Any


class SettingResponse(BaseModel):
    key: str
    value: Any
    value_type: str
    description: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    is_global: bool
    requires_restart: bool


@app.get("/api/settings")
async def list_settings(user_id: int = Depends(get_current_user_id)):
    """Get all settings with metadata."""
    from app.services.settings_service import get_all_settings, get_setting_info
    settings = get_all_settings(user_id if not DEMO_MODE else None)
    result = {}
    for key, value in settings.items():
        info = get_setting_info(key, user_id if not DEMO_MODE else None)
        if info:
            result[key] = {
                "value": info.value,
                "value_type": info.value_type,
                "description": info.description,
                "min_value": info.min_value,
                "max_value": info.max_value,
                "is_global": info.is_global,
                "requires_restart": info.requires_restart,
            }
    return result


@app.get("/api/settings/{key}")
async def get_setting(key: str, user_id: int = Depends(get_current_user_id)):
    """Get a specific setting with metadata."""
    from app.services.settings_service import get_setting_info
    info = get_setting_info(key, user_id if not DEMO_MODE else None)
    if not info:
        raise HTTPException(404, f"Setting not found: {key}")
    return {
        "key": info.key,
        "value": info.value,
        "value_type": info.value_type,
        "description": info.description,
        "min_value": info.min_value,
        "max_value": info.max_value,
        "is_global": info.is_global,
        "requires_restart": info.requires_restart,
    }


@app.put("/api/settings/{key}")
async def update_setting(
    key: str, 
    update: SettingUpdate, 
    user_id: int = Depends(get_current_user_id)
):
    """Update a setting value (admin only for global settings)."""
    from app.services.settings_service import set_setting, DEFAULTS
    if key not in DEFAULTS:
        raise HTTPException(404, f"Unknown setting: {key}")
    
    # Only admins can change global settings
    if DEMO_MODE:
        is_global = True
    else:
        with SessionLocal() as session:
            user = session.get(User, user_id)
            is_global = user.is_admin if user else False
    
    success, error = set_setting(key, update.value, user_id if not is_global else None, is_global)
    if not success:
        raise HTTPException(400, error)
    
    # Reload affected modules if needed
    if key in ("speaker_verification_threshold",):
        # Speaker verification threshold is read from env at module load
        # Would need restart or module reload to take effect
        pass
    
    return {"status": "updated", "key": key, "value": update.value}


@app.post("/api/settings/initialize")
async def initialize_settings(user_id: int = Depends(require_admin)):
    """Initialize default settings in database (admin only)."""
    from app.services.settings_service import initialize_defaults
    count = initialize_defaults()
    return {"status": "initialized", "settings_created": count}


# ----------------------------- Health Check -----------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring.
    
    Returns:
        - status: "healthy" or "degraded"
        - timestamp: current UTC time
        - version: application version
        - checks: individual dependency health checks
    """
    from datetime import datetime
    import sqlite3
    
    checks = {}
    overall_healthy = True
    
    # Database check
    try:
        with SessionLocal() as session:
            session.execute("SELECT 1")
        checks["database"] = {"status": "healthy", "latency_ms": 0}
    except Exception as e:
        checks["database"] = {"status": "unhealthy", "error": str(e)}
        overall_healthy = False
    
    # Check if ML models are available
    try:
        from app.services.speaker_verification import _get_classifier
        classifier = _get_classifier()
        checks["speaker_verification_model"] = {
            "status": "healthy" if classifier else "unavailable",
            "loaded": classifier is not None
        }
    except Exception as e:
        checks["speaker_verification_model"] = {"status": "error", "error": str(e)}
    
    try:
        from app.services.stt import _get_model
        model = _get_model()
        checks["stt_model"] = {
            "status": "healthy" if model else "unavailable",
            "loaded": model is not None
        }
    except Exception as e:
        checks["stt_model"] = {"status": "error", "error": str(e)}
    
    # Overall status
    status = "healthy" if overall_healthy else "degraded"
    
    response = {
        "status": status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0",
        "checks": checks,
    }
    
    # Return 503 if degraded
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content=response,
        status_code=200 if status == "healthy" else 503
    )


# ----------------------------- Trusted speakers -----------------------------

@app.post("/api/speakers/enroll", dependencies=[Depends(enroll_rate_limit)])
async def enroll_speaker(
    speaker_id: str = Form(...), 
    name: str = Form(...),
    audio: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id)
):
    raw = await audio.read()
    try:
        waveform, sr = sf.read(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not read audio: {e}")

    ok = speaker_verification.enroll_speaker(speaker_id, np.asarray(waveform), sr)
    if not ok:
        raise HTTPException(503, "Speaker verification model unavailable on this server "
                                  "(model weights could not be loaded).")

    # Ensure speaker record exists in database (embedding persisted by enroll_speaker)
    with SessionLocal() as session:
        existing = session.get(TrustedSpeaker, speaker_id)
        if not existing:
            session.add(TrustedSpeaker(id=speaker_id, name=name, enrollment_clips=1, user_id=user_id))
            session.commit()
        elif existing.name != name:
            # Update name if changed
            existing.name = name
            session.commit()

    return {"status": "enrolled", "speaker_id": speaker_id}


@app.get("/api/speakers")
def list_speakers(user_id: int = Depends(get_current_user_id)):
    with SessionLocal() as session:
        if DEMO_MODE:
            rows = session.query(TrustedSpeaker).all()
        else:
            rows = session.query(TrustedSpeaker).filter(TrustedSpeaker.user_id == user_id).all()
        return [
            {"id": r.id, "name": r.name, "enrollment_clips": r.enrollment_clips,
             "created_at": r.created_at.isoformat()}
            for r in rows
        ]


# ----------------------------- Call history -----------------------------

@app.get("/api/calls")
def list_calls(user_id: int = Depends(get_current_user_id)):
    with SessionLocal() as session:
        if DEMO_MODE:
            rows = session.query(Call).order_by(Call.started_at.desc()).all()
        else:
            rows = session.query(Call).filter(Call.user_id == user_id).order_by(Call.started_at.desc()).all()
        return [
            {"id": c.id, "claimed_speaker_id": c.claimed_speaker_id,
             "started_at": c.started_at.isoformat(),
             "ended_at": c.ended_at.isoformat() if c.ended_at else None,
             "final_risk_score": c.final_risk_score,
             "final_risk_level": c.final_risk_level,
             "demo_scenario": c.demo_scenario}
            for c in rows
        ]


@app.get("/api/calls/{call_id}")
def get_call_detail(call_id: str, user_id: int = Depends(get_current_user_id)):
    with SessionLocal() as session:
        call = session.get(Call, call_id)
        if not call:
            raise HTTPException(404, "call not found")
        if not DEMO_MODE and call.user_id != user_id:
            raise HTTPException(403, "Not authorized to access this call")
        snapshots = (session.query(Snapshot)
                     .filter(Snapshot.call_id == call_id)
                     .order_by(Snapshot.elapsed_seconds).all())
        return {
            "call": {"id": call.id, "claimed_speaker_id": call.claimed_speaker_id,
                      "started_at": call.started_at.isoformat(),
                      "demo_scenario": call.demo_scenario},
            "snapshots": [s.payload for s in snapshots],
        }


@app.get("/api/demo/scenarios")
def list_demo_scenarios(user_id: int = Depends(get_current_user_id)):
    return [{"key": k, "label": v["label"]} for k, v in SCENARIOS.items()]


def _persist_snapshot(session, call_id: str, snapshot):
    session.add(Snapshot(
        call_id=call_id,
        timestamp=snapshot.timestamp,
        elapsed_seconds=snapshot.elapsed_seconds,
        risk_score=snapshot.risk_score,
        risk_level=snapshot.risk_level.value,
        trust_score=snapshot.trust_score,
        reasons=[r.model_dump() for r in snapshot.reasons],
        payload=snapshot.model_dump(mode="json"),
    ))
    call = session.get(Call, call_id)
    if call:
        call.final_risk_score = snapshot.risk_score
        call.final_risk_level = snapshot.risk_level.value
    session.commit()


async def _process_speech_segment(
    call_id: str,
    claimed_speaker_id: Optional[str],
    segment: SpeechSegment
) -> None:
    """
    Process a single speech segment through the full pipeline.
    Runs in a thread pool to avoid blocking the WebSocket loop.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    
    # Run the heavy ML inference in executor
    def run_pipeline():
        # STT + Language ID
        language = stt.transcribe_and_detect_language(segment.audio, segment.sample_rate)
        
        # Other detectors (these are fast, run inline)
        speaker = speaker_verification.verify_speaker(segment.audio, segment.sample_rate, claimed_speaker_id)
        deepfake = deepfake_detection.detect_deepfake(segment.audio, segment.sample_rate)
        replay = replay_detection.detect_replay(segment.audio, segment.sample_rate)
        threat = threat_nlp.analyze_text(language.transcript if language.available else "")
        
        # Risk engine evaluation
        snapshot = risk_engine.evaluate(
            call_id, language=language, speaker=speaker,
            deepfake=deepfake, replay=replay, threat=threat,
        )
        return snapshot
    
    snapshot = await loop.run_in_executor(None, run_pipeline)
    
    # Persist snapshot
    with SessionLocal() as session:
        _persist_snapshot(session, call_id, snapshot)
    
    return snapshot


# WebSocket configuration constants
MAX_WS_MESSAGE_SIZE = 1024 * 1024  # 1MB max message size
MAX_AUDIO_QUEUE_SIZE = 100  # Max pending audio segments per call


async def authenticate_ws(ws: WebSocket) -> Optional[int]:
    """Authenticate WebSocket connection. Returns user_id or None if demo mode."""
    # Try query parameter first
    token = ws.query_params.get("token")
    
    # Fallback to Authorization header
    if token is None:
        auth_header = ws.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
    
    if DEMO_MODE:
        return 1
    
    if token is None:
        await ws.close(code=4001, reason="Authentication required")
        return None
    
    payload = decode_token(token)
    if payload is None or payload.get("type") != "access":
        await ws.close(code=4001, reason="Invalid or expired token")
        return None
    
    user_id = payload.get("sub")
    if user_id is None:
        await ws.close(code=4001, reason="Invalid token payload")
        return None
    
    return int(user_id)


@app.websocket("/ws/call")
async def live_call(ws: WebSocket):
    """
    Live call WebSocket endpoint with bounded audio buffering and VAD.
    
    Protocol (JSON frames from client):
      {"type": "start", "claimed_speaker_id": "s1"}
      {"type": "audio", "sample_rate": 16000, "channels": 1, "sequence": 0, "pcm_f32": [...]}
      {"type": "end"}
    Server pushes RiskSnapshot JSON frames after each processed speech segment.
    
    Authentication: Provide JWT token as query parameter ?token=... or Authorization: Bearer header
    """
    user_id = await authenticate_ws(ws)
    if user_id is None:
        return
    
    await ws.accept()
    call_id = str(uuid.uuid4())
    claimed_speaker_id = None
    call_start_time = datetime.utcnow()

    with SessionLocal() as session:
        session.add(Call(id=call_id, started_at=call_start_time, user_id=user_id))
        session.commit()

    # Audio processing pipeline
    audio_buffer = AudioBuffer(sample_rate=TARGET_SAMPLE_RATE)
    
    # Track if we've sent initial snapshot (for UI initialization)
    sent_initial_snapshot = False

    async def send_error(message: str):
        """Safely send error to client without crashing."""
        try:
            await ws.send_json({"type": "error", "message": message})
        except Exception:
            pass

    async def send_snapshot(snapshot):
        """Safely send snapshot to client."""
        try:
            await ws.send_json({"type": "snapshot", "data": snapshot.model_dump(mode="json")})
        except Exception:
            pass

    try:
        while True:
            # Receive with size limit to prevent memory exhaustion
            raw = await ws.receive_text()
            
            if len(raw) > MAX_WS_MESSAGE_SIZE:
                await send_error("Message too large")
                continue
            
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send_error("Invalid JSON")
                continue

            # Validate message structure
            if not isinstance(msg, dict):
                await send_error("Message must be a JSON object")
                continue
                
            msg_type = msg.get("type")

            if msg_type == "start":
                # Prevent duplicate start messages
                if claimed_speaker_id is not None:
                    await send_error("Call already started")
                    continue
                    
                claimed_speaker_id = msg.get("claimed_speaker_id")
                risk_engine.start_call(call_id, claimed_speaker_id=claimed_speaker_id)
                with SessionLocal() as session:
                    call = session.get(Call, call_id)
                    call.claimed_speaker_id = claimed_speaker_id
                    session.commit()
                await ws.send_json({"type": "started", "call_id": call_id})

            elif msg_type == "audio":
                # New bounded chunk protocol
                try:
                    # Validate audio
                    sample_rate = int(msg.get("sample_rate", 0))
                    channels = int(msg.get("channels", 1))
                    pcm_data = msg.get("pcm_f32")
                    
                    waveform = validate_audio_chunk(pcm_data, sample_rate, channels)
                    
                    # Add to buffer and get completed speech segments
                    segments = audio_buffer.add_audio(waveform)
                    
                    # Process each completed speech segment
                    for segment in segments:
                        snapshot = await _process_speech_segment(
                            call_id, claimed_speaker_id, segment
                        )
                        await send_snapshot(snapshot)
                        
                except AudioValidationError as e:
                    # Log validation error but don't crash the connection
                    logger.warning(f"Audio validation error: {e}")
                    await send_error(f"Invalid audio: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing audio: {e}", exc_info=True)
                    await send_error("Audio processing error")
                    
            elif msg_type == "audio_chunk":
                # Legacy protocol support - convert and process same way
                try:
                    sr = int(msg.get("sample_rate", TARGET_SAMPLE_RATE))
                    pcm_data = msg.get("pcm_f32")
                    
                    waveform = validate_audio_chunk(pcm_data, sr, 1)
                    
                    segments = audio_buffer.add_audio(waveform)
                    
                    for segment in segments:
                        snapshot = await _process_speech_segment(
                            call_id, claimed_speaker_id, segment
                        )
                        await send_snapshot(snapshot)
                        
                except AudioValidationError as e:
                    logger.warning(f"Audio validation error: {e}")
                    await send_error(f"Invalid audio: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing audio: {e}", exc_info=True)
                    await send_error("Audio processing error")
  
            elif msg_type == "end":
                # Flush any remaining speech
                final_segment = audio_buffer.flush()
                if final_segment:
                    snapshot = await _process_speech_segment(
                        call_id, claimed_speaker_id, final_segment
                    )
                    await send_snapshot(snapshot)
                
                with SessionLocal() as session:
                    call = session.get(Call, call_id)
                    call.ended_at = datetime.utcnow()
                    session.commit()
                await ws.send_json({"type": "ended"})
                break
  
            else:
                await send_error(f"Unknown message type: {msg_type}")
  
    except WebSocketDisconnect:
        # Flush on disconnect
        try:
            final_segment = audio_buffer.flush()
            if final_segment:
                snapshot = await _process_speech_segment(
                    call_id, claimed_speaker_id, final_segment
                )
                await send_snapshot(snapshot)
        except Exception:
            pass
        # Update call end time
        with SessionLocal() as session:
            call = session.get(Call, call_id)
            if call and call.ended_at is None:
                call.ended_at = datetime.utcnow()
                session.commit()
    except Exception as e:
        # Catch-all for any unexpected errors in the WebSocket handler
        logger.error(f"Unexpected WebSocket error: {e}", exc_info=True)
        try:
            await ws.close(code=1011, reason="Internal server error")
        except Exception:
            pass
        with SessionLocal() as session:
            call = session.get(Call, call_id)
            if call and call.ended_at is None:
                call.ended_at = datetime.utcnow()
                session.commit()


@app.websocket("/ws/demo/{scenario_key}")
async def demo_call(ws: WebSocket, scenario_key: str):
    user_id = await authenticate_ws(ws)
    if user_id is None:
        return
    
    await ws.accept()
    if scenario_key not in SCENARIOS:
        await ws.send_json({"type": "error", "message": "unknown scenario"})
        await ws.close()
        return

    call_id = f"demo-{uuid.uuid4()}"
    with SessionLocal() as session:
        session.add(Call(id=call_id, started_at=datetime.utcnow(), demo_scenario=scenario_key, user_id=user_id))
        session.commit()

    async def on_snapshot(snapshot):
        with SessionLocal() as session:
            _persist_snapshot(session, call_id, snapshot)
        await ws.send_json({"type": "snapshot", "data": snapshot.model_dump(mode="json")})

    await ws.send_json({"type": "started", "call_id": call_id})
    await run_demo_scenario(scenario_key, call_id, on_snapshot)
    with SessionLocal() as session:
        call = session.get(Call, call_id)
        call.ended_at = datetime.utcnow()
        session.commit()
    await ws.send_json({"type": "ended"})
    await ws.close()