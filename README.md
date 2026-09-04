# VoiceGuard — SIH26104: Real-Time Voice-Cloning Impersonation Detection

A real-time pipeline that scores an in-progress phone call for AI voice-clone
impersonation risk, with a **Trust Continuity Score** that evolves through
the call and explains *why* it changed at each step.

## Architecture

```
Audio Input
    ↓
Speech-to-Text + Language Detection (Whisper)
    ↓
 ┌───────────────────────────────────────────┐
 │ Speaker Verification   (SpeechBrain ECAPA) │
 │ Deepfake Detection     (signal heuristic / pretrained hook) │
 │ Replay Detection       (spectral heuristic, offline)        │
 │ Threat/Sensitive NLP   (multilingual rules + optional ML)   │
 └───────────────────────────────────────────┘
    ↓
Context + Risk Engine  (app/services/risk_engine.py)
    ↓
Risk Score (0–100) + Trust Continuity + explanations
    ↓
FastAPI WebSocket → React console
```

Backend: **FastAPI + WebSockets**, Python/PyTorch for ML, SQLite by default
(swap `DATABASE_URL` to Postgres for production).
Frontend: **React + Vite + Tailwind**.

## What's real vs. what needs your machine's internet access

Every module in `backend/app/services/` calls a real algorithm — nothing
returns `random.random()`. Two categories:

| Module | Status |
|---|---|
| `risk_engine.py` | Pure Python, fully deterministic, **tested** (`tests/test_risk_engine.py`) |
| `replay_detection.py` | Pure signal processing (librosa), no downloads needed, **tested** (`tests/test_signal_detectors.py`) |
| `deepfake_detection.py` | Signal-heuristic fallback works offline (**tested**); pretrained AASIST/RawNet2 hook needs a checkpoint you supply via `DEEPFAKE_MODEL_PATH` |
| `threat_nlp.py` | Rule layer works offline in 5 languages (**tested**); optional zero-shot ML layer downloads a HuggingFace model on first use |
| `stt.py` (Whisper) | Downloads model weights from OpenAI's CDN on first use |
| `speaker_verification.py` (SpeechBrain ECAPA-TDNN) | Downloads weights from HuggingFace Hub on first use |

The last three need normal internet access the first time you run them (to
fetch pretrained weights) — they were built and wired for real use, but
this development sandbox's network is locked to a small allow-list that
doesn't include those model registries, so they couldn't be downloaded
*here*. On your own machine, `pip install -r requirements.txt` and running
the server once will fetch everything automatically. If a model fails to
load (no internet, disk space, etc.), the corresponding result comes back
with `available: false` and an `error` message — the risk engine excludes
that signal rather than substituting a fake number.

## Running it

### Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
First run will download Whisper + SpeechBrain checkpoints (a few hundred MB).

Run the offline-testable parts:
```bash
pytest tests/ -v
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173. Vite proxies `/api` and `/ws` to `localhost:8000`.

### Try it without a microphone
Open the app → Live Call → pick a scenario under "Or run a demo scenario".
Demo Mode drives the **real** risk engine with a scripted timeline of
upstream signal values (see `backend/app/demo/scenarios.py` for exactly why
this is an honest way to demo the pipeline without labeled attack audio).

### Enroll a trusted speaker
Trusted Speakers → record a 5–10s voice sample → Save voiceprint. This
calls the real ECAPA-TDNN embedding pipeline (requires SpeechBrain's
weights to be downloaded, see table above).

## Recalibration before production use

`replay_detection.py` and `deepfake_detection.py`'s signal-heuristic
fallbacks use baseline constants documented inline (e.g. "typical live
speech hf_ratio ~0.12") that were validated against synthetic test
signals in `tests/test_signal_detectors.py`, not a labeled real-world
corpus. Before relying on them for a hackathon demo or beyond, recalibrate
against a dataset like ASVspoof2019 (LA + PA tracks) and update the
thresholds — the code is structured so that's a small, localized change.

## Project layout

```
backend/
  app/
    main.py              # FastAPI app, REST + WebSocket endpoints
    models/schemas.py     # Typed contracts between modules
    services/
      stt.py                    # Whisper STT + language ID
      speaker_verification.py   # SpeechBrain ECAPA-TDNN enrollment/verification
      deepfake_detection.py     # AI-voice probability
      replay_detection.py       # Replay/recording probability
      threat_nlp.py             # Sensitive-request detection (5 languages)
      risk_engine.py            # Trust Continuity + explainable risk scoring
    demo/scenarios.py     # Demo Mode scripted scenarios
    db/database.py        # SQLAlchemy models (calls, snapshots, speakers)
  tests/                 # Offline-runnable pytest suite
frontend/
  src/
    pages/               # LiveCall, TrustedSpeakers, CallHistory, CallDetail, Settings
    components/          # Meter, RiskBadge, TrustContinuity
    lib/                 # API client, WebSocket hook
```
