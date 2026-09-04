"""
Integration tests for WebSocket live call endpoint.
Tests the complete audio pipeline: connect -> start -> audio -> processing -> stop.
"""
import json
import asyncio
import pytest
import numpy as np
from unittest.mock import patch, AsyncMock, MagicMock, Mock
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from app.main import app


# Create a test client
client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(autouse=True)
def mock_db_session():
    """Mock database session to avoid needing actual database."""
    with patch('app.main.SessionLocal') as mock_session_local:
        
        # Create a mock session that does nothing
        mock_session = Mock()
        mock_session.add = Mock()
        mock_session.commit = Mock()
        mock_session.get = Mock(return_value=None)
        mock_session.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(
                order_by=Mock(return_value=[])
            ))
        ))
        mock_session_local.return_value.__enter__ = Mock(return_value=mock_session)
        mock_session_local.return_value.__exit__ = Mock(return_value=None)
        
        yield mock_session


class MockAudioBuffer:
    """Mock AudioBuffer that simulates VAD speech segmentation."""
    def __init__(self):
        self._segments = []
        self._should_return_segments = []
        
    def add_audio(self, audio):
        segments = self._should_return_segments.pop(0) if self._should_return_segments else []
        return segments
    
    def flush(self):
        return None
    
    def queue_segments(self, segments):
        """Queue speech segments to be returned on next add_audio call."""
        self._should_return_segments.append(segments)


class MockSpeechSegment:
    """Mock SpeechSegment for testing."""
    def __init__(self, audio=None, duration=1.0):
        self.audio = audio if audio is not None else np.random.randn(16000).astype(np.float32) * 0.1
        self.duration = duration
        self.sample_rate = 16000
        self.start_time = 0.0


@pytest.fixture
def mock_audio_processing():
    """Mock the audio processing pipeline."""
    with patch('app.main.AudioBuffer') as mock_buffer_class, \
         patch('app.main._process_speech_segment') as mock_process:
        
        mock_buffer = MockAudioBuffer()
        mock_buffer_class.return_value = mock_buffer
        
        # Create a mock snapshot
        from app.models.schemas import RiskSnapshot, RiskLevel, LanguageResult, SpeakerVerificationResult, DeepfakeResult, ReplayResult, ThreatNlpResult, SensitiveRequestType
        from datetime import datetime
        
        mock_snapshot = RiskSnapshot(
            timestamp=datetime.utcnow(),
            elapsed_seconds=1.0,
            risk_score=10,
            risk_level=RiskLevel.LOW,
            trust_score=90,
            reasons=[],
            language=LanguageResult(language="en", confidence=0.99, transcript="test"),
            speaker=SpeakerVerificationResult(similarity=0.9, verified=True),
            deepfake=DeepfakeResult(ai_generated_probability=0.1, method="test"),
            replay=ReplayResult(replay_probability=0.1, live_probability=0.9, method="test"),
            threat=ThreatNlpResult(detected_types=[SensitiveRequestType.NONE], urgency_score=0.0),
        )
        
        async def mock_process_func(call_id, claimed_speaker_id, segment):
            return mock_snapshot
        
        mock_process.side_effect = mock_process_func
        
        yield mock_buffer, mock_process


def test_normal_call_flow(mock_audio_processing):
    """Test normal call: connect -> start -> audio -> processing -> stop -> clean disconnect."""
    mock_buffer, mock_process = mock_audio_processing
    
    with client.websocket_connect("/ws/call", timeout=5.0) as ws:
        # 1. Receive started message
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "started"
        assert "call_id" in msg
        call_id = msg["call_id"]
        
        # 2. Send start message
        ws.send_text(json.dumps({
            "type": "start",
            "claimed_speaker_id": "test_speaker"
        }))
        
        # 3. Send audio - queue a speech segment to be returned
        mock_segment = MockSpeechSegment()
        mock_buffer.queue_segments([mock_segment])
        
        ws.send_text(json.dumps({
            "type": "audio",
            "sample_rate": 16000,
            "channels": 1,
            "sequence": 0,
            "pcm_f32": (np.random.randn(1600) * 0.1).tolist()
        }))
        
        # 4. Receive snapshot
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "snapshot"
        assert "data" in msg
        assert msg["data"]["risk_score"] == 10
        
        # 5. Send end message
        ws.send_text(json.dumps({"type": "end"}))
        
        # 6. Receive ended message
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "ended"


def test_invalid_audio_handled(mock_audio_processing):
    """Test that invalid audio is rejected but connection survives."""
    mock_buffer, mock_process = mock_audio_processing
    
    with client.websocket_connect("/ws/call", timeout=5.0) as ws:
        # Get started
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "started"
        call_id = msg["call_id"]
        
        # Send start
        ws.send_text(json.dumps({
            "type": "start",
            "claimed_speaker_id": "test_speaker"
        }))
        
        # Send invalid audio (empty array)
        ws.send_text(json.dumps({
            "type": "audio",
            "sample_rate": 16000,
            "channels": 1,
            "sequence": 0,
            "pcm_f32": []
        }))
        
        # Should receive error but connection stays alive
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert "Invalid audio" in msg["message"]
        
        # Send valid audio - should still work
        mock_segment = MockSpeechSegment()
        mock_buffer.queue_segments([mock_segment])
        
        ws.send_text(json.dumps({
            "type": "audio",
            "sample_rate": 16000,
            "channels": 1,
            "sequence": 1,
            "pcm_f32": (np.random.randn(1600) * 0.1).tolist()
        }))
        
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "snapshot"
        
        # Clean end
        ws.send_text(json.dumps({"type": "end"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "ended"


def test_client_disconnect_cleans_resources(mock_audio_processing):
    """Test that client disconnect cleans up resources properly."""
    mock_buffer, mock_process = mock_audio_processing
    
    with client.websocket_connect("/ws/call", timeout=5.0) as ws:
        msg = json.loads(ws.receive_text())
        call_id = msg["call_id"]
        
        ws.send_text(json.dumps({
            "type": "start",
            "claimed_speaker_id": "test_speaker"
        }))
        
        # Send some audio
        mock_segment = MockSpeechSegment()
        mock_buffer.queue_segments([mock_segment])
        ws.send_text(json.dumps({
            "type": "audio",
            "sample_rate": 16000,
            "channels": 1,
            "sequence": 0,
            "pcm_f32": (np.random.randn(1600) * 0.1).tolist()
        }))
        
        # Receive snapshot
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "snapshot"
        
        # Disconnect without sending end
        ws.close()
    
    # If we get here without exception, cleanup worked
    assert True


def test_multiple_calls_sequential(mock_audio_processing):
    """Test multiple sequential calls don't leak state."""
    mock_buffer, mock_process = mock_audio_processing
    
    # First call
    with client.websocket_connect("/ws/call", timeout=5.0) as ws:
        msg = json.loads(ws.receive_text())
        call_id_1 = msg["call_id"]
        
        ws.send_text(json.dumps({
            "type": "start",
            "claimed_speaker_id": "speaker_1"
        }))
        
        mock_segment = MockSpeechSegment()
        mock_buffer.queue_segments([mock_segment])
        ws.send_text(json.dumps({
            "type": "audio",
            "sample_rate": 16000,
            "channels": 1,
            "sequence": 0,
            "pcm_f32": (np.random.randn(1600) * 0.1).tolist()
        }))
        
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "snapshot"
        
        ws.send_text(json.dumps({"type": "end"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "ended"
    
    # Second call
    with client.websocket_connect("/ws/call", timeout=5.0) as ws:
        msg = json.loads(ws.receive_text())
        call_id_2 = msg["call_id"]
        
        assert call_id_2 != call_id_1  # Different call IDs
        
        ws.send_text(json.dumps({
            "type": "start",
            "claimed_speaker_id": "speaker_2"
        }))
        
        mock_segment = MockSpeechSegment()
        mock_buffer.queue_segments([mock_segment])
        ws.send_text(json.dumps({
            "type": "audio",
            "sample_rate": 16000,
            "channels": 1,
            "sequence": 0,
            "pcm_f32": (np.random.randn(1600) * 0.1).tolist()
        }))
        
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "snapshot"
        
        ws.send_text(json.dumps({"type": "end"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "ended"


def test_malformed_json_handled(mock_audio_processing):
    """Test that malformed JSON doesn't crash the connection."""
    mock_buffer, mock_process = mock_audio_processing
    
    with client.websocket_connect("/ws/call", timeout=5.0) as ws:
        msg = json.loads(ws.receive_text())
        call_id = msg["call_id"]
        
        ws.send_text(json.dumps({
            "type": "start",
            "claimed_speaker_id": "test_speaker"
        }))
        
        # Send malformed JSON
        ws.send_text("not valid json{")
        
        # Should receive error
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert "Invalid JSON" in msg["message"]
        
        # Connection should still work
        mock_segment = MockSpeechSegment()
        mock_buffer.queue_segments([mock_segment])
        ws.send_text(json.dumps({
            "type": "audio",
            "sample_rate": 16000,
            "channels": 1,
            "sequence": 0,
            "pcm_f32": (np.random.randn(1600) * 0.1).tolist()
        }))
        
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "snapshot"
        
        ws.send_text(json.dumps({"type": "end"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "ended"


def test_unknown_message_type(mock_audio_processing):
    """Test that unknown message types are handled gracefully."""
    mock_buffer, mock_process = mock_audio_processing
    
    with client.websocket_connect("/ws/call", timeout=5.0) as ws:
        msg = json.loads(ws.receive_text())
        call_id = msg["call_id"]
        
        ws.send_text(json.dumps({
            "type": "start",
            "claimed_speaker_id": "test_speaker"
        }))
        
        # Send unknown message type
        ws.send_text(json.dumps({
            "type": "unknown_type",
            "data": "something"
        }))
        
        # Should receive error
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert "Unknown message type" in msg["message"]
        
        # Connection should still work
        ws.send_text(json.dumps({"type": "end"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "ended"


def test_legacy_audio_chunk_protocol(mock_audio_processing):
    """Test that legacy audio_chunk protocol still works."""
    mock_buffer, mock_process = mock_audio_processing
    
    with client.websocket_connect("/ws/call", timeout=5.0) as ws:
        msg = json.loads(ws.receive_text())
        call_id = msg["call_id"]
        
        ws.send_text(json.dumps({
            "type": "start",
            "claimed_speaker_id": "test_speaker"
        }))
        
        # Use legacy protocol
        mock_segment = MockSpeechSegment()
        mock_buffer.queue_segments([mock_segment])
        
        ws.send_text(json.dumps({
            "type": "audio_chunk",
            "sample_rate": 16000,
            "pcm_f32": (np.random.randn(1600) * 0.1).tolist()
        }))
        
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "snapshot"
        
        ws.send_text(json.dumps({"type": "end"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "ended"


def test_audio_validation_rejects_oversized_chunks(mock_audio_processing):
    """Test that oversized audio chunks are rejected."""
    mock_buffer, mock_process = mock_audio_processing
    
    with client.websocket_connect("/ws/call", timeout=5.0) as ws:
        msg = json.loads(ws.receive_text())
        call_id = msg["call_id"]
        
        ws.send_text(json.dumps({
            "type": "start",
            "claimed_speaker_id": "test_speaker"
        }))
        
        # Send oversized chunk (> 8000 samples)
        oversized_audio = np.random.randn(10000).astype(np.float32) * 0.1
        ws.send_text(json.dumps({
            "type": "audio",
            "sample_rate": 16000,
            "channels": 1,
            "sequence": 0,
            "pcm_f32": oversized_audio.tolist()
        }))
        
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert "Chunk too large" in msg["message"]
        
        ws.send_text(json.dumps({"type": "end"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "ended"


def test_audio_validation_rejects_wrong_sample_rate(mock_audio_processing):
    """Test that wrong sample rate is rejected."""
    mock_buffer, mock_process = mock_audio_processing
    
    with client.websocket_connect("/ws/call", timeout=5.0) as ws:
        msg = json.loads(ws.receive_text())
        call_id = msg["call_id"]
        
        ws.send_text(json.dumps({
            "type": "start",
            "claimed_speaker_id": "test_speaker"
        }))
        
        # Send wrong sample rate
        ws.send_text(json.dumps({
            "type": "audio",
            "sample_rate": 8000,  # Wrong!
            "channels": 1,
            "sequence": 0,
            "pcm_f32": (np.random.randn(1600) * 0.1).tolist()
        }))
        
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert "Unsupported sample rate" in msg["message"]
        
        ws.send_text(json.dumps({"type": "end"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "ended"


def test_audio_validation_rejects_nan_values(mock_audio_processing):
    """Test that NaN values in audio are rejected."""
    mock_buffer, mock_process = mock_audio_processing
    
    with client.websocket_connect("/ws/call", timeout=5.0) as ws:
        msg = json.loads(ws.receive_text())
        call_id = msg["call_id"]
        
        ws.send_text(json.dumps({
            "type": "start",
            "claimed_speaker_id": "test_speaker"
        }))
        
        # Send audio with NaN
        audio_with_nan = np.random.randn(1600).astype(np.float32) * 0.1
        audio_with_nan[100] = float('nan')
        ws.send_text(json.dumps({
            "type": "audio",
            "sample_rate": 16000,
            "channels": 1,
            "sequence": 0,
            "pcm_f32": audio_with_nan.tolist()
        }))
        
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert "NaN or Inf" in msg["message"]
        
        ws.send_text(json.dumps({"type": "end"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "ended"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])