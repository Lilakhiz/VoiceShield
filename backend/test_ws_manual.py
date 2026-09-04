"""Manual websocket test to debug hanging."""
import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
print('TestClient created')

# Try websocket connect
with client.websocket_connect('/ws/call') as ws:
    msg = ws.receive_text()
    print('Received:', msg)
    ws.send_text(json.dumps({"type": "end"}))
    msg = ws.receive_text()
    print('Received end:', msg)
print('WebSocket test passed')