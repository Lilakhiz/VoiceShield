"""
Distributed state management for horizontal scaling.

This module provides abstractions for managing state across multiple
application instances. It supports both in-memory (single-instance)
and Redis-backed (multi-instance) operation modes.

State categories:
- PERSISTENT: Stored in database, survives restarts, shared across instances
- CACHED: In-memory copy of persistent data, refreshed from DB
- EPHEMERAL: Process-local, lost on restart, not shared (e.g., WebSocket connections)
- DISTRIBUTED: Shared across instances via Redis (e.g., rate limits, locks)
"""
from __future__ import annotations
import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
USE_REDIS = os.environ.get("USE_REDIS", "false").lower() == "true"

_redis_client = None
_redis_available = False


class StateBackend(ABC):
    """Abstract interface for distributed state operations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def increment(self, key: str, amount: int = 1, expire: Optional[int] = None) -> int:
        pass
    
    @abstractmethod
    async def setnx(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        """Set if not exists (for distributed locks)."""
        pass
    
    @abstractmethod
    async def pubsub_subscribe(self, channel: str, handler: Callable[[str], None]) -> None:
        pass
    
    @abstractmethod
    async def pubsub_publish(self, channel: str, message: str) -> int:
        pass
    
    @abstractmethod
    async def close(self) -> None:
        pass


class InMemoryStateBackend(StateBackend):
    """In-memory state backend for single-instance operation."""
    
    def __init__(self):
        self._data: Dict[str, str] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._pubsub_channels: Dict[str, List[Callable[[str], None]]] = {}
    
    def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]
    
    async def get(self, key: str) -> Optional[str]:
        return self._data.get(key)
    
    async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        self._data[key] = value
        return True
    
    async def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        return key in self._data
    
    async def increment(self, key: str, amount: int = 1, expire: Optional[int] = None) -> int:
        async with self._get_lock(key):
            current = int(self._data.get(key, "0"))
            new_value = current + amount
            self._data[key] = str(new_value)
            return new_value
    
    async def setnx(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        async with self._get_lock(key):
            if key not in self._data:
                self._data[key] = value
                return True
            return False
    
    async def pubsub_subscribe(self, channel: str, handler: Callable[[str], None]) -> None:
        if channel not in self._pubsub_channels:
            self._pubsub_channels[channel] = []
        self._pubsub_channels[channel].append(handler)
    
    async def pubsub_publish(self, channel: str, message: str) -> int:
        handlers = self._pubsub_channels.get(channel, [])
        for handler in handlers:
            try:
                handler(message)
            except Exception as e:
                logger.warning(f"Pubsub handler error on {channel}: {e}")
        return len(handlers)
    
    async def close(self) -> None:
        self._data.clear()
        self._locks.clear()
        self._pubsub_channels.clear()


class RedisStateBackend(StateBackend):
    """Redis-backed state backend for multi-instance operation."""
    
    def __init__(self, redis_url: str = REDIS_URL):
        self.redis_url = redis_url
        self._redis = None
        self._pubsub = None
        self._pubsub_tasks: List[asyncio.Task] = []
    
    async def _ensure_connected(self):
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(self.redis_url, decode_responses=True)
                await self._redis.ping()
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
    
    async def get(self, key: str) -> Optional[str]:
        await self._ensure_connected()
        return await self._redis.get(key)
    
    async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        await self._ensure_connected()
        if expire:
            return await self._redis.set(key, value, ex=expire)
        return await self._redis.set(key, value)
    
    async def delete(self, key: str) -> bool:
        await self._ensure_connected()
        result = await self._redis.delete(key)
        return result > 0
    
    async def exists(self, key: str) -> bool:
        await self._ensure_connected()
        return await self._redis.exists(key) > 0
    
    async def increment(self, key: str, amount: int = 1, expire: Optional[int] = None) -> int:
        await self._ensure_connected()
        pipe = self._redis.pipeline()
        pipe.incrby(key, amount)
        if expire:
            pipe.expire(key, expire)
        results = await pipe.execute()
        return results[0]
    
    async def setnx(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        await self._ensure_connected()
        result = await self._redis.set(key, value, nx=True, ex=expire)
        return result is not None
    
    async def pubsub_subscribe(self, channel: str, handler: Callable[[str], None]) -> None:
        await self._ensure_connected()
        self._pubsub = self._redis.pubsub()
        await self._pubsub.subscribe(channel)
        
        async def listen():
            try:
                async for message in self._pubsub.listen():
                    if message["type"] == "message":
                        try:
                            handler(message["data"])
                        except Exception as e:
                            logger.warning(f"Pubsub handler error on {channel}: {e}")
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Pubsub listen error on {channel}: {e}")
        
        task = asyncio.create_task(listen())
        self._pubsub_tasks.append(task)
    
    async def pubsub_publish(self, channel: str, message: str) -> int:
        await self._ensure_connected()
        return await self._redis.publish(channel, message)
    
    async def close(self) -> None:
        for task in self._pubsub_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()


# Global state backend instance
_state_backend: Optional[StateBackend] = None


def get_state_backend() -> StateBackend:
    """Get the global state backend instance."""
    global _state_backend
    if _state_backend is None:
        if USE_REDIS:
            try:
                _state_backend = RedisStateBackend()
                logger.info("Using Redis state backend for distributed operation")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis backend, falling back to in-memory: {e}")
                _state_backend = InMemoryStateBackend()
        else:
            _state_backend = InMemoryStateBackend()
            logger.info("Using in-memory state backend (single-instance mode)")
    return _state_backend


async def close_state_backend():
    """Close the global state backend."""
    global _state_backend
    if _state_backend:
        await _state_backend.close()
        _state_backend = None


# Distributed locking utilities
class DistributedLock:
    """Distributed lock using Redis SETNX or in-memory fallback."""
    
    def __init__(self, key: str, backend: Optional[StateBackend] = None, ttl: int = 30):
        self.key = f"lock:{key}"
        self.backend = backend or get_state_backend()
        self.ttl = ttl
        self._acquired = False
    
    async def acquire(self, timeout: float = 10.0) -> bool:
        """Try to acquire the lock."""
        start = asyncio.get_event_loop().time()
        while True:
            acquired = await self.backend.setnx(self.key, "1", expire=self.ttl)
            if acquired:
                self._acquired = True
                return True
            if asyncio.get_event_loop().time() - start > timeout:
                return False
            await asyncio.sleep(0.1)
    
    async def release(self) -> bool:
        """Release the lock."""
        if self._acquired:
            await self.backend.delete(self.key)
            self._acquired = False
            return True
        return False
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()


# Rate limiting using distributed state
class DistributedRateLimiter:
    """Distributed rate limiter using sliding window."""
    
    def __init__(self, key_prefix: str, backend: Optional[StateBackend] = None):
        self.key_prefix = f"ratelimit:{key_prefix}"
        self.backend = backend or get_state_backend()
    
    async def is_allowed(self, identifier: str, max_requests: int, window_seconds: int) -> tuple[bool, dict]:
        """
        Check if request is allowed.
        Returns (allowed, info_dict) where info_dict contains limit info.
        """
        key = f"{self.key_prefix}:{identifier}"
        now = int(asyncio.get_event_loop().time())
        window_start = now - window_seconds
        
        # Clean old entries and add current request
        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local max_requests = tonumber(ARGV[3])
        local window_seconds = tonumber(ARGV[4])
        
        -- Remove expired entries
        redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
        
        -- Count current requests
        local count = redis.call('ZCARD', key)
        
        if count < max_requests then
            redis.call('ZADD', key, now, now .. '-' .. math.random())
            redis.call('EXPIRE', key, window_seconds)
            return {1, count + 1}
        else
            return {0, count}
        end
        """
        
        try:
            if isinstance(self.backend, RedisStateBackend):
                await self.backend._ensure_connected()
                result = await self.backend._redis.eval(
                    lua_script, 1, key, now, window_start, max_requests, window_seconds
                )
                allowed = bool(result[0])
                current_count = int(result[1])
            else:
                # Fallback for in-memory (simplified)
                allowed = True
                current_count = 0
        except Exception as e:
            logger.warning(f"Rate limiter error, allowing request: {e}")
            allowed = True
            current_count = 0
        
        info = {
            "limit": max_requests,
            "remaining": max(0, max_requests - current_count),
            "reset": now + window_seconds,
        }
        
        return allowed, info


# Rate limiter factory
def get_rate_limiter(prefix: str) -> DistributedRateLimiter:
    return DistributedRateLimiter(prefix)


# Call state management for horizontal scaling
class CallStateManager:
    """
    Manages call state across instances.
    Uses database for persistence and Redis for real-time coordination.
    """
    
    def __init__(self, backend: Optional[StateBackend] = None):
        self.backend = backend or get_state_backend()
    
    async def set_call_active(self, call_id: str, user_id: int, metadata: Dict[str, Any] = None) -> bool:
        """Mark a call as active on this instance."""
        key = f"call:active:{call_id}"
        data = {
            "instance_id": os.getpid(),
            "user_id": user_id,
            "started_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        return await self.backend.set(key, json.dumps(data), expire=3600)
    
    async def remove_call_active(self, call_id: str) -> bool:
        """Mark a call as no longer active."""
        key = f"call:active:{call_id}"
        return await self.backend.delete(key)
    
    async def get_call_instance(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Get which instance is handling a call."""
        key = f"call:active:{call_id}"
        data = await self.backend.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def broadcast_call_event(self, call_id: str, event: str, data: Dict[str, Any]) -> int:
        """Broadcast an event to all instances handling this call."""
        channel = f"call:events:{call_id}"
        message = json.dumps({"event": event, "data": data, "timestamp": datetime.utcnow().isoformat()})
        return await self.backend.pubsub_publish(channel, message)
    
    async def subscribe_call_events(self, call_id: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to call events."""
        channel = f"call:events:{call_id}"
        async def wrapped_handler(message: str):
            try:
                data = json.loads(message)
                handler(data)
            except Exception as e:
                logger.warning(f"Call event handler error: {e}")
        await self.backend.pubsub_subscribe(channel, wrapped_handler)


# Speaker cache management
class SpeakerCacheManager:
    """
    Manages speaker embedding cache across instances.
    Invalidates cache when embeddings are updated.
    """
    
    def __init__(self, backend: Optional[StateBackend] = None):
        self.backend = backend or get_state_backend()
    
    async def invalidate_speaker(self, speaker_id: str) -> int:
        """Invalidate speaker cache across all instances."""
        channel = f"speaker:invalidate:{speaker_id}"
        message = json.dumps({"speaker_id": speaker_id, "timestamp": datetime.utcnow().isoformat()})
        return await self.backend.pubsub_publish(channel, message)
    
    async def subscribe_invalidations(self, handler: Callable[[str], None]) -> None:
        """Subscribe to speaker invalidation events."""
        channel = "speaker:invalidate:*"
        await self.backend.pubsub_subscribe(channel, handler)


# Export
__all__ = [
    "StateBackend",
    "InMemoryStateBackend",
    "RedisStateBackend",
    "get_state_backend",
    "close_state_backend",
    "DistributedLock",
    "DistributedRateLimiter",
    "get_rate_limiter",
    "CallStateManager",
    "SpeakerCacheManager",
    "USE_REDIS",
]