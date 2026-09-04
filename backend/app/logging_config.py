"""
Structured logging configuration for VoiceGuard.

Provides consistent, structured logging across the application with:
- JSON formatting for production
- Human-readable formatting for development
- Contextual fields (call_id, speaker_id, user_id, etc.)
- Appropriate log levels
"""
from __future__ import annotations
import os
import sys
import logging
import logging.config
import json
from datetime import datetime
from typing import Optional, Dict, Any
from contextvars import ContextVar

# Context variables for adding contextual information to logs
call_id_var: ContextVar[Optional[str]] = ContextVar('call_id', default=None)
speaker_id_var: ContextVar[Optional[str]] = ContextVar('speaker_id', default=None)
user_id_var: ContextVar[Optional[int]] = ContextVar('user_id', default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class ContextFilter(logging.Filter):
    """Add contextual fields to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.call_id = call_id_var.get()
        record.speaker_id = speaker_id_var.get()
        record.user_id = user_id_var.get()
        record.request_id = request_id_var.get()
        return True


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for production logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add contextual fields if present
        for field in ('call_id', 'speaker_id', 'user_id', 'request_id'):
            value = getattr(record, field, None)
            if value is not None:
                log_data[field] = value
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in log_data and not key.startswith('_'):
                try:
                    json.dumps(value)  # Test serializability
                    log_data[key] = value
                except (TypeError, ValueError):
                    pass
        
        return json.dumps(log_data)


class HumanFormatter(logging.Formatter):
    """Human-readable formatter for development."""
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Build context prefix
        context_parts = []
        for field in ('call_id', 'speaker_id', 'user_id', 'request_id'):
            value = getattr(record, field, None)
            if value is not None:
                context_parts.append(f"{field}={value}")
        
        context = f" [{', '.join(context_parts)}]" if context_parts else ""
        
        return f"{timestamp} {record.levelname:8s} {record.name}{context} - {record.getMessage()}"


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None
) -> None:
    """
    Configure application logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON formatting (for production)
        log_file: Optional file path for file logging
    """
    # Determine format
    if json_format or os.environ.get("LOG_JSON", "false").lower() == "true":
        formatter_class = JSONFormatter
    else:
        formatter_class = HumanFormatter
    
    # Base configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": formatter_class,
            },
        },
        "filters": {
            "context": {
                "()": ContextFilter,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "filters": ["context"],
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
        "loggers": {
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "sqlalchemy.engine": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
            "app": {
                "level": level,
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }
    
    # Add file handler if requested
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "filters": ["context"],
            "filename": log_file,
            "maxBytes": 10_485_760,  # 10MB
            "backupCount": 5,
        }
        config["root"]["handlers"].append("file")
        for logger_config in config["loggers"].values():
            logger_config["handlers"].append("file")
    
    logging.config.dictConfig(config)
    
    # Set level for app loggers
    logging.getLogger("app").setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the app namespace."""
    return logging.getLogger(f"app.{name}")


def set_call_context(call_id: Optional[str] = None, speaker_id: Optional[str] = None, 
                     user_id: Optional[int] = None, request_id: Optional[str] = None) -> None:
    """Set contextual variables for the current execution context."""
    if call_id is not None:
        call_id_var.set(call_id)
    if speaker_id is not None:
        speaker_id_var.set(speaker_id)
    if user_id is not None:
        user_id_var.set(user_id)
    if request_id is not None:
        request_id_var.set(request_id)


def clear_call_context() -> None:
    """Clear all contextual variables."""
    call_id_var.set(None)
    speaker_id_var.set(None)
    user_id_var.set(None)
    request_id_var.set(None)


# Convenience functions for common log patterns
def log_call_event(logger: logging.Logger, call_id: str, event: str, **kwargs) -> None:
    """Log a call-related event with structured data."""
    set_call_context(call_id=call_id)
    try:
        logger.info(f"Call event: {event}", extra=kwargs)
    finally:
        clear_call_context()


def log_speaker_event(logger: logging.Logger, speaker_id: str, event: str, **kwargs) -> None:
    """Log a speaker-related event."""
    set_call_context(speaker_id=speaker_id)
    try:
        logger.info(f"Speaker event: {event}", extra=kwargs)
    finally:
        clear_call_context()


def log_auth_event(logger: logging.Logger, user_id: int, event: str, **kwargs) -> None:
    """Log an authentication event."""
    set_call_context(user_id=user_id)
    try:
        logger.info(f"Auth event: {event}", extra=kwargs)
    finally:
        clear_call_context()


def log_security_event(logger: logging.Logger, event: str, severity: str = "WARNING", **kwargs) -> None:
    """Log a security-related event."""
    log_method = getattr(logger, severity.lower(), logger.warning)
    log_method(f"Security event: {event}", extra=kwargs)


# Initialize logging on import if not already configured
if not logging.getLogger().handlers:
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    json_logs = os.environ.get("LOG_JSON", "false").lower() == "true"
    setup_logging(level=log_level, json_format=json_logs)


# Export
__all__ = [
    "setup_logging",
    "get_logger",
    "set_call_context",
    "clear_call_context",
    "log_call_event",
    "log_speaker_event",
    "log_auth_event",
    "log_security_event",
    "ContextFilter",
    "JSONFormatter",
    "HumanFormatter",
]