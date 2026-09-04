from __future__ import annotations
import os
from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, JSON, LargeBinary, Boolean, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

# Default to SQLite for zero-setup local dev; swap DATABASE_URL to a
# postgresql+psycopg2:// URL in production per the architecture doc.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./voiceguard.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String)
    full_name: Mapped[str | None] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class TrustedSpeaker(Base):
    __tablename__ = "trusted_speakers"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    enrollment_clips: Mapped[int] = mapped_column(Integer, default=0)
    # ECAPA-TDNN embedding (192-dim float32) stored as binary blob
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    # Embedding version for future compatibility (model version, dimension)
    embedding_version: Mapped[int] = mapped_column(Integer, default=1)
    # Owner user ID (for multi-user deployments)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Call(Base):
    __tablename__ = "calls"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    claimed_speaker_id: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    final_risk_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    final_risk_level: Mapped[str | None] = mapped_column(String, nullable=True)
    demo_scenario: Mapped[str | None] = mapped_column(String, nullable=True)
    # Owner user ID (for multi-user deployments)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    snapshots: Mapped[list["Snapshot"]] = relationship(back_populates="call")


class Snapshot(Base):
    __tablename__ = "snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    elapsed_seconds: Mapped[float] = mapped_column(Float)
    risk_score: Mapped[int] = mapped_column(Integer)
    risk_level: Mapped[str] = mapped_column(String)
    trust_score: Mapped[int] = mapped_column(Integer)
    reasons: Mapped[dict] = mapped_column(JSON)
    payload: Mapped[dict] = mapped_column(JSON)  # full snapshot JSON for detail view

    call: Mapped["Call"] = relationship(back_populates="snapshots")


class Setting(Base):
    """Global and user-specific settings."""
    __tablename__ = "settings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String, index=True)
    value: Mapped[str] = mapped_column(String)
    value_type: Mapped[str] = mapped_column(String, default="string")  # string, int, float, bool, json
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    is_global: Mapped[bool] = mapped_column(Boolean, default=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # for user-specific settings
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
