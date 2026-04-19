"""
Database connection and session management.

Supports SQLite for development and PostgreSQL for production.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from pathlib import Path
import os
from typing import Optional

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./nba_stats.db"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    from db_schema import Base
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables initialized")
