"""
Database models and configuration for RepoSynth.

This module defines the SQLAlchemy models for job tracking and database setup.
Jobs progress through states: pending -> processing -> completed/failed
"""

import os
import datetime
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, JSON, Date
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

# Get database URL from environment
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Create database engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()


class Job(Base):
    """
    Job model representing a repository analysis task.
    
    Attributes:
        id: Unique job identifier (UUID)
        status: Current job state (pending, processing, completed, failed)
        repo_url: URL of the git repository to analyze
        mode: Analysis mode (semantic, hybrid, full)
        created_at: Timestamp when job was created
        started_at: Timestamp when job processing started
        finished_at: Timestamp when job completed or failed
        result_url: URL to download the result pack from S3
        error_message: Error details if job failed
        metadata: Additional job metadata (JSON)
    """
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    status = Column(String, default="pending", nullable=False, index=True)
    repo_url = Column(String, nullable=False)
    mode = Column(String, default="semantic", nullable=False)
    config = Column(JSON, nullable=True, comment="Full job configuration JSON")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    
    # Results
    result_url = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Statistics
    processing_time_seconds = Column(Integer, nullable=True)
    pack_size_bytes = Column(Integer, nullable=True)
    
    def __repr__(self):
        return f"<Job(id={self.id}, status={self.status}, repo_url={self.repo_url})>"
    
    def to_dict(self):
        """Convert job to dictionary for API responses."""
        return {
            "id": self.id,
            "status": self.status,
            "repo_url": self.repo_url,
            "mode": self.mode,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "result_url": self.result_url,
            "error_message": self.error_message,
            "processing_time_seconds": self.processing_time_seconds,
            "pack_size_bytes": self.pack_size_bytes,
        }


class RateLimitRecord(Base):
    """
    Rate limit tracking for IP-based API limits.
    
    Attributes:
        ip_address: Client IP address
        date: Date of the rate limit record
        request_count: Number of requests made on this date
        last_request_at: Timestamp of the last request
    """
    __tablename__ = "rate_limits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ip_address = Column(String, nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    request_count = Column(Integer, default=0, nullable=False)
    last_request_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<RateLimitRecord(ip={self.ip_address}, date={self.date}, count={self.request_count})>"


def create_tables():
    """
    Create all database tables.
    Called on application startup to ensure schema exists.
    """
    try:
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError as e:
        raise RuntimeError(f"Failed to create database tables: {e}")


def get_db():
    """
    Dependency injection for database sessions.
    Yields a database session and ensures it's closed after use.
    
    Usage in FastAPI:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
