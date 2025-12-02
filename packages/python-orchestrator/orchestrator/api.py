# --- FILE: packages/python-orchestrator/orchestrator/api.py ---
"""
FastAPI application for RepoSynth.
Week 6: Token estimation API endpoint.
Week 8: Job queue and background processing.
Week 10: Production security - Rate limiting, authentication, CORS.
"""

import logging
import datetime
from fastapi import FastAPI, HTTPException, status, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from pathlib import Path
import os
import uuid
import boto3
import tempfile
import json
from typing import Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

from .schemas import (
    EstimateRequest,
    EstimateResponse,
    HealthResponse,
    LanguageStats as LanguageStatsSchema,
    FeatureEstimate as FeatureEstimateSchema,
    GitHubEstimateRequest,
    JobConfiguration,
    ConfiguratorEstimateRequest,
    ConfiguratorEstimateResponse,
    VibePromptRequest,
    VibePromptResponse
)
from .prompt_engine import generate_vibe_prompt, suggest_entry_points
from .estimator import estimate_tokens, TIKTOKEN_AVAILABLE, PYGOUNT_AVAILABLE
from .git_utils import clone_repository, cleanup_cloned_repo

# Week 8: Database and worker imports
from .database import create_tables, get_db, Job, RateLimitRecord
from . import worker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# S3/MinIO Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "reposynth-packs")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")

# API Configuration
API_KEY = os.environ.get("REPOSYNTH_API_KEY", "")  # Optional API key for additional auth
DAILY_RATE_LIMIT = int(os.environ.get("DAILY_RATE_LIMIT", "5"))  # Requests per day per IP
DISABLE_RATE_LIMIT = os.environ.get("DISABLE_RATE_LIMIT", "false").lower() == "true"  # Disable for local dev

# Allowed origins for CORS (production domains)
ALLOWED_ORIGINS = [
    "https://www.reposynth.com",
    "https://reposynth.com",
    "https://reposynth.vercel.app",
]

# Add localhost and development origins if enabled
if os.environ.get("ALLOW_DEV_ORIGINS", "false").lower() == "true":
    ALLOWED_ORIGINS.extend([
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ])
    # Also allow any IP-based access (for local network development)
    ALLOWED_ORIGINS.append("*")

# Initialize S3 client
s3_client = None
try:
    s3_client = boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
except Exception as e:
    logger.warning(f"Failed to initialize S3 client: {e}")

# Application metadata
__version__ = "2.1.0"  # Week 10 update - Production security

# ============================================================================
# Rate Limiter Setup
# ============================================================================

def get_client_ip(request: Request) -> str:
    """Extract client IP, considering proxy headers."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host if request.client else "unknown"


def get_client_identifier(request: Request) -> str:
    """
    Get a unique client identifier combining IP and device fingerprint.
    This helps distinguish different devices behind the same IP (NAT/proxy).
    
    The client can send an X-Device-ID header with a unique browser fingerprint
    (generated client-side using localStorage or fingerprint.js).
    """
    ip = get_client_ip(request)
    
    # Check for device ID header (should be set by frontend)
    device_id = request.headers.get("X-Device-ID")
    
    if device_id:
        # Combine IP and device ID for unique identification
        return f"{ip}:{device_id}"
    
    # Fallback to user-agent as a weak differentiator
    user_agent = request.headers.get("User-Agent", "")
    if user_agent:
        # Use a hash of user-agent to keep the identifier shorter
        import hashlib
        ua_hash = hashlib.md5(user_agent.encode()).hexdigest()[:8]
        return f"{ip}:{ua_hash}"
    
    return ip


def is_localhost(request: Request) -> bool:
    """Check if request is from localhost."""
    ip = get_client_ip(request)
    return ip in ["127.0.0.1", "localhost", "::1"]

limiter = Limiter(key_func=get_client_ip)

# ============================================================================
# FastAPI Application
# ============================================================================

# Create FastAPI application
app = FastAPI(
    title="RepoSynth API",
    description="Token estimation and repository analysis pipeline with background job processing",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiter to app state
app.state.limiter = limiter

# Mount worker_packs directory to serve generated files locally
# This allows downloading packs without S3/MinIO if needed
if os.environ.get("WORKER_PACKS_DIR"):
    worker_packs_dir = Path(os.environ.get("WORKER_PACKS_DIR"))
elif Path("/app/worker_packs").exists() or os.environ.get("DOCKER_ENV"):
    worker_packs_dir = Path("/app/worker_packs")
else:
    # Local development fallback (relative to project root)
    # api.py is in packages/python-orchestrator/orchestrator/
    # Project root is 3 levels up
    project_root = Path(__file__).parent.parent.parent.parent
    worker_packs_dir = project_root / "worker_packs"

worker_packs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/packs", StaticFiles(directory=str(worker_packs_dir)), name="packs")

# ============================================================================
# Middleware
# ============================================================================

# CORS - Only allow specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
)

# Rate limit middleware
app.add_middleware(SlowAPIMiddleware)


# ============================================================================
# Rate Limit Helpers
# ============================================================================

def get_rate_limit_info(db: Session, client_id: str, request: Request = None) -> Dict:
    """
    Get current rate limit status for a client.
    
    Args:
        db: Database session
        client_id: Unique client identifier (IP + device fingerprint)
        request: Optional request object to check for localhost bypass
    """
    # Check if rate limiting is disabled
    if DISABLE_RATE_LIMIT:
        return {
            "limit": -1,
            "remaining": -1,
            "reset_at": None,
            "unlimited": True
        }
    
    # Skip rate limiting entirely when dev origins are allowed (development mode)
    if os.environ.get("ALLOW_DEV_ORIGINS", "false").lower() == "true":
        return {
            "limit": -1,
            "remaining": -1,
            "reset_at": None,
            "unlimited": True
        }
    
    today = datetime.date.today()
    
    record = db.query(RateLimitRecord).filter(
        RateLimitRecord.ip_address == client_id,
        RateLimitRecord.date == today
    ).first()
    
    if not record:
        return {
            "limit": DAILY_RATE_LIMIT,
            "remaining": DAILY_RATE_LIMIT,
            "reset_at": datetime.datetime.combine(
                today + datetime.timedelta(days=1),
                datetime.time.min
            ).isoformat() + "Z"
        }
    
    remaining = max(0, DAILY_RATE_LIMIT - record.request_count)
    reset_at = datetime.datetime.combine(
        today + datetime.timedelta(days=1),
        datetime.time.min
    )
    
    return {
        "limit": DAILY_RATE_LIMIT,
        "remaining": remaining,
        "reset_at": reset_at.isoformat() + "Z"
    }


def check_and_increment_rate_limit(db: Session, client_id: str, request: Request = None) -> Dict:
    """
    Check rate limit and increment counter. Raises HTTPException if exceeded.
    
    Args:
        db: Database session
        client_id: Unique client identifier (IP + device fingerprint)
        request: Optional request object to check for localhost bypass
    """
    # Bypass rate limiting if disabled or if localhost and dev mode
    if DISABLE_RATE_LIMIT:
        return {
            "limit": -1,  # -1 indicates unlimited
            "remaining": -1,
            "reset_at": None,
            "unlimited": True
        }
    
    # Skip rate limiting entirely when dev origins are allowed (development mode)
    if os.environ.get("ALLOW_DEV_ORIGINS", "false").lower() == "true":
        return {
            "limit": -1,
            "remaining": -1,
            "reset_at": None,
            "unlimited": True
        }
    
    today = datetime.date.today()
    
    record = db.query(RateLimitRecord).filter(
        RateLimitRecord.ip_address == client_id,
        RateLimitRecord.date == today
    ).first()
    
    reset_at = datetime.datetime.combine(
        today + datetime.timedelta(days=1),
        datetime.time.min
    )
    
    if not record:
        record = RateLimitRecord(
            ip_address=client_id,
            date=today,
            request_count=1,
            last_request_at=datetime.datetime.utcnow()
        )
        db.add(record)
        db.commit()
        
        return {
            "limit": DAILY_RATE_LIMIT,
            "remaining": DAILY_RATE_LIMIT - 1,
            "reset_at": reset_at.isoformat() + "Z"
        }
    
    if record.request_count >= DAILY_RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "message": f"You have exceeded the daily limit of {DAILY_RATE_LIMIT} API calls.",
                "limit": DAILY_RATE_LIMIT,
                "remaining": 0,
                "reset_at": reset_at.isoformat() + "Z"
            }
        )
    
    record.request_count += 1
    record.last_request_at = datetime.datetime.utcnow()
    db.commit()
    
    return {
        "limit": DAILY_RATE_LIMIT,
        "remaining": DAILY_RATE_LIMIT - record.request_count,
        "reset_at": reset_at.isoformat() + "Z"
    }


# ============================================================================
# Authentication
# ============================================================================

def verify_origin(request: Request):
    """Verify the request comes from an allowed origin."""
    origin = request.headers.get("origin") or request.headers.get("referer")
    
    # Allow requests without origin header (direct API calls, health checks)
    if not origin:
        return True
    
    # Check if origin matches allowed list
    for allowed in ALLOWED_ORIGINS:
        if origin.startswith(allowed):
            return True
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Origin not allowed"
    )


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key if configured."""
    if not API_KEY:
        return True
    
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    return True


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors from SlowAPI."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": "Rate limit exceeded. Please try again later."
        }
    )


@app.on_event("startup")
def on_startup():
    """Initialize database tables on application startup."""
    logger.info("Starting RepoSynth API...")
    create_tables()
    logger.info("Database initialized")


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors with proper HTTP status."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )


# ============================================================================
# Public Endpoints (No Rate Limit)
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": "RepoSynth API",
        "version": __version__,
        "description": "Token estimation and repository analysis",
        "endpoints": {
            "health": "/health",
            "estimate": "/estimate-tokens",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint. Returns the status of the API and its dependencies."""
    database_healthy = False
    try:
        db.execute(text("SELECT 1"))
        database_healthy = True
    except Exception:
        pass

    dependencies: Dict[str, bool] = {
        "tiktoken": TIKTOKEN_AVAILABLE,
        "pygount": PYGOUNT_AVAILABLE,
        "database": database_healthy,
        "worker_queue": True,
    }

    is_healthy = PYGOUNT_AVAILABLE and database_healthy

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        version=__version__,
        dependencies=dependencies
    )


@app.get("/rate-limit-status", tags=["Rate Limiting"])
async def get_rate_limit_status(request: Request, db: Session = Depends(get_db)):
    """Get current rate limit status for your IP/device."""
    client_id = get_client_identifier(request)
    rate_info = get_rate_limit_info(db, client_id, request)
    
    # Check if unlimited (dev mode)
    if rate_info.get("unlimited"):
        return {
            "client_id": client_id,
            "daily_limit": "unlimited",
            "remaining_calls": "unlimited",
            "reset_at": None,
            "message": "Rate limiting is disabled (development mode)."
        }
    
    return {
        "client_id": client_id,
        "daily_limit": rate_info["limit"],
        "remaining_calls": rate_info["remaining"],
        "reset_at": rate_info["reset_at"],
        "message": f"You have {rate_info['remaining']} API calls remaining today."
    }


# ============================================================================
# Rate-Limited Endpoints
# ============================================================================

@app.post("/estimate-tokens", response_model=EstimateResponse, tags=["Estimation"])
async def estimate_tokens_endpoint(
    request: EstimateRequest,
    req: Request = None,
    db: Session = Depends(get_db),
    _origin: bool = Depends(verify_origin)
):
    """
    Estimate tokens and time for repository analysis.
    """
    client_id = get_client_identifier(req)
    rate_info = check_and_increment_rate_limit(db, client_id, req)
    
    repo_path = Path(request.repo_path)

    if not repo_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {request.repo_path}"
        )

    if not repo_path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path is not a directory: {request.repo_path}"
        )

    try:
        result = estimate_tokens(repo_path, request.config.model_dump())

        language_breakdown = {
            lang: LanguageStatsSchema(
                files=stats.files,
                lines=stats.lines,
                code=stats.code,
                comments=stats.comments,
                blanks=stats.blanks,
                estimated_tokens=stats.estimated_tokens
            )
            for lang, stats in result.language_breakdown.items()
        }

        feature_breakdown = {
            feature: FeatureEstimateSchema(
                enabled=estimate.enabled,
                estimated_tokens=estimate.estimated_tokens,
                estimated_time_seconds=estimate.estimated_time_seconds,
                description=estimate.description
            )
            for feature, estimate in result.feature_breakdown.items()
        }

        return EstimateResponse(
            total_tokens=result.total_tokens,
            total_time_seconds=result.total_time_seconds,
            base_tokens=result.base_tokens,
            num_files=result.num_files,
            total_lines=result.total_lines,
            language_breakdown=language_breakdown,
            feature_breakdown=feature_breakdown,
            summary=result.summary,
            warnings=result.warnings,
            rate_limit=rate_info
        )

    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Missing required dependency: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error during estimation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Estimation failed: {str(e)}"
        )


@app.post("/estimate-tokens-from-github", response_model=EstimateResponse, tags=["Estimation"])
async def estimate_tokens_from_github_endpoint(
    request: GitHubEstimateRequest,
    req: Request = None,
    db: Session = Depends(get_db),
    _origin: bool = Depends(verify_origin)
):
    """Estimate tokens and time for a GitHub repository."""
    client_id = get_client_identifier(req)
    rate_info = check_and_increment_rate_limit(db, client_id, req)
    
    project_root = Path(__file__).parent.parent.parent.parent
    temp_repos_dir = project_root / "temp_repos"
    temp_repos_dir.mkdir(parents=True, exist_ok=True)

    cloned_repo_path = None

    try:
        cloned_repo_path = clone_repository(
            git_url=request.repo_url,
            temp_dir=temp_repos_dir,
            force_reclone=True
        )

        result = estimate_tokens(cloned_repo_path, request.config.model_dump())

        language_breakdown = {
            lang: LanguageStatsSchema(
                files=stats.files,
                lines=stats.lines,
                code=stats.code,
                comments=stats.comments,
                blanks=stats.blanks,
                estimated_tokens=stats.estimated_tokens
            )
            for lang, stats in result.language_breakdown.items()
        }

        feature_breakdown = {
            feature: FeatureEstimateSchema(
                enabled=estimate.enabled,
                estimated_tokens=estimate.estimated_tokens,
                estimated_time_seconds=estimate.estimated_time_seconds,
                description=estimate.description
            )
            for feature, estimate in result.feature_breakdown.items()
        }

        return EstimateResponse(
            total_tokens=result.total_tokens,
            total_time_seconds=result.total_time_seconds,
            base_tokens=result.base_tokens,
            num_files=result.num_files,
            total_lines=result.total_lines,
            language_breakdown=language_breakdown,
            feature_breakdown=feature_breakdown,
            summary=result.summary,
            warnings=result.warnings,
            rate_limit=rate_info
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Missing required dependency: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error during GitHub estimation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Estimation failed: {str(e)}"
        )
    finally:
        if request.cleanup and cloned_repo_path and cloned_repo_path.exists():
            try:
                cleanup_cloned_repo(cloned_repo_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup cloned repo: {e}")


@app.post("/estimate", response_model=ConfiguratorEstimateResponse, tags=["Estimation"])
async def estimate_from_config(
    request: ConfiguratorEstimateRequest,
    req: Request = None,
    db: Session = Depends(get_db),
    _origin: bool = Depends(verify_origin)
):
    """Estimate tokens and cost based on configuration without cloning the repository."""
    client_id = get_client_identifier(req)
    rate_info = check_and_increment_rate_limit(db, client_id, req)
    
    config = request.config
    
    base_estimates = {
        "semantic": {"tokens": 50000, "time": 30},
        "hybrid": {"tokens": 100000, "time": 60},
        "full": {"tokens": 200000, "time": 120}
    }
    
    estimate = base_estimates.get(config.mode, base_estimates["semantic"])
    estimated_tokens = estimate["tokens"]
    estimated_time = float(estimate["time"])
    
    feature_overhead = {
        "enable_ast": 0.3,
        "enable_imports": 0.2,
        "enable_complexity": 0.15,
        "enable_security": 0.25,
        "enable_embeddings": 0.4
    }
    
    # Apply feature toggles
    features_enabled = {}
    for feature, overhead in feature_overhead.items():
        enabled = getattr(config, feature, True)
        features_enabled[feature] = enabled
        if enabled:
            estimated_tokens = int(estimated_tokens * (1 + overhead))
            estimated_time = estimated_time * (1 + overhead * 0.5)
    
    # Calculate cost (example: $0.0001 per 1K tokens)
    estimated_cost = (estimated_tokens / 1000) * 0.0001
    
    # Generate warnings
    warnings = []
    if config.mode == "full":
        warnings.append("Full mode may take 2-5 minutes for large repositories")
    if config.enable_security:
        warnings.append("Security scanning significantly increases processing time")
    if estimated_tokens > 500000:
        warnings.append("Large token count - consider using semantic mode instead")
    
    return ConfiguratorEstimateResponse(
        estimated_tokens=estimated_tokens,
        estimated_time_seconds=round(estimated_time, 1),
        estimated_cost_usd=round(estimated_cost, 6),
        mode=config.mode,
        features_enabled=features_enabled,
        warnings=warnings,
        rate_limit=rate_info
    )


# ============================================================================
# Week 8: Job Queue Endpoints
# ============================================================================

@app.post("/jobs", status_code=status.HTTP_202_ACCEPTED, tags=["Jobs"])
async def create_job(
    repo_url: str,
    request: Request,
    config: Optional[JobConfiguration] = None,
    db: Session = Depends(get_db),
    _origin: bool = Depends(verify_origin)
):
    """Submit a repository analysis job for background processing."""
    client_id = get_client_identifier(request)
    rate_info = check_and_increment_rate_limit(db, client_id, request)
    
    if config is None:
        config = JobConfiguration()
    
    mode = config.mode
    
    if mode not in ["semantic", "hybrid", "full"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode '{mode}'. Must be 'semantic', 'hybrid', or 'full'."
        )
    
    if not repo_url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid repository URL. Must start with http:// or https://"
        )
    
    job_id = str(uuid.uuid4())
    
    new_job = Job(
        id=job_id,
        repo_url=repo_url,
        mode=mode,
        status="pending",
        config=config.model_dump()
    )
    
    try:
        db.add(new_job)
        db.commit()
        db.refresh(new_job)
        
        worker.process_repository.send(job_id, repo_url, config.model_dump())
        
        logger.info(f"Created job {job_id} for {repo_url} (mode: {mode})")
        
        return {
            "job_id": job_id,
            "status": "pending",
            "repo_url": repo_url,
            "mode": mode,
            "config": config.model_dump(),
            "message": "Job submitted successfully. Use GET /jobs/{job_id} to check status.",
            "rate_limit": rate_info
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}"
        )


@app.get("/jobs/{job_id}", tags=["Jobs"])
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Get the status of a submitted job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}"
        )
    
    return job.to_dict()


@app.get("/jobs", tags=["Jobs"])
async def list_jobs(
    status_filter: str = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List all jobs with optional filtering."""
    limit = min(limit, 200)
    
    query = db.query(Job)
    
    if status_filter:
        if status_filter not in ["pending", "processing", "completed", "failed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status filter: {status_filter}"
            )
        query = query.filter(Job.status == status_filter)
    
    total = query.count()
    
    jobs = query.order_by(Job.created_at.desc()).offset(offset).limit(limit).all()
    
    return {
        "jobs": [job.to_dict() for job in jobs],
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + limit) < total
    }


@app.get("/jobs/{job_id}/files", tags=["Jobs"])
async def get_job_files(job_id: str):
    """Get list of files and suggested entry points for a job."""
    pack_path = worker_packs_dir / job_id
    
    graph_candidates = [
        pack_path / "import_graph.json",
        pack_path / "pack" / "import_graph.json",
    ]
    
    graph_path = None
    for candidate in graph_candidates:
        if candidate.exists():
            graph_path = candidate
            break
    
    # If not found locally, try to read from a zip file
    if not graph_path or not graph_path.exists():
        zip_files = list(pack_path.glob("*.zip")) if pack_path.exists() else []
        if zip_files:
            import zipfile
            try:
                with zipfile.ZipFile(zip_files[0], 'r') as zf:
                    # Try different paths inside the zip
                    for zip_internal_path in ["import_graph.json", "pack/import_graph.json"]:
                        if zip_internal_path in zf.namelist():
                            graph_data = json.loads(zf.read(zip_internal_path))
                            
                            if "edges" in graph_data:
                                 files = set()
                                 for edge in graph_data["edges"]:
                                     files.add(edge["source"])
                                     files.add(edge["target"])
                                 file_list = sorted(list(files))
                            else:
                                 file_list = sorted(list(graph_data.keys()))
                                 
                            return {
                                "files": file_list,
                                "roots": suggest_entry_points(graph_data)
                            }
            except Exception as e:
                logger.warning(f"Error reading zip file: {e}")
        
        return {"files": [], "roots": []}
        
    try:
        graph = json.loads(graph_path.read_text(encoding='utf-8'))
        
        if "edges" in graph:
             files = set()
             for edge in graph["edges"]:
                 files.add(edge["source"])
                 files.add(edge["target"])
             file_list = sorted(list(files))
        else:
             file_list = sorted(list(graph.keys()))

        return {
            "files": file_list,
            "roots": suggest_entry_points(graph)
        }
    except Exception as e:
        logger.warning(f"Error reading job files: {e}")
        return {"files": [], "roots": []}


# ============================================================================
# Vibe Coding / Prompt Generation Endpoints
# ============================================================================

@app.post("/vibe-prompt", response_model=VibePromptResponse, tags=["Vibe Coding"])
async def generate_vibe_prompt_endpoint(
    request: VibePromptRequest,
    req: Request,
    db: Session = Depends(get_db),
    _origin: bool = Depends(verify_origin)
):
    """Generate an LLM-optimized prompt from a completed job's pack."""
    client_id = get_client_identifier(req)
    rate_info = check_and_increment_rate_limit(db, client_id, req)
    
    from .prompt_engine import generate_vibe_prompt as engine_generate_prompt

    job = db.query(Job).filter(Job.id == request.job_id).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {request.job_id}"
        )

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not completed. Current status: {job.status}"
        )

    if not job.result_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No result pack found for job {request.job_id}"
        )

    if request.mode == "focus" and not request.query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query is required for 'focus' mode"
        )

    if request.mode == "bundle" and not request.entry_point and not request.query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Entry point or query is required for 'bundle' mode"
        )

    try:
        local_job_dir = worker_packs_dir / request.job_id
        local_pack_path = None
        
        if local_job_dir.exists():
            if (local_job_dir / "name_registry.json").exists():
                local_pack_path = local_job_dir
            else:
                toons = list(local_job_dir.glob("*.toon"))
                zips = list(local_job_dir.glob("*.zip"))
                jsons = list(local_job_dir.glob("pack*.json"))
                
                if toons:
                    local_pack_path = toons[0]
                elif zips:
                    local_pack_path = zips[0]
                elif jsons:
                    local_pack_path = jsons[0]

        if local_pack_path and local_pack_path.exists():
            result = engine_generate_prompt(
                pack_path=str(local_pack_path),
                mode=request.mode,
                query=request.query,
                entry_point=request.entry_point,
                max_files=request.max_files,
                max_depth=request.max_depth,
                repo_url=job.repo_url,
                token_limit=request.token_limit
            )
            
            return VibePromptResponse(
                prompt=result["prompt"],
                metadata=result["metadata"],
                rate_limit=rate_info
            )
            
        with tempfile.TemporaryDirectory(prefix="vibe_prompt_") as temp_dir:
            result_url = job.result_url
            
            if not result_url:
                 raise ValueError("No result URL found for job")

            parts = result_url.split('/')
            
            if "/packs/" in result_url:
                 raise ValueError("Local pack file not found on disk")

            bucket_name = parts[-3] if len(parts) >= 3 else S3_BUCKET
            s3_key = "/".join(parts[-2:])
            
            file_ext = Path(s3_key).suffix
            temp_pack_path = Path(temp_dir) / f"pack{file_ext}"

            if s3_client is None:
                raise RuntimeError("S3 client not initialized and local file not found")

            s3_client.download_file(bucket_name, s3_key, str(temp_pack_path))

            result = engine_generate_prompt(
                pack_path=str(temp_pack_path),
                mode=request.mode,
                query=request.query,
                entry_point=request.entry_point,
                max_files=request.max_files,
                max_depth=request.max_depth,
                token_limit=request.token_limit
            )

            return VibePromptResponse(
                prompt=result["prompt"],
                metadata=result["metadata"],
                rate_limit=rate_info
            )

    except Exception as e:
        logger.error(f"Error generating vibe prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate prompt: {str(e)}"
        )
