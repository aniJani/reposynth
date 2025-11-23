# --- FILE: packages/python-orchestrator/orchestrator/api.py ---
"""
FastAPI application for RepoSynth.
Week 6: Token estimation API endpoint.
Week 8: Job queue and background processing.
"""

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
import sys
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
from .database import create_tables, get_db, Job
from . import worker

# S3/MinIO Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "reposynth-packs")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")

# Initialize S3 client
try:
    s3_client = boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
except Exception as e:
    print(f"Warning: Failed to initialize S3 client: {e}", file=sys.stderr)
    s3_client = None

# Application metadata
__version__ = "2.0.0"  # Week 8 update

# Create FastAPI application
app = FastAPI(
    title="RepoSynth API",
    description="Token estimation and repository analysis pipeline with background job processing",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Mount worker_packs directory to serve generated files locally
# This allows downloading packs without S3/MinIO if needed
worker_packs_dir = Path("/app/worker_packs")
worker_packs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/packs", StaticFiles(directory="/app/worker_packs"), name="packs")

# Enable CORS for future web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """Initialize database tables on application startup."""
    print("🚀 Starting RepoSynth API...")
    create_tables()
    print("✓ Database initialized")


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors with proper HTTP status."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )


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
    """
    Health check endpoint.
    Returns the status of the API and its dependencies.
    """
    # Test database connection
    database_healthy = False
    try:
        db.execute(text("SELECT 1"))
        database_healthy = True
    except Exception as e:
        print(f"Database health check failed: {e}", file=sys.stderr)

    dependencies: Dict[str, bool] = {
        "tiktoken": TIKTOKEN_AVAILABLE,
        "pygount": PYGOUNT_AVAILABLE,
        "database": database_healthy,
        "worker_queue": True,  # TODO: Add Redis health check
    }

    # Overall health: healthy if all critical dependencies are available
    is_healthy = PYGOUNT_AVAILABLE and database_healthy

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        version=__version__,
        dependencies=dependencies
    )


@app.post("/estimate-tokens", response_model=EstimateResponse, tags=["Estimation"])
async def estimate_tokens_endpoint(request: EstimateRequest):
    """
    Estimate tokens and time for repository analysis.

    This endpoint performs a fast analysis of the repository to estimate:
    - Total tokens (base code + feature overhead)
    - Estimated processing time for each pipeline stage
    - Language breakdown with LoC statistics
    - Warnings about expensive operations

    The estimation is accurate (uses tiktoken sampling) but fast (< 5 seconds
    for most repositories).

    Args:
        request: EstimateRequest with repo_path and config

    Returns:
        EstimateResponse with detailed estimates

    Raises:
        HTTPException: If repository doesn't exist or estimation fails
    """
    repo_path = Path(request.repo_path)

    # Validate repository exists (should be caught by Pydantic, but double-check)
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
        # Run estimation
        result = estimate_tokens(repo_path, request.config.model_dump())

        # Convert dataclasses to Pydantic models
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
            warnings=result.warnings
        )

    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Missing required dependency: {str(e)}"
        )
    except Exception as e:
        # Log the error (in production, use proper logging)
        print(f"Error during estimation: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Estimation failed: {str(e)}"
        )


@app.post("/estimate-tokens-from-github", response_model=EstimateResponse, tags=["Estimation"])
async def estimate_tokens_from_github_endpoint(request: GitHubEstimateRequest):
    """
    Estimate tokens and time for a GitHub repository.

    This endpoint:
    1. Clones the repository from GitHub (shallow clone)
    2. Runs token estimation
    3. Optionally cleans up the cloned repository

    This is useful for quickly estimating the cost of analyzing a public GitHub
    repository without manually cloning it first.

    Args:
        request: GitHubEstimateRequest with repo_url, config, and cleanup flag

    Returns:
        EstimateResponse with detailed estimates

    Raises:
        HTTPException: If cloning fails or estimation fails
    """
    # Get project root (api.py is in orchestrator/, project root is 3 levels up)
    project_root = Path(__file__).parent.parent.parent.parent
    temp_repos_dir = project_root / "temp_repos"
    temp_repos_dir.mkdir(parents=True, exist_ok=True)

    cloned_repo_path = None

    try:
        # Clone the repository
        print(f"Cloning repository from {request.repo_url}...", file=sys.stderr)
        cloned_repo_path = clone_repository(
            git_url=request.repo_url,
            temp_dir=temp_repos_dir,
            force_reclone=True  # Always get fresh copy
        )

        # Run estimation on cloned repo
        result = estimate_tokens(cloned_repo_path, request.config.model_dump())

        # Convert dataclasses to Pydantic models
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
            warnings=result.warnings
        )

    except ValueError as e:
        # Git clone or validation error
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
        # Log the error (in production, use proper logging)
        print(f"Error during GitHub estimation: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Estimation failed: {str(e)}"
        )
    finally:
        # Cleanup cloned repository if requested
        if request.cleanup and cloned_repo_path and cloned_repo_path.exists():
            try:
                cleanup_cloned_repo(cloned_repo_path)
            except Exception as e:
                print(f"Warning: Failed to cleanup cloned repo: {e}", file=sys.stderr)


@app.post("/estimate", response_model=ConfiguratorEstimateResponse, tags=["Estimation"])
async def estimate_from_config(request: ConfiguratorEstimateRequest):
    """
    Estimate tokens and cost based on configuration without cloning the repository.
    
    This endpoint provides fast, deterministic estimates based on:
    - Analysis mode (semantic, hybrid, full)
    - Feature toggles (AST, imports, complexity, security, embeddings)
    
    No actual cloning or analysis is performed - this is for UI feedback only.
    
    Args:
        request: ConfiguratorEstimateRequest with repo_url and config
    
    Returns:
        ConfiguratorEstimateResponse with estimated tokens, time, and cost
    """
    config = request.config
    
    # Base estimates (these are rough heuristics - you can refine them)
    base_estimates = {
        "semantic": {"tokens": 50000, "time": 30},
        "hybrid": {"tokens": 100000, "time": 60},
        "full": {"tokens": 200000, "time": 120}
    }
    
    estimate = base_estimates.get(config.mode, base_estimates["semantic"])
    estimated_tokens = estimate["tokens"]
    estimated_time = float(estimate["time"])
    
    # Feature multipliers
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
        warnings=warnings
    )


# ============================================================================
# Week 8: Job Queue Endpoints
# ============================================================================

@app.post("/jobs", status_code=status.HTTP_202_ACCEPTED, tags=["Jobs"])
async def create_job(
    repo_url: str,
    config: Optional[JobConfiguration] = None,
    db: Session = Depends(get_db)
):
    """
    Submit a repository analysis job for background processing.
    
    This endpoint:
    1. Creates a new job record in the database with configuration
    2. Enqueues the job for processing by a worker
    3. Returns immediately with the job ID
    
    The actual analysis runs asynchronously. Use GET /jobs/{job_id} to check status.
    
    Args:
        repo_url: Git repository URL (e.g., https://github.com/user/repo)
        config: Optional job configuration (mode, features). If not provided, defaults are used.
        db: Database session (injected)
    
    Returns:
        JSON with job_id and status
    
    Example:
        POST /jobs?repo_url=https://github.com/jquense/yup
        Body: {"mode": "hybrid", "enable_ast": true, ...}
        Response: {"job_id": "abc-123", "status": "pending"}
    """
    # Use config if provided, otherwise use defaults
    if config is None:
        config = JobConfiguration()
    
    # Extract mode for backward compatibility
    mode = config.mode
    
    # Validate mode
    if mode not in ["semantic", "hybrid", "full"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode '{mode}'. Must be 'semantic', 'hybrid', or 'full'."
        )
    
    # Validate URL format
    if not repo_url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid repository URL. Must start with http:// or https://"
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Create job record with configuration
    new_job = Job(
        id=job_id,
        repo_url=repo_url,
        mode=mode,
        status="pending",
        config=config.model_dump()  # Store full config as JSON
    )
    
    try:
        db.add(new_job)
        db.commit()
        db.refresh(new_job)
        
        # Enqueue job for background processing with full config
        # Note: worker.process_repository signature needs to accept config dict
        worker.process_repository.send(job_id, repo_url, config.model_dump())
        
        print(f"✓ Created job {job_id} for {repo_url} (mode: {mode})")
        
        return {
            "job_id": job_id,
            "status": "pending",
            "repo_url": repo_url,
            "mode": mode,
            "config": config.model_dump(),
            "message": "Job submitted successfully. Use GET /jobs/{job_id} to check status."
        }
        
    except Exception as e:
        db.rollback()
        print(f"✗ Failed to create job: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}"
        )


@app.get("/jobs/{job_id}", tags=["Jobs"])
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """
    Get the status of a submitted job.
    
    Returns detailed information about the job including:
    - Current status (pending, processing, completed, failed)
    - Timestamps (created, started, finished)
    - Result URL (when completed)
    - Error message (if failed)
    - Processing statistics
    
    Args:
        job_id: Unique job identifier returned from POST /jobs
        db: Database session (injected)
    
    Returns:
        JSON with complete job details
    
    Example:
        GET /jobs/abc-123
        Response: {
            "id": "abc-123",
            "status": "completed",
            "repo_url": "https://github.com/jquense/yup",
            "result_url": "http://minio:9000/reposynth-packs/abc-123.zip",
            ...
        }
    """
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
    """
    List all jobs with optional filtering.
    
    Args:
        status_filter: Filter by status (pending, processing, completed, failed)
        limit: Maximum number of jobs to return (default: 50, max: 200)
        offset: Number of jobs to skip for pagination (default: 0)
        db: Database session (injected)
    
    Returns:
        JSON with list of jobs and pagination info
    
    Example:
        GET /jobs?status_filter=completed&limit=10
    """
    # Validate and limit pagination
    limit = min(limit, 200)
    
    # Build query
    query = db.query(Job)
    
    if status_filter:
        if status_filter not in ["pending", "processing", "completed", "failed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status filter: {status_filter}"
            )
        query = query.filter(Job.status == status_filter)
    
    # Get total count
    total = query.count()
    
    # Apply pagination and ordering
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
    """
    Get list of files and suggested entry points for a job.
    Useful for populating UI dropdowns in Bundle Mode.
    """
    # Locate pack (adjust path if using S3 vs local)
    # Try local first
    pack_path = Path(f"/app/worker_packs/{job_id}")
    
    # Check for unzipped pack folder first
    if (pack_path / "pack").exists():
        work_dir = pack_path / "pack"
    # Check for zip file
    elif (pack_path / f"reposynth_{job_id}.zip").exists():
        # We can't easily list files inside zip without extracting or using zipfile
        # For now, let's assume if it's zipped, we might need to peek inside
        # But simpler is to look for import_graph.json if it was extracted
        # If not extracted, we might fail or need to implement zip reading here
        # Let's try to find import_graph.json in the pack directory if it exists
        work_dir = pack_path
    else:
        # Fallback to just checking if we can find import_graph.json anywhere
        work_dir = pack_path

    graph_path = work_dir / "import_graph.json"
    
    # If not found locally, we might need to check S3 or fail
    # For this implementation, we assume local availability or extracted pack
    if not graph_path.exists():
        # Try looking inside the zip if it exists
        zip_files = list(pack_path.glob("*.zip"))
        if zip_files:
            import zipfile
            try:
                with zipfile.ZipFile(zip_files[0], 'r') as zf:
                    if "import_graph.json" in zf.namelist():
                        graph_data = json.loads(zf.read("import_graph.json"))
                        
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
            except Exception:
                pass
        
        return {"files": [], "roots": []}
        
    try:
        graph = json.loads(graph_path.read_text(encoding='utf-8'))
        
        # Handle edges format for file list
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
        print(f"Error reading job files: {e}", file=sys.stderr)
        return {"files": [], "roots": []}


# ============================================================================
# Vibe Coding / Prompt Generation Endpoints
# ============================================================================

@app.post("/vibe-prompt", response_model=VibePromptResponse, tags=["Vibe Coding"])
async def generate_vibe_prompt_endpoint(request: VibePromptRequest, db: Session = Depends(get_db)):
    """
    Generate an LLM-optimized prompt from a completed job's pack.

    This endpoint supports three compression modes:
    - **Blueprint Mode**: Structure only (5-10K tokens) - No source code, just architecture
    - **Focus Mode**: Structure + relevant files (20-50K tokens) - Query-based file selection
    - **Bundle Mode**: Structure + dependency slice (50-200K+ tokens) - Full dependency tree

    The job must be completed (status='completed') and have a result pack available.
    """
    # Import engine function locally to avoid name conflict if any
    from .prompt_engine import generate_vibe_prompt as engine_generate_prompt

    # Get job from database
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

    # Validate mode-specific requirements
    if request.mode == "focus" and not request.query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query is required for 'focus' mode"
        )

    if request.mode == "bundle" and not request.entry_point:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Entry point is required for 'bundle' mode"
        )

    try:
        # Check if the pack exists locally (shared volume)
        local_job_dir = Path("/app/worker_packs") / request.job_id
        local_pack_path = None
        
        if local_job_dir.exists():
            # Find the pack file in the job directory
            # Priority: TOON > ZIP > JSON
            toons = list(local_job_dir.glob("*.toon"))
            zips = list(local_job_dir.glob("*.zip"))
            jsons = list(local_job_dir.glob("*.json"))
            
            if toons:
                local_pack_path = toons[0]
                print(f"Found local pack (toon): {local_pack_path}")
            elif zips:
                local_pack_path = zips[0]
                print(f"Found local pack (zip): {local_pack_path}")
            elif jsons:
                local_pack_path = jsons[0]
                print(f"Found local pack (json): {local_pack_path}")

        if local_pack_path and local_pack_path.exists():
            # Use local file directly
            result = engine_generate_prompt(
                pack_path=str(local_pack_path),
                mode=request.mode,
                query=request.query,
                entry_point=request.entry_point,
                max_files=request.max_files,
                max_depth=request.max_depth,
                repo_url=job.repo_url  # Pass repo_url for fallback source reading
            )
            
            return VibePromptResponse(
                prompt=result["prompt"],
                metadata=result["metadata"]
            )
            
        # Fallback to S3 download if not found locally
        with tempfile.TemporaryDirectory(prefix="vibe_prompt_") as temp_dir:
            # Parse S3 URL to get bucket and key
            result_url = job.result_url
            
            if not result_url:
                 raise ValueError("No result URL found for job")

            # Extract bucket and key from URL
            parts = result_url.split('/')
            
            # Handle local file URLs (http://localhost:8000/packs/...)
            if "/packs/" in result_url:
                 raise ValueError("Local pack file not found on disk")

            bucket_name = parts[-3] if len(parts) >= 3 else S3_BUCKET
            s3_key = "/".join(parts[-2:])  # job_id/filename.zip
            
            # Determine file extension from key
            file_ext = Path(s3_key).suffix
            temp_pack_path = Path(temp_dir) / f"pack{file_ext}"

            print(f"Downloading pack from S3: bucket={bucket_name}, key={s3_key}")

            if s3_client is None:
                raise RuntimeError("S3 client not initialized and local file not found")

            # Download from S3
            s3_client.download_file(bucket_name, s3_key, str(temp_pack_path))

            # Generate prompt using prompt engine
            result = engine_generate_prompt(
                pack_path=str(temp_pack_path),
                mode=request.mode,
                query=request.query,
                entry_point=request.entry_point,
                max_files=request.max_files,
                max_depth=request.max_depth
            )

            return VibePromptResponse(
                prompt=result["prompt"],
                metadata=result["metadata"]
            )

    except Exception as e:
        print(f"Error generating vibe prompt: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate prompt: {str(e)}"
        )
