# --- FILE: packages/python-orchestrator/orchestrator/api.py ---
"""
FastAPI application for RepoSynth.
Week 6: Token estimation API endpoint.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import sys
from typing import Dict

from .schemas import (
    EstimateRequest,
    EstimateResponse,
    HealthResponse,
    LanguageStats as LanguageStatsSchema,
    FeatureEstimate as FeatureEstimateSchema,
    GitHubEstimateRequest
)
from .estimator import estimate_tokens, TIKTOKEN_AVAILABLE, PYGOUNT_AVAILABLE
from .git_utils import clone_repository, cleanup_cloned_repo

# Application metadata
__version__ = "1.0.0"

# Create FastAPI application
app = FastAPI(
    title="RepoSynth API",
    description="Token estimation and repository analysis pipeline",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for future web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
async def health_check():
    """
    Health check endpoint.
    Returns the status of the API and its dependencies.
    """
    dependencies: Dict[str, bool] = {
        "tiktoken": TIKTOKEN_AVAILABLE,
        "pygount": PYGOUNT_AVAILABLE,
    }

    # Overall health: healthy if all critical dependencies are available
    is_healthy = PYGOUNT_AVAILABLE  # tiktoken is optional (fallback available)

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


# Additional endpoints can be added here in the future:
# - POST /run-pipeline - Execute full pipeline
# - POST /run-pipeline-from-github - Execute pipeline on GitHub repo
# - GET /pipeline-status/{job_id} - Check pipeline status
# - GET /results/{job_id} - Retrieve pipeline results
