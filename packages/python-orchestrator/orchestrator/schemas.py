# --- FILE: packages/python-orchestrator/orchestrator/schemas.py ---
"""
Pydantic models for the RepoSynth API.
Week 6: Token estimation and configuration schemas.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, Optional
from pathlib import Path


class ConfigModel(BaseModel):
    """Configuration for pipeline stages."""
    run_parsing: bool = True
    build_graphs: bool = True
    run_analysis: bool = True
    run_embeddings: bool = True
    build_variable_registry: bool = False
    store_spans: bool = False
    no_cache: bool = False

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_parsing": True,
                "build_graphs": True,
                "run_analysis": True,
                "run_embeddings": True,
                "build_variable_registry": True,
                "store_spans": True,
                "no_cache": False
            }
        }
    )


class EstimateRequest(BaseModel):
    """Request model for token estimation."""
    repo_path: str = Field(
        ...,
        description="Absolute path to the repository to analyze"
    )
    config: ConfigModel = Field(
        default_factory=ConfigModel,
        description="Pipeline configuration"
    )

    @field_validator('repo_path')
    @classmethod
    def validate_repo_path(cls, v: str) -> str:
        """Validate that the repository path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Repository path does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Repository path is not a directory: {v}")
        return str(path.resolve())

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "repo_path": "/path/to/repository",
                "config": {
                    "run_embeddings": True,
                    "build_variable_registry": True
                }
            }
        }
    )


class LanguageStats(BaseModel):
    """Statistics for a single programming language."""
    files: int
    lines: int
    code: int
    comments: int
    blanks: int
    estimated_tokens: int


class FeatureEstimate(BaseModel):
    """Estimate for a single pipeline feature."""
    enabled: bool
    estimated_tokens: int
    estimated_time_seconds: float
    description: str


class EstimateResponse(BaseModel):
    """Response model for token estimation."""
    # Overall estimates
    total_tokens: int = Field(..., description="Total estimated tokens")
    total_time_seconds: float = Field(..., description="Total estimated time in seconds")

    # Base code statistics
    base_tokens: int = Field(..., description="Tokens from source code only")
    num_files: int = Field(..., description="Total number of files")
    total_lines: int = Field(..., description="Total lines of code")

    # Language breakdown
    language_breakdown: Dict[str, LanguageStats] = Field(
        ...,
        description="Statistics per programming language"
    )

    # Feature breakdown
    feature_breakdown: Dict[str, FeatureEstimate] = Field(
        ...,
        description="Token and time estimates per feature"
    )

    # Human-readable summary
    summary: str = Field(..., description="Human-readable summary of estimates")

    # Warnings (e.g., "Large repo - embeddings may take 10+ minutes")
    warnings: list[str] = Field(default_factory=list, description="Estimation warnings")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_tokens": 50000,
                "total_time_seconds": 45.5,
                "base_tokens": 35000,
                "num_files": 120,
                "total_lines": 15000,
                "language_breakdown": {
                    "TypeScript": {
                        "files": 80,
                        "lines": 10000,
                        "code": 8000,
                        "comments": 1500,
                        "blanks": 500,
                        "estimated_tokens": 25000
                    }
                },
                "feature_breakdown": {
                    "run_embeddings": {
                        "enabled": True,
                        "estimated_tokens": 15000,
                        "estimated_time_seconds": 30.0,
                        "description": "Generate embeddings for 150 public APIs"
                    }
                },
                "summary": "Repository: 120 files, 15K LoC. Estimated: 50K tokens, ~45 seconds.",
                "warnings": ["Large number of public APIs - embeddings will be expensive"]
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    dependencies: Dict[str, bool] = Field(
        default_factory=dict,
        description="Status of required dependencies"
    )


class GitHubEstimateRequest(BaseModel):
    """Request model for estimating tokens from a GitHub repository."""
    repo_url: str = Field(
        ...,
        description="GitHub repository URL (HTTPS or SSH)",
        examples=["https://github.com/jquense/yup", "git@github.com:facebook/react.git"]
    )
    config: ConfigModel = Field(
        default_factory=ConfigModel,
        description="Pipeline configuration"
    )
    cleanup: bool = Field(
        default=True,
        description="Whether to delete the cloned repository after estimation"
    )

    @field_validator('repo_url')
    @classmethod
    def validate_repo_url(cls, v: str) -> str:
        """Validate that the repository URL looks valid."""
        v = v.strip()
        if not v:
            raise ValueError("Repository URL cannot be empty")

        # Check if it looks like a Git URL
        valid_prefixes = ['https://', 'http://', 'git@', 'ssh://']
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(
                f"Invalid Git URL. Must start with one of: {', '.join(valid_prefixes)}"
            )

        # Check for common Git hosts
        valid_hosts = ['github.com', 'gitlab.com', 'bitbucket.org']
        if not any(host in v for host in valid_hosts):
            # Warning, but don't fail (could be private Git server)
            pass

        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "repo_url": "https://github.com/jquense/yup",
                "config": {
                    "run_embeddings": True,
                    "build_variable_registry": True
                },
                "cleanup": True
            }
        }
    )
