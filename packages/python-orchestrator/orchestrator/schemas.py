# --- FILE: packages/python-orchestrator/orchestrator/schemas.py ---
"""
Pydantic models for the RepoSynth API.
Week 6: Token estimation and configuration schemas.
Week 10: Rate limiting response fields.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, Optional, Any
from pathlib import Path


class RateLimitInfo(BaseModel):
    """Rate limit information included in API responses."""
    limit: int = Field(..., description="Maximum API calls per day")
    remaining: int = Field(..., description="Remaining API calls for today")
    reset_at: str = Field(..., description="ISO timestamp when rate limit resets")


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
    
    # Rate limit info
    rate_limit: Optional[RateLimitInfo] = Field(None, description="Rate limit status")

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


# ============================================================================
# Week 9: Frontend Configuration Schemas
# ============================================================================

class JobConfiguration(BaseModel):
    """Frontend job configuration for Week 9 UI."""
    mode: str = Field(
        default="semantic",
        description="Analysis mode: semantic, hybrid, or full"
    )
    enable_ast: bool = Field(
        default=True,
        description="Enable AST parsing"
    )
    enable_imports: bool = Field(
        default=True,
        description="Enable import graph analysis"
    )
    enable_complexity: bool = Field(
        default=True,
        description="Enable code complexity metrics"
    )
    enable_security: bool = Field(
        default=False,
        description="Enable security scanning"
    )
    enable_embeddings: bool = Field(
        default=True,
        description="Enable semantic embeddings"
    )
    output_format: str = Field(
        default="zip",
        description="Output format: zip, markdown, or json"
    )

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate analysis mode."""
        valid_modes = ['semantic', 'hybrid', 'full']
        if v not in valid_modes:
            raise ValueError(f"Invalid mode '{v}'. Must be one of: {', '.join(valid_modes)}")
        return v

    @field_validator('output_format')
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate output format."""
        valid_formats = ['zip', 'markdown', 'json', 'toon']
        if v not in valid_formats:
            raise ValueError(f"Invalid output_format '{v}'. Must be one of: {', '.join(valid_formats)}")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mode": "hybrid",
                "enable_ast": True,
                "enable_imports": True,
                "enable_complexity": True,
                "enable_security": False,
                "enable_embeddings": True,
                "output_format": "zip"
            }
        }
    )


class ConfiguratorEstimateRequest(BaseModel):
    """Request for configurator-based estimation (no repo clone needed)."""
    repo_url: str = Field(
        ...,
        description="GitHub repository URL"
    )
    config: JobConfiguration

    @field_validator('repo_url')
    @classmethod
    def validate_repo_url(cls, v: str) -> str:
        """Validate repository URL."""
        v = v.strip()
        if not v:
            raise ValueError("Repository URL cannot be empty")
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Repository URL must start with http:// or https://")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "repo_url": "https://github.com/jquense/yup",
                "config": {
                    "mode": "hybrid",
                    "enable_ast": True,
                    "enable_imports": True,
                    "enable_complexity": True,
                    "enable_security": False,
                    "enable_embeddings": True
                }
            }
        }
    )


class ConfiguratorEstimateResponse(BaseModel):
    """Response with estimated costs and time."""
    estimated_tokens: int = Field(..., description="Estimated total tokens")
    estimated_time_seconds: float = Field(..., description="Estimated processing time")
    estimated_cost_usd: float = Field(..., description="Estimated cost at $0.0001 per 1K tokens")
    mode: str = Field(..., description="Analysis mode used")
    features_enabled: Dict[str, bool] = Field(..., description="Map of feature names to enabled status")
    warnings: list[str] = Field(default_factory=list, description="Configuration warnings")
    rate_limit: Optional[RateLimitInfo] = Field(None, description="Rate limit status")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "estimated_tokens": 125000,
                "estimated_time_seconds": 75.0,
                "estimated_cost_usd": 0.0125,
                "mode": "hybrid",
                "features_enabled": {
                    "enable_ast": True,
                    "enable_imports": True,
                    "enable_complexity": True,
                    "enable_security": False,
                    "enable_embeddings": True
                },
                "warnings": ["Hybrid mode may take 60-120 seconds for large repositories"]
            }
        }
    )


# ============================================================================
# Vibe Coding / Prompt Generation Schemas
# ============================================================================

class VibePromptRequest(BaseModel):
    """Request for generating a Vibe Coding prompt."""
    job_id: str = Field(
        ...,
        description="Job ID of completed analysis (pack must exist)"
    )
    mode: str = Field(
        ...,
        description="Prompt mode: 'blueprint', 'focus', or 'bundle'"
    )
    query: Optional[str] = Field(
        None,
        description="Query string (required for 'focus' mode)"
    )
    entry_point: Optional[str] = Field(
        None,
        description="Entry point file path (required for 'bundle' mode)"
    )
    max_files: int = Field(
        default=5,
        description="Maximum files to include in focus mode"
    )
    max_depth: int = Field(
        default=3,
        description="Maximum dependency depth in bundle mode"
    )
    token_limit: Optional[int] = Field(
        default=None,
        description="Optional token budget for context optimization (bundle mode only). If set, uses Graph-Knapsack algorithm to select files."
    )

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate prompt mode."""
        valid_modes = ['blueprint', 'focus', 'bundle']
        if v not in valid_modes:
            raise ValueError(f"Invalid mode '{v}'. Must be one of: {', '.join(valid_modes)}")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "abc-123",
                "mode": "focus",
                "query": "How does authentication work?",
                "max_files": 5
            }
        }
    )


class VibePromptResponse(BaseModel):
    """Response with generated Vibe Coding prompt."""
    prompt: str = Field(..., description="Generated prompt optimized for LLMs")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the prompt")
    rate_limit: Optional[RateLimitInfo] = Field(None, description="Rate limit status")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "# CODEBASE CONTEXT...\n\n## Structural Blueprint...",
                "metadata": {
                    "mode": "focus",
                    "token_estimate": 15000,
                    "files_included": 5,
                    "description": "Focused context for: authentication"
                }
            }
        }
    )
