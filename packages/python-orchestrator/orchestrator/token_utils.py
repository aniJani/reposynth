# --- FILE: packages/python-orchestrator/orchestrator/token_utils.py ---
"""
Shared token counting utilities for RepoSynth.
Used by both the Estimator (Week 6) and Pipeline (Week 10 Context Optimizer).
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import sys
import logging

logger = logging.getLogger("reposynth.token_utils")

# Try to import tiktoken
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not installed. Using fallback token estimation.")

# Cache the encoding to avoid repeated initialization
_encoding_cache: Optional[object] = None


def get_encoding():
    """Get the tiktoken encoding, caching it for reuse."""
    global _encoding_cache
    if _encoding_cache is None and TIKTOKEN_AVAILABLE:
        # Use cl100k_base encoding (GPT-4, GPT-3.5-turbo, Claude)
        _encoding_cache = tiktoken.get_encoding("cl100k_base")
    return _encoding_cache


def count_tokens_in_text(text: str) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for.
        
    Returns:
        Number of tokens in the text.
    """
    if not TIKTOKEN_AVAILABLE:
        # Fallback: rough estimate of 0.25 tokens per character
        # (typical ratio for code)
        return max(1, len(text) // 4)
    
    encoding = get_encoding()
    if encoding is None:
        return max(1, len(text) // 4)
    
    try:
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Token counting failed: {e}. Using fallback.")
        return max(1, len(text) // 4)


def count_tokens_in_file(file_path: Union[str, Path]) -> int:
    """
    Count the number of tokens in a file.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        Number of tokens in the file, or 0 if file cannot be read.
    """
    file_path = Path(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return count_tokens_in_text(content)
    except Exception as e:
        logger.warning(f"Could not read file {file_path}: {e}")
        return 0


def count_tokens_in_files(file_paths: List[Path]) -> int:
    """
    Count total tokens across multiple files.
    
    Args:
        file_paths: List of file paths to count.
        
    Returns:
        Total token count across all files.
    """
    total = 0
    for file_path in file_paths:
        total += count_tokens_in_file(file_path)
    return total


def generate_token_map(
    repo_path: Path,
    file_paths: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Generate a token count map for all files in a repository.
    
    Args:
        repo_path: Root path of the repository.
        file_paths: Optional list of specific file paths (relative to repo).
                   If None, discovers all source files.
        exclude_patterns: Patterns to exclude (e.g., ["test", "spec", "__pycache__"])
        
    Returns:
        Dictionary mapping relative file paths to their token counts.
        Example: {"src/index.ts": 150, "src/schema.ts": 4500}
    """
    token_map: Dict[str, int] = {}
    repo_path = Path(repo_path)
    
    # Default exclude patterns
    if exclude_patterns is None:
        exclude_patterns = [
            ".git", ".venv", "venv", "node_modules", "__pycache__",
            "build", "dist", ".tox", ".pytest_cache", ".mypy_cache",
            "*.min.js", "*.min.css", "*.map"
        ]
    
    # Source file extensions to include
    source_extensions = {
        ".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go", ".java",
        ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
        ".kt", ".scala", ".vue", ".svelte", ".md", ".json", ".yaml", ".yml",
        ".css", ".scss", ".html", ".htm"
    }
    
    if file_paths:
        # Use provided file list
        files_to_process = [repo_path / fp for fp in file_paths]
    else:
        # Discover all source files
        files_to_process = []
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue
                
            # Check if file should be excluded
            rel_path_str = str(file_path.relative_to(repo_path))
            should_exclude = False
            
            for pattern in exclude_patterns:
                if pattern.startswith("*."):
                    # Extension pattern
                    if file_path.suffix == pattern[1:]:
                        should_exclude = True
                        break
                elif pattern in rel_path_str.split("/") or pattern in rel_path_str.split("\\"):
                    # Directory pattern
                    should_exclude = True
                    break
            
            if should_exclude:
                continue
                
            # Check extension
            if file_path.suffix.lower() in source_extensions:
                files_to_process.append(file_path)
    
    # Count tokens for each file
    for file_path in files_to_process:
        try:
            rel_path = str(file_path.relative_to(repo_path))
            # Normalize path separators
            rel_path = rel_path.replace("\\", "/")
            token_count = count_tokens_in_file(file_path)
            token_map[rel_path] = token_count
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            continue
    
    return token_map


def estimate_tokens_from_lines(lines_of_code: int, language: str = "unknown") -> int:
    """
    Estimate token count from lines of code.
    Uses language-specific ratios based on empirical measurements.
    
    Args:
        lines_of_code: Number of lines of code.
        language: Programming language (for language-specific ratios).
        
    Returns:
        Estimated token count.
    """
    # Tokens per line ratios (empirically measured)
    ratios = {
        "python": 6.5,
        "typescript": 7.0,
        "javascript": 6.8,
        "java": 8.0,
        "rust": 7.5,
        "go": 6.0,
        "c": 5.5,
        "cpp": 6.5,
        "unknown": 6.5,  # Default
    }
    
    ratio = ratios.get(language.lower(), ratios["unknown"])
    return int(lines_of_code * ratio)


# Expose availability flag
def is_tiktoken_available() -> bool:
    """Check if tiktoken is available for accurate token counting."""
    return TIKTOKEN_AVAILABLE
