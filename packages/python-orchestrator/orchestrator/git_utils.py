# --- FILE: packages/python-orchestrator/orchestrator/git_utils.py ---
"""
Git utilities for cloning and managing repositories.
Week 6 Extension: Support GitHub URLs in API.
"""

import subprocess
import shutil
import stat
import os
from pathlib import Path
from urllib.parse import urlparse
import sys


def extract_repo_name_from_url(git_url: str) -> str:
    """
    Extract repository name from a Git URL.

    Examples:
        https://github.com/jquense/yup -> yup
        https://github.com/expressjs/express.git -> express
        git@github.com:facebook/react.git -> react
    """
    # Parse URL
    if git_url.startswith('git@'):
        # SSH format: git@github.com:user/repo.git
        repo_part = git_url.split(':')[-1]
    else:
        # HTTPS format: https://github.com/user/repo.git
        parsed = urlparse(git_url)
        repo_part = parsed.path

    # Extract repo name
    repo_name = repo_part.rstrip('/').split('/')[-1]

    # Remove .git extension if present
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]

    return repo_name


def clone_repository(git_url: str, temp_dir: Path, force_reclone: bool = True) -> Path:
    """
    Clone a Git repository to a temporary directory.

    Args:
        git_url: Git repository URL (HTTPS or SSH)
        temp_dir: Base directory for temporary repos (e.g., temp_repos/)
        force_reclone: If True, delete existing clone and reclone

    Returns:
        Path to the cloned repository

    Raises:
        subprocess.CalledProcessError: If git clone fails
        ValueError: If git_url is invalid
    """
    if not git_url:
        raise ValueError("Git URL cannot be empty")

    # Extract repository name
    repo_name = extract_repo_name_from_url(git_url)
    if not repo_name:
        raise ValueError(f"Could not extract repository name from URL: {git_url}")

    # Ensure temp directory exists
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Target path for cloned repo
    clone_path = temp_dir / repo_name

    # Remove existing clone if requested
    if force_reclone and clone_path.exists():
        print(f"Removing existing clone at {clone_path}...", file=sys.stderr)
        # Use the same Windows-safe deletion
        shutil.rmtree(clone_path, onerror=_handle_remove_readonly)

    # Clone repository
    if not clone_path.exists():
        print(f"Cloning {git_url} to {clone_path}...", file=sys.stderr)

        try:
            # Run git clone (shallow clone for speed)
            result = subprocess.run(
                ['git', 'clone', '--depth=1', git_url, str(clone_path)],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Successfully cloned repository to {clone_path}", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            error_msg = f"Git clone failed: {e.stderr}"
            print(error_msg, file=sys.stderr)
            raise ValueError(error_msg) from e
        except FileNotFoundError:
            raise ValueError("Git is not installed or not in PATH")
    else:
        print(f"Repository already exists at {clone_path}, using existing clone", file=sys.stderr)

    return clone_path


def _handle_remove_readonly(func, path, exc):
    """
    Error handler for Windows readonly file deletion.
    Used with shutil.rmtree to handle permission errors on Windows.
    """
    # Check if it's a permission error
    if not os.access(path, os.W_OK):
        # Try to make the file writable
        os.chmod(path, stat.S_IWUSR | stat.S_IREAD)
        # Retry the operation
        func(path)
    else:
        # Re-raise the exception if it's not a permission issue
        raise


def cleanup_cloned_repo(repo_path: Path):
    """
    Remove a cloned repository.
    Handles Windows readonly file issues in .git directory.

    Args:
        repo_path: Path to the cloned repository to remove
    """
    if repo_path.exists() and repo_path.is_dir():
        print(f"Cleaning up cloned repository at {repo_path}...", file=sys.stderr)

        # Use onerror handler for Windows compatibility
        # This handles readonly files in .git directory
        shutil.rmtree(repo_path, onerror=_handle_remove_readonly)

        print("Cleanup complete.", file=sys.stderr)
