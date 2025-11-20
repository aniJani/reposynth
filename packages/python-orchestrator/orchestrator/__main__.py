import sys
from pathlib import Path
import argparse
import subprocess
import tempfile
import shutil
from urllib.parse import urlparse

# Ensure the orchestrator package is in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.pipeline_runner import Pipeline

def main():
    """The main entrypoint for running the configurable RepoSynth pipeline."""

    parser = argparse.ArgumentParser(
        prog="reposynth",
        description="RepoSynth: A powerful repository analysis and synthesis pipeline",
        epilog="""
Examples:
  # Analyze current directory with semantic mode
  python -m orchestrator

  # Analyze a specific repository with hybrid mode
  python -m orchestrator --repo /path/to/repo --mode hybrid

  # Run with security scanning enabled
  python -m orchestrator --repo /path/to/repo --with-security-scans

  # Disable caching for fresh analysis
  python -m orchestrator --repo /path/to/repo --no-cache

  # Generate blast-radius context for LLM (VibeCode mode)
  python -m orchestrator --mode blast-radius --query "login authentication user"

  # Blast-radius with custom max files
  python -m orchestrator --mode blast-radius --query "payment processing" --max-shockwave-files 30

For more information, visit: https://github.com/aniJani/reposynth
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # --- Primary Arguments ---
    parser.add_argument(
        "--repo",
        type=str,
        metavar="PATH",
        help="Path to the local repository to analyze. If not provided, analyzes the RepoSynth project itself."
    )

    # --- Mode-based Configuration ---
    parser.add_argument(
        "--mode",
        type=str,
        default="semantic",
        choices=["semantic", "hybrid", "full", "blast-radius"],
        metavar="MODE",
        help="""Pipeline mode (default: semantic).

        semantic:      Lightweight analysis with AST, graphs, metrics, and embeddings (loose files).
        hybrid:        Adds variable registry and source spans, packaged as .zip archive.
        full:          Complete analysis with all features and raw AST files in .zip archive.
        blast-radius:  Smart context generation for LLMs (requires --query). Finds relevant files
                       and their dependencies, provides full source for epicenter files and
                       skeleton interfaces for dependencies. Outputs vibecode_prompt.md.
        """
    )

    parser.add_argument(
        "--query",
        type=str,
        metavar="TEXT",
        help="Search query for blast-radius mode. Specifies what code to focus on (e.g., 'login authentication')."
    )

    parser.add_argument(
        "--max-shockwave-files",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of dependency files to include in blast-radius mode (default: 50)."
    )

    # --- Fine-grained Feature Toggles ---
    parser.add_argument(
        "--with-parsing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Parse repository files into AST (Abstract Syntax Trees). Required for all other stages."
    )

    parser.add_argument(
        "--with-graphs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build import dependency graphs and name registry (symbol definitions)."
    )

    parser.add_argument(
        "--with-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run static analysis to calculate complexity metrics using Ruff."
    )

    parser.add_argument(
        "--with-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate semantic embeddings for public APIs using SentenceTransformers."
    )

    parser.add_argument(
        "--with-security-scans",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run security scans to detect hardcoded secrets and vulnerabilities (Week 7 feature)."
    )

    parser.add_argument(
        "--with-variable-registry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Build variable registry to track variable declarations (hybrid/full modes)."
    )

    parser.add_argument(
        "--with-spans",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store source code spans for public APIs in source_spans.json and append to repoBrief.md (enabled by default for all modes)."
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable all caching. Forces fresh analysis of all stages (slower but ensures clean state)."
    )

    args = parser.parse_args()

    # --- Construct the Configuration Dictionary ---
    # Start with mode-based defaults
    if args.mode == "semantic":
        config = {
            "run_parsing": True,
            "build_graphs": True,
            "run_analysis": True,
            "run_embeddings": True,
            "run_security_scans": False,
            "build_variable_registry": False,
            "store_spans": True,  # Now enabled for semantic mode to include source code in repoBrief.md
            "pack_mode": "semantic",
        }
    elif args.mode == "hybrid":
        config = {
            "run_parsing": True,
            "build_graphs": True,
            "run_analysis": True,
            "run_embeddings": True,
            "run_security_scans": True,
            "build_variable_registry": True,
            "store_spans": True,
            "pack_mode": "hybrid",
        }
    elif args.mode == "full":
        config = {
            "run_parsing": True,
            "build_graphs": True,
            "run_analysis": True,
            "run_embeddings": True,
            "run_security_scans": True,
            "build_variable_registry": True,
            "store_spans": True,
            "pack_mode": "full",
        }
    elif args.mode == "blast-radius":
        # Blast Radius Mode: Minimal stages needed for smart context generation
        if not args.query:
            print("FATAL: --query is required for blast-radius mode", file=sys.stderr)
            print("Example: python -m orchestrator --mode blast-radius --query 'login authentication'", file=sys.stderr)
            sys.exit(1)

        config = {
            "run_parsing": True,           # Need AST for skeleton generation
            "build_graphs": True,          # Need import_graph and name_registry
            "run_analysis": True,          # Need complexity data for scoring
            "run_embeddings": False,       # Not needed for blast-radius
            "run_security_scans": False,   # Not needed for blast-radius
            "build_variable_registry": False,  # Not needed for blast-radius
            "store_spans": False,          # Not needed for blast-radius
            "pack_mode": "blast-radius",
            "query": args.query,
            "max_shockwave_files": args.max_shockwave_files,
        }
    else:  # Default to semantic for safety
        config = {
            "run_parsing": True,
            "build_graphs": True,
            "run_analysis": True,
            "run_embeddings": True,
            "run_security_scans": False,
            "build_variable_registry": False,
            "store_spans": True,  # Now enabled for semantic mode to include source code in repoBrief.md
            "pack_mode": "semantic",
        }

    # Override defaults with specific command-line toggles
    config["run_parsing"] = args.with_parsing
    config["build_graphs"] = args.with_graphs
    config["run_analysis"] = args.with_analysis
    config["run_embeddings"] = args.with_embeddings
    config["no_cache"] = args.no_cache

    if args.with_security_scans is not None:
        config["run_security_scans"] = args.with_security_scans
    if args.with_variable_registry is not None:
        config["build_variable_registry"] = args.with_variable_registry
    if args.with_spans is not None:
        config["store_spans"] = args.with_spans
    
    # --- Path Resolution Logic ---
    root_dir = Path(__file__).parent.parent.parent.parent.resolve()

    # Track if we cloned a repo so we can clean it up later
    cloned_repo = None

    if args.repo:
        # Check if the input is a URL
        if args.repo.startswith("http://") or args.repo.startswith("https://"):
            print(f"--- Detected URL: {args.repo} ---")
            print(f"--- Cloning repository... ---")

            # Extract repo name from URL
            parsed_url = urlparse(args.repo)
            repo_name = Path(parsed_url.path).name
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]

            # Create a temp directory for the clone
            temp_dir = root_dir / "temp_repos"
            temp_dir.mkdir(exist_ok=True)
            clone_path = temp_dir / repo_name

            # Remove existing clone if present
            if clone_path.exists():
                print(f"--- Removing existing clone at {clone_path} ---")
                shutil.rmtree(clone_path)

            # Clone the repository
            try:
                subprocess.run(
                    ["git", "clone", args.repo, str(clone_path)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                repo_to_parse = clone_path
                cloned_repo = clone_path
                print(f"--- Repository cloned to: {repo_to_parse} ---")
            except subprocess.CalledProcessError as e:
                print(f"FATAL: Failed to clone repository: {e.stderr}", file=sys.stderr)
                sys.exit(1)
            except FileNotFoundError:
                print("FATAL: git command not found. Please install git.", file=sys.stderr)
                sys.exit(1)
        else:
            # It's a local path
            repo_to_parse = Path(args.repo).resolve()
            if not repo_to_parse.exists():
                print(f"FATAL: Repository path does not exist: {repo_to_parse}", file=sys.stderr)
                sys.exit(1)
            print(f"--- Target repository: {repo_to_parse} ---")
    else:
        repo_to_parse = root_dir
        print(f"--- No --repo specified. Parsing the RepoSynth project itself. ---")

    daemon_path = root_dir / "packages/rust-parser-daemon/target/release/rust-parser-daemon"
    
    if sys.platform == "darwin": pass
    elif sys.platform == "win32": daemon_path = daemon_path.with_suffix('.exe')

    if not daemon_path.exists():
        print(f"FATAL: Daemon executable not found at path: {daemon_path}", file=sys.stderr)
        sys.exit(1)

    output_pack_dir = root_dir / "pack"
    
    print(f"--- Initializing Pipeline with Config: {config} ---")
    
    pipeline = Pipeline(
        repo_path=str(repo_to_parse),
        output_path=str(output_pack_dir),
        daemon_path=str(daemon_path),
    )

    try:
        pipeline.run(config=config)
    finally:
        # Clean up cloned repository if it was temporary
        # Note: We keep temp clones for caching purposes
        # Users can manually delete temp_repos/ if needed
        pass

if __name__ == '__main__':
    main()