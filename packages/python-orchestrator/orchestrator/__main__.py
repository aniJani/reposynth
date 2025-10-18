import sys
from pathlib import Path
import argparse

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
        choices=["semantic", "hybrid", "full"],
        metavar="MODE",
        help="""Pipeline mode (default: semantic).

        semantic: Lightweight analysis with AST, graphs, metrics, and embeddings (loose files).
        hybrid:   Adds variable registry and source spans, packaged as .zip archive.
        full:     Complete analysis with all features and raw AST files in .zip archive.
        """
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
        help="Store source code spans for public APIs in spans.zip (hybrid/full modes)."
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
            "store_spans": False,
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
    else:  # Default to semantic for safety
        config = {
            "run_parsing": True,
            "build_graphs": True,
            "run_analysis": True,
            "run_embeddings": True,
            "run_security_scans": False,
            "build_variable_registry": False,
            "store_spans": False,
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
    
    if args.repo:
        repo_to_parse = Path(args.repo).resolve()
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
    pipeline.run(config=config)

if __name__ == '__main__':
    main()