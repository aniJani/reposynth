#!/usr/bin/env python3
"""
RepoSynth API Server Entry Point.
Week 6: Standalone server for token estimation and pipeline control.

Usage:
    python api_server.py                    # Start server on default port 8000
    python api_server.py --port 8080        # Start server on custom port
    python api_server.py --host 0.0.0.0     # Bind to all interfaces (for Docker/remote access)
"""

import argparse
import sys
from pathlib import Path

# Ensure the orchestrator package is in the Python path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "python-orchestrator"))

try:
    import uvicorn
except ImportError:
    print("Error: uvicorn is required to run the API server.", file=sys.stderr)
    print("Install with: pip install uvicorn[standard]", file=sys.stderr)
    sys.exit(1)


def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(
        description="RepoSynth API Server - Token estimation and pipeline control"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (watches for file changes)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Logging level (default: info)"
    )

    args = parser.parse_args()

    # Print startup information
    print("=" * 60)
    print("🚀 Starting RepoSynth API Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Reload: {args.reload}")
    print(f"Log Level: {args.log_level}")
    print()
    print(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
    print(f"🔍 Health Check: http://{args.host}:{args.port}/health")
    print("=" * 60)
    print()

    # Run the server
    uvicorn.run(
        "orchestrator.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,  # reload doesn't work with multiple workers
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
