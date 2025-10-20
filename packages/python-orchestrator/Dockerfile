# RepoSynth Python Services - Dockerfile
# Multi-stage build for API and Worker services

# Stage 1: Build Rust parser daemon
FROM rust:1.82-slim as rust-builder

WORKDIR /build

# Install git (needed for tree-sitter dependencies)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy Rust project
COPY packages/rust-parser-daemon /build

# Build the Rust parser daemon in release mode
RUN cargo build --release

# Stage 2: Python runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY packages/python-orchestrator/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure
COPY . /app

# Copy the built Rust daemon from the builder stage
COPY --from=rust-builder /build/target/release/rust-parser-daemon /app/packages/rust-parser-daemon/target/release/rust-parser-daemon

# Ensure the Rust daemon is executable
RUN chmod +x /app/packages/rust-parser-daemon/target/release/rust-parser-daemon

# Set Python path to include the orchestrator package
ENV PYTHONPATH=/app/packages/python-orchestrator:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Create directories for temporary data
RUN mkdir -p /app/temp_repos /app/worker_packs /app/.reposynth_cache

# Expose API port (for API service)
EXPOSE 8000

# Default command (override in docker-compose)
CMD ["uvicorn", "orchestrator.api:app", "--host", "0.0.0.0", "--port", "8000"]
