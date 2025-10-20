"""
Background worker for RepoSynth job processing.

This module handles asynchronous repository analysis jobs using Dramatiq.
Workers receive jobs from Redis queue, process them, and upload results to S3.
"""

import os
import sys
import shutil
import subprocess
import datetime
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import dramatiq
import boto3
from dramatiq.brokers.redis import RedisBroker
from sqlalchemy.orm import Session

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.database import SessionLocal, Job
from orchestrator.pipeline_runner import Pipeline

# Configure Redis broker
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_broker = RedisBroker(url=REDIS_URL)
dramatiq.set_broker(redis_broker)

# S3/MinIO Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "reposynth-packs")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    endpoint_url=AWS_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def update_job_status(job_id: str, status: str, **kwargs):
    """
    Update job status in database.
    
    Args:
        job_id: Job identifier
        status: New status (pending, processing, completed, failed)
        **kwargs: Additional fields to update (error_message, result_url, etc.)
    """
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = status
            
            # Update timestamps
            if status == "processing" and not job.started_at:
                job.started_at = datetime.datetime.utcnow()
            elif status in ["completed", "failed"]:
                job.finished_at = datetime.datetime.utcnow()
                
                # Calculate processing time
                if job.started_at:
                    processing_time = (job.finished_at - job.started_at).total_seconds()
                    job.processing_time_seconds = int(processing_time)
            
            # Update additional fields
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            db.commit()
            print(f"✓ Updated job {job_id}: status={status}")
        else:
            print(f"✗ Job not found: {job_id}", file=sys.stderr)
    except Exception as e:
        db.rollback()
        print(f"✗ Failed to update job status: {e}", file=sys.stderr)
    finally:
        db.close()


def clone_repository(repo_url: str, target_dir: Path) -> Path:
    """
    Clone a git repository.
    
    Args:
        repo_url: Git repository URL
        target_dir: Directory to clone into
    
    Returns:
        Path to cloned repository
    
    Raises:
        RuntimeError: If git clone fails
    """
    # Extract repo name from URL
    parsed_url = urlparse(repo_url)
    repo_name = Path(parsed_url.path).name
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    
    clone_path = target_dir / repo_name
    
    # Remove existing clone if present
    if clone_path.exists():
        print(f"Removing existing clone at {clone_path}")
        shutil.rmtree(clone_path)
    
    # Clone the repository
    print(f"Cloning {repo_url}...")
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", "--no-tags", repo_url, str(clone_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        print(f"✓ Repository cloned to {clone_path}")
        return clone_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git clone failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Git clone timed out after 5 minutes")
    except FileNotFoundError:
        raise RuntimeError("git command not found. Please install git.")


def upload_to_s3(file_path: Path, s3_key: str) -> str:
    """
    Upload a file to S3/MinIO.
    
    Args:
        file_path: Local file path
        s3_key: S3 object key (filename in bucket)
    
    Returns:
        Public URL to the uploaded file
    
    Raises:
        RuntimeError: If upload fails
    """
    try:
        print(f"Uploading {file_path.name} to S3 bucket {S3_BUCKET}...")
        
        # Upload file
        s3_client.upload_file(
            str(file_path),
            S3_BUCKET,
            s3_key,
            ExtraArgs={"ContentType": "application/zip"}
        )
        
        # Generate public URL
        # For MinIO, the URL format is: http://endpoint/bucket/key
        if AWS_ENDPOINT_URL:
            # Replace internal Docker hostname with external hostname
            external_url = AWS_ENDPOINT_URL.replace("minio:9000", "localhost:9000")
            result_url = f"{external_url}/{S3_BUCKET}/{s3_key}"
        else:
            # AWS S3 URL format
            result_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
        
        print(f"✓ Uploaded to: {result_url}")
        return result_url
        
    except Exception as e:
        raise RuntimeError(f"S3 upload failed: {e}")


@dramatiq.actor(max_retries=0, time_limit=1800000)  # 30 minute timeout
def process_repository(job_id: str, repo_url: str, mode: str = "semantic"):
    """
    Background task to process a repository analysis job.
    
    This is the main worker function that:
    1. Clones the repository
    2. Runs the RepoSynth pipeline
    3. Uploads the result pack to S3
    4. Updates job status in database
    
    Args:
        job_id: Unique job identifier
        repo_url: Git repository URL
        mode: Analysis mode (semantic, hybrid, full)
    """
    print(f"\n{'='*60}")
    print(f"🔧 Starting job {job_id}")
    print(f"   Repository: {repo_url}")
    print(f"   Mode: {mode}")
    print(f"{'='*60}\n")
    
    # Update status to processing
    update_job_status(job_id, "processing")
    
    # Setup paths
    worker_root = Path("/app/worker_packs") / job_id
    worker_root.mkdir(parents=True, exist_ok=True)
    
    temp_repos = Path("/app/temp_repos")
    temp_repos.mkdir(parents=True, exist_ok=True)
    
    cloned_repo_path = None
    
    try:
        # Step 1: Clone repository
        cloned_repo_path = clone_repository(repo_url, temp_repos)
        
        # Step 2: Configure pipeline based on mode
        if mode == "semantic":
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
        elif mode == "hybrid":
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
        elif mode == "full":
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
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Step 3: Run pipeline
        output_pack_dir = worker_root / "pack"
        daemon_path = "/app/packages/rust-parser-daemon/target/release/rust-parser-daemon"
        
        print(f"\n🔄 Running pipeline...")
        pipeline = Pipeline(
            repo_path=str(cloned_repo_path),
            output_path=str(output_pack_dir),
            daemon_path=daemon_path,
        )
        
        pipeline.run(config=config)
        
        print(f"✓ Pipeline completed successfully")
        
        # Step 4: Package results
        # For hybrid/full modes, the pipeline creates a .zip file
        # For semantic mode, we need to create one
        
        repo_name = cloned_repo_path.name
        pack_filename = f"reposynth_{repo_name}_{mode}_{job_id}.zip"
        pack_path = worker_root / pack_filename
        
        if mode == "semantic":
            # Create zip archive from loose files
            print(f"📦 Creating semantic pack archive...")
            with zipfile.ZipFile(pack_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in output_pack_dir.glob("*"):
                    if file.is_file():
                        zf.write(file, arcname=f"pack/{file.name}")
                        print(f"   ✓ Added: {file.name}")
        else:
            # For hybrid/full, the zip was created by the pipeline
            existing_zip = list(worker_root.glob("reposynth_*.zip"))
            if existing_zip:
                pack_path = existing_zip[0]
                # Rename to include job_id
                new_name = f"reposynth_{repo_name}_{mode}_{job_id}.zip"
                pack_path = pack_path.rename(pack_path.parent / new_name)
            else:
                # Fallback: create from pack directory
                print(f"📦 Creating {mode} pack archive...")
                with zipfile.ZipFile(pack_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for file in output_pack_dir.rglob("*"):
                        if file.is_file():
                            rel_path = file.relative_to(output_pack_dir)
                            zf.write(file, arcname=f"pack/{rel_path}")
        
        # Get pack size
        pack_size = pack_path.stat().st_size
        print(f"✓ Pack created: {pack_path.name} ({pack_size / 1024 / 1024:.2f} MB)")
        
        # Step 5: Upload to S3
        s3_key = f"{job_id}/{pack_filename}"
        result_url = upload_to_s3(pack_path, s3_key)
        
        # Step 6: Mark job as completed
        update_job_status(
            job_id,
            "completed",
            result_url=result_url,
            pack_size_bytes=pack_size
        )
        
        print(f"\n{'='*60}")
        print(f"✅ Job {job_id} completed successfully!")
        print(f"   Result URL: {result_url}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        # Log error and update job status
        error_msg = f"Job failed: {str(e)}"
        print(f"\n{'='*60}")
        print(f"❌ Job {job_id} failed!")
        print(f"   Error: {error_msg}")
        print(f"{'='*60}\n", file=sys.stderr)
        
        update_job_status(
            job_id,
            "failed",
            error_message=error_msg
        )
        
        raise  # Re-raise to let Dramatiq handle it
        
    finally:
        # Cleanup: Remove cloned repository
        if cloned_repo_path and cloned_repo_path.exists():
            try:
                print(f"🧹 Cleaning up cloned repository...")
                shutil.rmtree(cloned_repo_path)
            except Exception as e:
                print(f"Warning: Failed to cleanup cloned repo: {e}", file=sys.stderr)
        
        # Cleanup: Remove worker directory (keep only S3 copy)
        try:
            if worker_root.exists():
                print(f"🧹 Cleaning up worker directory...")
                shutil.rmtree(worker_root)
        except Exception as e:
            print(f"Warning: Failed to cleanup worker directory: {e}", file=sys.stderr)


if __name__ == "__main__":
    # This allows testing the worker locally
    print("Dramatiq worker for RepoSynth")
    print(f"Redis URL: {REDIS_URL}")
    print(f"S3 Bucket: {S3_BUCKET}")
    print("Run with: dramatiq orchestrator.worker")
