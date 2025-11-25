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
import json
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
try:
    s3_client = boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
except Exception as e:
    print(f"Warning: Failed to initialize S3 client: {e}", file=sys.stderr)
    s3_client = None


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


def generate_markdown_output(pack_dir: Path, repo_name: str) -> str:
    """
    Generate a comprehensive markdown document from pack contents.
    
    Args:
        pack_dir: Path to the pack directory
        repo_name: Name of the repository
    
    Returns:
        Markdown content as string
    """
    markdown_lines = []
    
    # Header
    markdown_lines.append(f"# RepoSynth Analysis: {repo_name}\n")
    markdown_lines.append(f"Generated: {datetime.datetime.utcnow().isoformat()}\n\n")
    markdown_lines.append("---\n\n")
    
    # Repository Brief
    repo_brief_path = pack_dir / "repoBrief.md"
    if repo_brief_path.exists():
        markdown_lines.append("## Repository Overview\n\n")
        markdown_lines.append(repo_brief_path.read_text(encoding='utf-8'))
        markdown_lines.append("\n\n---\n\n")
    
    # Manifest
    manifest_path = pack_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            
            markdown_lines.append("## Pack Manifest\n\n")
            
            # New format: manifest contains metadata about artifacts
            if 'artifacts' in manifest:
                artifacts = manifest['artifacts']
                markdown_lines.append(f"**Pack Version:** {manifest.get('packVersion', '1.0')}\n")
                markdown_lines.append(f"**Mode:** {manifest.get('mode', 'unknown')}\n")
                markdown_lines.append(f"**Created:** {manifest.get('createdAt', 'unknown')}\n")
                markdown_lines.append(f"**Commit:** `{manifest.get('commit_hash', 'unknown')[:8]}`\n\n")
                
                markdown_lines.append("### Artifacts\n\n")
                markdown_lines.append("| File | Size | SHA256 (first 16 chars) |\n")
                markdown_lines.append("|------|------|-------------------------|\n")
                for artifact_name, artifact_info in artifacts.items():
                    size_kb = artifact_info.get('size_bytes', 0) / 1024
                    sha_short = artifact_info.get('sha256', '')[:16]
                    markdown_lines.append(f"| `{artifact_name}` | {size_kb:.2f} KB | `{sha_short}...` |\n")
            else:
                # Old format or unknown structure
                markdown_lines.append("*Manifest format not recognized*\n")
            
            markdown_lines.append("\n\n---\n\n")
        except Exception as e:
            markdown_lines.append(f"## Pack Manifest\n\n*Error parsing manifest: {e}*\n\n---\n\n")
    
    # Import Graph
    import_graph_path = pack_dir / "import_graph.json"
    if import_graph_path.exists():
        try:
            import_graph = json.loads(import_graph_path.read_text(encoding='utf-8'))
            markdown_lines.append("## Import Dependencies\n\n")
            
            # New format: dictionary where keys are files and values are arrays of imports
            if isinstance(import_graph, dict):
                total_files = len(import_graph)
                total_imports = sum(len(imports) for imports in import_graph.values())
                
                markdown_lines.append(f"**Total Files:** {total_files}\n")
                markdown_lines.append(f"**Total Import Statements:** {total_imports}\n\n")
                
                # Find files with most imports
                files_by_imports = sorted(
                    import_graph.items(),
                    key=lambda x: len(x[1]),
                    reverse=True
                )
                
                if files_by_imports:
                    markdown_lines.append("### Files by Import Count\n\n")
                    markdown_lines.append("| File | Imports |\n")
                    markdown_lines.append("|------|--------:|\n")
                    for file_path, imports in files_by_imports[:20]:  # Top 20
                        import_count = len(imports)
                        markdown_lines.append(f"| `{file_path}` | {import_count} |\n")
                    
                    if len(files_by_imports) > 20:
                        markdown_lines.append(f"\n*... and {len(files_by_imports) - 20} more files*\n")
            else:
                # Old format with nodes/edges (keep for backwards compatibility)
                nodes = import_graph.get('nodes', [])
                edges = import_graph.get('edges', [])
                markdown_lines.append(f"**Modules:** {len(nodes)}\n")
                markdown_lines.append(f"**Dependencies:** {len(edges)}\n\n")
                
                if nodes:
                    markdown_lines.append("### Key Modules\n\n")
                    for node in (nodes[:20] if isinstance(nodes, list) else list(nodes.values())[:20]):
                        if isinstance(node, dict):
                            name = node.get('name', node.get('id', 'unknown'))
                        else:
                            name = str(node)
                        markdown_lines.append(f"- `{name}`\n")
            
            markdown_lines.append("\n\n---\n\n")
        except Exception as e:
            markdown_lines.append(f"## Import Dependencies\n\n*Error parsing import graph: {e}*\n\n---\n\n")
    
    # Security Report
    security_path = pack_dir / "security_report.json"
    if security_path.exists():
        security = json.loads(security_path.read_text(encoding='utf-8'))
        markdown_lines.append("## Security Analysis\n\n")
        findings = security.get('findings', [])
        markdown_lines.append(f"**Total Findings:** {len(findings)}\n\n")
        
        if findings:
            markdown_lines.append("### Security Issues\n\n")
            for finding in findings[:10]:  # Top 10 findings
                severity = finding.get('severity', 'unknown')
                description = finding.get('description', 'No description')
                file_path = finding.get('file', 'unknown')
                markdown_lines.append(f"- **{severity.upper()}**: {description} (in `{file_path}`)\n")
        markdown_lines.append("\n\n---\n\n")
    
    # Name Registry
    name_registry_path = pack_dir / "name_registry.json"
    if name_registry_path.exists():
        try:
            name_registry = json.loads(name_registry_path.read_text(encoding='utf-8'))
            markdown_lines.append("## Symbol Registry\n\n")
            markdown_lines.append(f"**Total Symbols:** {len(name_registry)}\n\n")
            
            # Group by kind (not type!)
            by_kind = {}
            public_symbols = []
            for symbol_key, symbol_data in name_registry.items():
                # Keys are in format "file:symbol_name"
                sym_kind = symbol_data.get('kind', 'unknown')
                by_kind[sym_kind] = by_kind.get(sym_kind, 0) + 1
                
                # Track public symbols
                if symbol_data.get('is_public', False):
                    symbol_name = symbol_key.split(':')[-1] if ':' in symbol_key else symbol_key
                    public_symbols.append({
                        'name': symbol_name,
                        'kind': sym_kind,
                        'file': symbol_data.get('file_path', 'unknown')
                    })
            
            markdown_lines.append("### Symbol Types\n\n")
            for sym_kind, count in sorted(by_kind.items(), key=lambda x: x[1], reverse=True):
                markdown_lines.append(f"- **{sym_kind}**: {count}\n")
            
            # Public API Surface
            if public_symbols:
                markdown_lines.append(f"\n### Public API Surface\n\n")
                markdown_lines.append(f"**Total Public Symbols:** {len(public_symbols)}\n\n")
                
                # Group public symbols by kind
                public_by_kind = {}
                for sym in public_symbols:
                    kind = sym['kind']
                    if kind not in public_by_kind:
                        public_by_kind[kind] = []
                    public_by_kind[kind].append(sym)
                
                for kind, symbols in sorted(public_by_kind.items()):
                    markdown_lines.append(f"\n**{kind}** ({len(symbols)}):\n")
                    for sym in sorted(symbols, key=lambda x: x['name'])[:20]:
                        markdown_lines.append(f"- `{sym['name']}` (in `{sym['file']}`)\n")
                    if len(symbols) > 20:
                        markdown_lines.append(f"  *... and {len(symbols) - 20} more*\n")
            
            markdown_lines.append("\n\n---\n\n")
        except Exception as e:
            markdown_lines.append(f"## Symbol Registry\n\n*Error parsing name registry: {e}*\n\n---\n\n")
    
    # Variable Registry (for export/import analysis)
    variable_registry_path = pack_dir / "variable_registry.json"
    if variable_registry_path.exists():
        try:
            variable_registry = json.loads(variable_registry_path.read_text(encoding='utf-8'))
            markdown_lines.append("## Module Exports\n\n")
            
            # Count exports vs internal variables
            total_exports = 0
            total_internal = 0
            files_with_exports = []
            
            for file_path, variables in variable_registry.items():
                exports_in_file = [v for v in variables if v.get('scope') == 'export']
                internal_in_file = [v for v in variables if v.get('scope') == 'internal']
                
                total_exports += len(exports_in_file)
                total_internal += len(internal_in_file)
                
                if exports_in_file:
                    files_with_exports.append({
                        'file': file_path,
                        'exports': exports_in_file
                    })
            
            markdown_lines.append(f"**Total Exported Variables:** {total_exports}\n")
            markdown_lines.append(f"**Total Internal Variables:** {total_internal}\n")
            markdown_lines.append(f"**Files with Exports:** {len(files_with_exports)}\n\n")
            
            if files_with_exports:
                # Show top files by export count
                files_with_exports.sort(key=lambda x: len(x['exports']), reverse=True)
                markdown_lines.append("### Top Files by Exports\n\n")
                markdown_lines.append("| File | Exports |\n")
                markdown_lines.append("|------|--------:|\n")
                for item in files_with_exports[:15]:
                    markdown_lines.append(f"| `{item['file']}` | {len(item['exports'])} |\n")
                
                if len(files_with_exports) > 15:
                    markdown_lines.append(f"\n*... and {len(files_with_exports) - 15} more files*\n")
            
            markdown_lines.append("\n\n---\n\n")
        except Exception as e:
            markdown_lines.append(f"## Module Exports\n\n*Error parsing variable registry: {e}*\n\n---\n\n")
    
    # Code Statistics from analysis.sqlite
    analysis_db_path = pack_dir / "analysis.sqlite"
    if analysis_db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(analysis_db_path))
            cursor = conn.cursor()
            
            markdown_lines.append("## Code Statistics\n\n")
            
            # Check which table exists (functions or complexity)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            if 'functions' in tables:
                # Full mode with functions table
                cursor.execute("SELECT COUNT(*), SUM(end_line - start_line + 1), AVG(cyclomatic_complexity), MAX(cyclomatic_complexity) FROM functions")
                func_count, total_lines, avg_complexity, max_complexity = cursor.fetchone()
                
                if func_count:
                    markdown_lines.append(f"**Total Functions:** {func_count}\n")
                    markdown_lines.append(f"**Total Lines of Code:** {total_lines or 0}\n")
                    markdown_lines.append(f"**Average Cyclomatic Complexity:** {avg_complexity:.2f if avg_complexity else 0}\n")
                    markdown_lines.append(f"**Max Cyclomatic Complexity:** {max_complexity or 0}\n\n")
                    
                    # Most complex functions
                    cursor.execute("""
                        SELECT name, file_path, cyclomatic_complexity 
                        FROM functions 
                        WHERE cyclomatic_complexity > 10
                        ORDER BY cyclomatic_complexity DESC 
                        LIMIT 10
                    """)
                    complex_funcs = cursor.fetchall()
                    
                    if complex_funcs:
                        markdown_lines.append("### High Complexity Functions (>10)\n\n")
                        markdown_lines.append("| Function | File | Complexity |\n")
                        markdown_lines.append("|----------|------|----------:|\n")
                        for name, file_path, complexity in complex_funcs:
                            markdown_lines.append(f"| `{name}` | `{file_path}` | {complexity} |\n")
                            
            elif 'complexity' in tables:
                # Hybrid mode with complexity table
                cursor.execute("SELECT COUNT(*), AVG(complexity), MAX(complexity) FROM complexity")
                symbol_count, avg_complexity, max_complexity = cursor.fetchone()
                
                if symbol_count and symbol_count > 0:
                    markdown_lines.append(f"**Total Analyzed Symbols:** {symbol_count}\n")
                    markdown_lines.append(f"**Average Cyclomatic Complexity:** {avg_complexity:.2f if avg_complexity else 0}\n")
                    markdown_lines.append(f"**Max Cyclomatic Complexity:** {max_complexity or 0}\n\n")
                    
                    # Most complex symbols
                    cursor.execute("""
                        SELECT symbol_name, file_path, complexity 
                        FROM complexity 
                        WHERE complexity > 10
                        ORDER BY complexity DESC 
                        LIMIT 10
                    """)
                    complex_symbols = cursor.fetchall()
                    
                    if complex_symbols:
                        markdown_lines.append("### High Complexity Symbols (>10)\n\n")
                        markdown_lines.append("| Symbol | File | Complexity |\n")
                        markdown_lines.append("|--------|------|----------:|\n")
                        for name, file_path, complexity in complex_symbols:
                            markdown_lines.append(f"| `{name}` | `{file_path}` | {complexity} |\n")
                    else:
                        markdown_lines.append("*All symbols have complexity ≤ 10*\n")
                else:
                    markdown_lines.append("*No complexity metrics available (hybrid mode may not support this language)*\n")
            else:
                markdown_lines.append("*No complexity analysis data available*\n")
            
            conn.close()
            markdown_lines.append("\n\n---\n\n")
        except Exception as e:
            markdown_lines.append(f"*Error reading analysis database: {e}*\n\n---\n\n")
    
    return "".join(markdown_lines)


def generate_json_output(pack_dir: Path, repo_name: str) -> str:
    """
    Generate a comprehensive JSON document from pack contents.
    
    Args:
        pack_dir: Path to the pack directory
        repo_name: Name of the repository
    
    Returns:
        JSON content as string
    """
    output = {
        "repository": repo_name,
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "analysis_version": "2.0.0"
    }
    
    # Load all JSON files
    json_files = ["manifest.json", "import_graph.json", "name_registry.json", 
                  "security_report.json", "variable_registry.json"]
    
    for json_file in json_files:
        file_path = pack_dir / json_file
        if file_path.exists():
            key = json_file.replace('.json', '')
            try:
                output[key] = json.loads(file_path.read_text(encoding='utf-8'))
            except Exception as e:
                output[key] = {"error": f"Failed to parse {json_file}: {str(e)}"}
    
    # Add repository brief as text
    repo_brief_path = pack_dir / "repoBrief.md"
    if repo_brief_path.exists():
        output["repository_brief"] = repo_brief_path.read_text(encoding='utf-8')
    
    # Add computed statistics
    statistics = {}
    
    # Symbol statistics from name_registry
    if "name_registry" in output and isinstance(output["name_registry"], dict):
        by_kind = {}
        public_count = 0
        for symbol_key, symbol_data in output["name_registry"].items():
            kind = symbol_data.get('kind', 'unknown')
            by_kind[kind] = by_kind.get(kind, 0) + 1
            if symbol_data.get('is_public', False):
                public_count += 1
        
        statistics["symbols"] = {
            "total": len(output["name_registry"]),
            "by_kind": by_kind,
            "public_count": public_count
        }
    
    # Export statistics from variable_registry
    if "variable_registry" in output and isinstance(output["variable_registry"], dict):
        total_exports = 0
        total_internal = 0
        for file_path, variables in output["variable_registry"].items():
            for var in variables:
                if var.get('scope') == 'export':
                    total_exports += 1
                elif var.get('scope') == 'internal':
                    total_internal += 1
        
        statistics["variables"] = {
            "total_exports": total_exports,
            "total_internal": total_internal,
            "files_with_exports": sum(1 for vars in output["variable_registry"].values() if any(v.get('scope') == 'export' for v in vars))
        }
    
    # Import statistics from import_graph
    if "import_graph" in output and isinstance(output["import_graph"], dict):
        total_imports = sum(len(imports) for imports in output["import_graph"].values())
        statistics["imports"] = {
            "total_files": len(output["import_graph"]),
            "total_import_statements": total_imports
        }
    
    # Code statistics from analysis.sqlite
    analysis_db_path = pack_dir / "analysis.sqlite"
    if analysis_db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(analysis_db_path))
            cursor = conn.cursor()
            
            # Check which table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            if 'functions' in tables:
                cursor.execute("""
                    SELECT COUNT(*), 
                           SUM(end_line - start_line + 1), 
                           AVG(cyclomatic_complexity), 
                           MAX(cyclomatic_complexity) 
                    FROM functions
                """)
                func_count, total_lines, avg_complexity, max_complexity = cursor.fetchone()
                
                statistics["code_metrics"] = {
                    "total_functions": func_count or 0,
                    "total_lines": total_lines or 0,
                    "average_complexity": round(avg_complexity, 2) if avg_complexity else 0,
                    "max_complexity": max_complexity or 0
                }
            elif 'complexity' in tables:
                cursor.execute("""
                    SELECT COUNT(*), AVG(complexity), MAX(complexity) 
                    FROM complexity
                """)
                symbol_count, avg_complexity, max_complexity = cursor.fetchone()
                
                statistics["code_metrics"] = {
                    "total_symbols": symbol_count or 0,
                    "average_complexity": round(avg_complexity, 2) if avg_complexity else 0,
                    "max_complexity": max_complexity or 0
                }
            
            conn.close()
        except Exception as e:
            statistics["code_metrics"] = {"error": str(e)}
    
    output["statistics"] = statistics
    
    return json.dumps(output, indent=2, ensure_ascii=False)


@dramatiq.actor(max_retries=0, time_limit=1800000)  # 30 minute timeout
def process_repository(job_id: str, repo_url: str, config: dict = None):
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
        config: Job configuration dict with mode and feature flags
    """
    # Default config if not provided (backward compatibility)
    if config is None:
        config = {
            "mode": "semantic",
            "enable_ast": True,
            "enable_imports": True,
            "enable_complexity": True,
            "enable_security": False,
            "enable_embeddings": True
        }
    
    mode = config.get("mode", "semantic")
    
    print(f"\n{'='*60}")
    print(f"🔧 Starting job {job_id}")
    print(f"   Repository: {repo_url}")
    print(f"   Mode: {mode}")
    print(f"   Config: {config}")
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
        
        # Step 2: Configure pipeline based on user configuration
        # Map frontend config to pipeline config
        pipeline_config = {
            "run_parsing": config.get("enable_ast", True),
            "build_graphs": config.get("enable_imports", True),
            "run_analysis": config.get("enable_complexity", True),
            "run_embeddings": config.get("enable_embeddings", True),
            "run_security_scans": config.get("enable_security", False),
            "build_variable_registry": mode in ["hybrid", "full"],
            "store_spans": mode in ["hybrid", "full"],
            "pack_mode": mode,
        }
        
        # Adjust config based on mode presets (if user chose a mode)
        if mode == "semantic":
            # Lightweight mode - enable only essential features
            pipeline_config.update({
                "run_parsing": config.get("enable_ast", True),
                "build_graphs": config.get("enable_imports", True),
                "run_analysis": config.get("enable_complexity", True),
                "run_embeddings": config.get("enable_embeddings", True),
                "run_security_scans": config.get("enable_security", False),
                "build_variable_registry": False,
                "store_spans": False,
            })
        elif mode == "hybrid":
            # Balanced mode - most features enabled
            pipeline_config.update({
                "run_parsing": config.get("enable_ast", True),
                "build_graphs": config.get("enable_imports", True),
                "run_analysis": config.get("enable_complexity", True),
                "run_embeddings": config.get("enable_embeddings", True),
                "run_security_scans": config.get("enable_security", True),
                "build_variable_registry": True,
                "store_spans": True,
            })
        elif mode == "full":
            # Complete mode - all features enabled
            pipeline_config.update({
                "run_parsing": config.get("enable_ast", True),
                "build_graphs": config.get("enable_imports", True),
                "run_analysis": config.get("enable_complexity", True),
                "run_embeddings": config.get("enable_embeddings", True),
                "run_security_scans": config.get("enable_security", True),
                "build_variable_registry": True,
                "store_spans": True,
            })
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Step 3: Run pipeline
        output_pack_dir = worker_root / "pack"
        daemon_path = "/app/packages/rust-parser-daemon/target/release/rust-parser-daemon"
        
        print(f"\n🔄 Running pipeline with config: {pipeline_config}")
        pipeline = Pipeline(
            repo_path=str(cloned_repo_path),
            output_path=str(output_pack_dir),
            daemon_path=daemon_path,
        )
        
        pipeline.run(config=pipeline_config)
        
        print(f"✓ Pipeline completed successfully")
        
        # Step 4: Package results based on output format
        repo_name = cloned_repo_path.name
        output_format = config.get("output_format", "zip")
        
        if output_format == "markdown":
            # For hybrid/full modes, extract the ZIP first to read its contents
            if mode in ["hybrid", "full"]:
                # Find the created ZIP file
                existing_zip = list(worker_root.glob("reposynth_*.zip"))
                if existing_zip:
                    zip_path = existing_zip[0]
                    # Extract to a temp directory for reading
                    extract_dir = worker_root / "extracted_pack"
                    extract_dir.mkdir(exist_ok=True)
                    
                    print(f"📦 Extracting {zip_path.name} for markdown generation...")
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(extract_dir)
                    
                    # Point to the extracted pack directory
                    # The ZIP contains pack/ directory, so adjust path
                    if (extract_dir / "pack").exists():
                        output_pack_dir = extract_dir / "pack"
                    else:
                        output_pack_dir = extract_dir
            
            # Generate markdown output
            print(f"📄 Generating markdown output...")
            markdown_content = generate_markdown_output(output_pack_dir, repo_name)
            pack_filename = f"reposynth_{repo_name}_{mode}_{job_id}.md"
            pack_path = worker_root / pack_filename
            pack_path.write_text(markdown_content, encoding='utf-8')
            content_type = "text/markdown"
            
        elif output_format == "json":
            # For hybrid/full modes, extract the ZIP first to read its contents
            if mode in ["hybrid", "full"]:
                # Find the created ZIP file
                existing_zip = list(worker_root.glob("reposynth_*.zip"))
                if existing_zip:
                    zip_path = existing_zip[0]
                    # Extract to a temp directory for reading
                    extract_dir = worker_root / "extracted_pack"
                    extract_dir.mkdir(exist_ok=True)
                    
                    print(f"📦 Extracting {zip_path.name} for JSON generation...")
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(extract_dir)
                    
                    # Point to the extracted pack directory
                    if (extract_dir / "pack").exists():
                        output_pack_dir = extract_dir / "pack"
                    else:
                        output_pack_dir = extract_dir
            
            # Generate JSON output
            print(f"📄 Generating JSON output...")
            json_content = generate_json_output(output_pack_dir, repo_name)
            pack_filename = f"reposynth_{repo_name}_{mode}_{job_id}.json"
            pack_path = worker_root / pack_filename
            pack_path.write_text(json_content, encoding='utf-8')
            content_type = "application/json"
            
        elif output_format == "toon":
            # For hybrid/full modes, extract the ZIP first to read its contents
            if mode in ["hybrid", "full"]:
                # Find the created ZIP file
                existing_zip = list(worker_root.glob("reposynth_*.zip"))
                if existing_zip:
                    zip_path = existing_zip[0]
                    # Extract to a temp directory for reading
                    extract_dir = worker_root / "extracted_pack"
                    extract_dir.mkdir(exist_ok=True)
                    
                    print(f"📦 Extracting {zip_path.name} for TOON retrieval...")
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        # We only need pack.toon if it exists
                        try:
                            zf.extract("pack/pack.toon", extract_dir)
                        except KeyError:
                            # Fallback: extract everything if path is different
                            zf.extractall(extract_dir)
                    
                    # Point to the extracted pack directory
                    if (extract_dir / "pack").exists():
                        output_pack_dir = extract_dir / "pack"
                    else:
                        output_pack_dir = extract_dir

            # Retrieve TOON output
            print(f"📄 Retrieving TOON output...")
            toon_path = output_pack_dir / "pack.toon"
            
            if toon_path.exists():
                toon_content = toon_path.read_text(encoding='utf-8')
            else:
                # Fallback: Generate it if missing
                print("Warning: pack.toon not found, generating on the fly...")
                try:
                    from orchestrator.toon_formatter import generate_toon_blueprint, convert_source_to_toon
                    
                    # Load registry and graph
                    registry = json.loads((output_pack_dir / "name_registry.json").read_text(encoding='utf-8'))
                    graph = json.loads((output_pack_dir / "import_graph.json").read_text(encoding='utf-8'))
                    
                    toon_content = generate_toon_blueprint(registry, graph)
                    
                    # Try to add source if available
                    source_spans_path = output_pack_dir / "source_spans.json"
                    if source_spans_path.exists():
                        source_spans = json.loads(source_spans_path.read_text(encoding='utf-8'))
                        toon_content += "\n" + convert_source_to_toon(source_spans)
                        
                except Exception as e:
                    toon_content = f"Error generating TOON format: {str(e)}"
            
            pack_filename = f"reposynth_{repo_name}_{mode}_{job_id}.toon"
            pack_path = worker_root / pack_filename
            pack_path.write_text(toon_content, encoding='utf-8')
            content_type = "text/plain"

        else:  # zip format (default)
            pack_filename = f"reposynth_{repo_name}_{mode}_{job_id}.zip"
            pack_path = worker_root / pack_filename
            content_type = "application/zip"
            
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
        
        # Step 5: Upload to S3 or use local storage
        s3_key = f"{job_id}/{pack_filename}"
        result_url = None
        
        # Try S3 upload first
        if s3_client:
            try:
                print(f"Uploading {pack_path.name} to S3 bucket {S3_BUCKET}...")
                
                # Set content type and disposition based on format
                extra_args = {"ContentType": content_type}
                
                # For markdown, JSON, and TOON, allow inline viewing in browser
                # For ZIP, force download
                if output_format in ["markdown", "json", "toon"]:
                    extra_args["ContentDisposition"] = "inline"
                else:
                    extra_args["ContentDisposition"] = f'attachment; filename="{pack_filename}"'
                
                # Upload file with metadata
                s3_client.upload_file(
                    str(pack_path),
                    S3_BUCKET,
                    s3_key,
                    ExtraArgs=extra_args
                )
                
                # Generate public URL
                if AWS_ENDPOINT_URL:
                    external_url = AWS_ENDPOINT_URL.replace("minio:9000", "localhost:9000")
                    result_url = f"{external_url}/{S3_BUCKET}/{s3_key}"
                else:
                    result_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
                
                print(f"✓ Uploaded to: {result_url}")
                
            except Exception as e:
                print(f"Warning: S3 upload failed: {e}", file=sys.stderr)
                # Fallback to local URL
        
        # If S3 upload failed or client not available, use local URL
        if not result_url:
            # Construct local URL served by API
            # API mounts /app/worker_packs at /packs
            # URL format: http://localhost:8000/packs/<job_id>/<filename>
            # Note: This assumes API is accessible at localhost:8000
            api_base = os.environ.get("API_PUBLIC_URL", "http://localhost:8000")
            result_url = f"{api_base}/packs/{job_id}/{pack_filename}"
            print(f"✓ Using local storage URL: {result_url}")
        
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
