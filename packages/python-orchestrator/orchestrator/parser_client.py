import subprocess
import json
import uuid
import threading
import queue
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys
from gitignore_parser import parse_gitignore


class ParserClient:
    def __init__(self, daemon_path: str):
        self.daemon_path = daemon_path
        self.process = None
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self._start_daemon()
        self._start_io_threads()

    def _start_daemon(self):
        daemon_dir = Path(self.daemon_path).parent
        self.process = subprocess.Popen(
            [self.daemon_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
            cwd=daemon_dir,
        )

    def _start_io_threads(self):
        self.writer_thread = threading.Thread(target=self._writer, daemon=True)
        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.stderr_thread = threading.Thread(target=self._stderr_reader, daemon=True)
        self.writer_thread.start()
        self.reader_thread.start()
        self.stderr_thread.start()

    def _writer(self):
        while True:
            request = self.request_queue.get()
            if request is None:
                break
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
            except (IOError, BrokenPipeError):
                # Daemon likely died, handle error
                break

    def _reader(self):
        for line in iter(self.process.stdout.readline, ""):
            if line:
                self.response_queue.put(json.loads(line))
        # Once loop is done, stdout is closed
        self.response_queue.put(None)

    def _stderr_reader(self):
        for line in iter(self.process.stderr.readline, ""):
            if line:
                # Print any errors from the daemon directly to the console
                print(f"[DAEMON STDERR] {line.strip()}", file=sys.stderr)

    def _discover_files(self, repo_path: Path):
        """Discovers all relevant files, respecting .gitignore rules."""

        gitignore_path = repo_path / ".gitignore"

        # The base path for the parser should be the repo root
        matches = parse_gitignore(gitignore_path, base_dir=repo_path)

        # Include .js files for JavaScript repositories
        all_files = (
            list(repo_path.glob("**/*.py"))
            + list(repo_path.glob("**/*.ts"))
            + list(repo_path.glob("**/*.js"))
        )

        files_to_parse = []
        for file_path in all_files:
            # The matches function expects a path relative to the base_dir
            if not matches(file_path):
                files_to_parse.append(file_path)

        return files_to_parse

    def parse_repository(self, repo_path: str, output_dir: str):
        repo_path = Path(repo_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print("Discovering files to parse (respecting .gitignore)...")
        files_to_parse = self._discover_files(repo_path)

        if not files_to_parse:
            print(f"Warning: No .py, .ts, or .js files found to parse in {repo_path}")
            return

        print(f"Found {len(files_to_parse)} files to parse.")

        pending_requests = {}
        # *** NEW: A dictionary to store our manifest data ***
        ast_manifest = {}

        for file_path in files_to_parse:
            req_id = str(uuid.uuid4())
            request = {"id": req_id, "path": str(file_path.resolve())}
            pending_requests[req_id] = file_path

            # *** NEW: Record the mapping before sending the request ***
            # The key is the unique filename we will create for the AST data
            unique_ast_filename = f"{file_path.name}_{req_id[:8]}.jsonl"
            ast_manifest[unique_ast_filename] = str(file_path.resolve())

            self.request_queue.put(request)

        # *** NEW: Save the manifest file immediately ***
        manifest_path = output_dir / "ast_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(ast_manifest, f, indent=2)

        print(f"AST manifest saved to {manifest_path}")

        # --- The rest of the loop is the same ---
        while pending_requests:
            try:
                response = self.response_queue.get(
                    timeout=20
                )  # Increased timeout for larger repos
                if response is None:
                    print(
                        "Daemon stdout closed unexpectedly. Check stderr for daemon errors.",
                        file=sys.stderr,
                    )
                    break

                req_id = response.get("id")
                if req_id in pending_requests:
                    file_path = pending_requests.pop(req_id)
                    if response.get("error"):
                        print(
                            f"Error parsing {file_path}: {response['error']}",
                            file=sys.stderr,
                        )
                    else:
                        unique_ast_filename = f"{file_path.name}_{req_id[:8]}.jsonl"
                        output_file = output_dir / unique_ast_filename
                        with open(output_file, "w", encoding="utf-8") as f:
                            for node in response.get("ast", []):
                                f.write(json.dumps(node) + "\n")

            except queue.Empty:
                print(
                    "Timeout waiting for response. Daemon might be stuck or crashed.",
                    file=sys.stderr,
                )
                break

        print("Parsing stage complete.")

    def shutdown(self):
        self.request_queue.put(None)
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

        self.writer_thread.join(timeout=1)
        self.reader_thread.join(timeout=1)

    def parse_files(self, files_to_parse: list[Path], output_dir: str):
        if not files_to_parse:
            print("No files to parse.")
            return

        output_dir = Path(output_dir)
        pending_requests = {}

        for file_path in files_to_parse:
            req_id = str(uuid.uuid4())
            request = {"id": req_id, "path": str(file_path.resolve())}
            pending_requests[req_id] = file_path
            self.request_queue.put(request)
        
        while pending_requests:
            try:
                response = self.response_queue.get(timeout=20)
                if response is None:
                    print("Daemon stdout closed.", file=sys.stderr)
                    break
                
                req_id = response.get("id")
                if req_id in pending_requests:
                    file_path = pending_requests.pop(req_id)
                    if not response.get("error"):
                        # Create a stable unique name based on the file path
                        relative_path = file_path.relative_to(Path.cwd()) # Adjust if needed
                        safe_name = str(relative_path).replace('/', '_').replace('\\', '_')
                        unique_ast_filename = f"{safe_name}.jsonl"
                        
                        output_file = output_dir / unique_ast_filename
                        with open(output_file, 'w', encoding='utf-8') as f:
                             for node in response.get('ast', []):
                                f.write(json.dumps(node) + '\n')
                    else:
                        print(f"Error parsing {file_path}: {response['error']}", file=sys.stderr)
            except queue.Empty:
                print("Timeout waiting for response from daemon.", file=sys.stderr)
                break
        
        print("Parsing of modified files complete.")


if __name__ == "__main__":
    # --- THIS IS THE ROBUST SOLUTION ---

    # 1. Get the absolute path to the directory containing THIS script.
    script_dir = Path(__file__).parent.resolve()

    # 2. Build the path to the daemon relative to this script's location.
    #    Path objects let us use '/' and it works on all OSes.
    daemon_path = (
        script_dir
        / "../../../packages/rust-parser-daemon/target/release/rust-parser-daemon"
    )

    # 3. Add the .exe extension ONLY if we are on Windows.
    if sys.platform == "win32":
        daemon_path = daemon_path.with_suffix(".exe")

    # 4. Check if the file actually exists before trying to run it.
    if not daemon_path.exists():
        print(f"FATAL: Daemon executable not found at expected path: {daemon_path}")
        print("Please run 'cargo build --release' in 'packages/rust-parser-daemon'")
        sys.exit(1)

    REPO_TO_PARSE = script_dir / "../../../"  # Parse the whole project!
    OUTPUT_DIR = script_dir / "./ast_raw"

    print(f"Daemon path: {daemon_path}")
    print(f"Repo to parse: {REPO_TO_PARSE}")
    print(f"Output directory: {OUTPUT_DIR}")

    client = ParserClient(daemon_path=str(daemon_path))
    try:
        client.parse_repository(
            repo_path=str(REPO_TO_PARSE), output_dir=str(OUTPUT_DIR)
        )
    finally:
        print("Shutting down daemon...")
        client.shutdown()
