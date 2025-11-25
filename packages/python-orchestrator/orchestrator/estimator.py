# --- FILE: packages/python-orchestrator/orchestrator/estimator.py ---
"""
Token and time estimation for RepoSynth pipeline.
Week 6: Fast, accurate estimation using pygount + tiktoken sampling.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict
import random
import sys

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not installed. Using fallback token estimation.", file=sys.stderr)

try:
    from pygount import SourceAnalysis
    from pygount.analysis import SourceState
    PYGOUNT_AVAILABLE = True
except ImportError:
    PYGOUNT_AVAILABLE = False
    SourceState = None
    print("Warning: pygount not installed. Using fallback LoC counting.", file=sys.stderr)


# Performance benchmarks (calibrated from real runs)
# These can be updated based on actual measurements
BENCHMARKS = {
    # Files processed per second (Conservative estimates for large repos)
    'parsing_files_per_sec': 50,      # Was 100
    'graph_building_files_per_sec': 100, # Was 50 (Optimized with index)
    'analysis_files_per_sec': 15,     # Was 20

    # Special operations
    'embedding_apis_per_sec': 5,      # Was 10 (ML model is slow)
    'variable_registry_files_per_sec': 20, # Was 30

    # Fixed overhead (startup time)
    'parsing_overhead_sec': 2.0,
    'embedding_model_load_sec': 5.0,
}


@dataclass
class LanguageStats:
    """Statistics for a single programming language."""
    files: int = 0
    lines: int = 0
    code: int = 0
    comments: int = 0
    blanks: int = 0
    estimated_tokens: int = 0


@dataclass
class FeatureEstimate:
    """Estimate for a single pipeline feature."""
    enabled: bool
    estimated_tokens: int
    estimated_time_seconds: float
    description: str


@dataclass
class EstimateResult:
    """Complete estimation result."""
    total_tokens: int
    total_time_seconds: float
    base_tokens: int
    num_files: int
    total_lines: int
    language_breakdown: Dict[str, LanguageStats]
    feature_breakdown: Dict[str, FeatureEstimate]
    summary: str
    warnings: List[str]


def _count_lines_with_pygount(repo_path: Path) -> Dict[str, LanguageStats]:
    """
    Count lines of code using pygount.
    Returns statistics per language.
    """
    if not PYGOUNT_AVAILABLE:
        raise ImportError("pygount is required for line counting. Install with: pip install pygount")

    stats_by_language = defaultdict(LanguageStats)

    print(f"[DEBUG] _count_lines_with_pygount: Scanning {repo_path}", file=sys.stderr)
    files_checked = 0
    files_skipped_hidden = 0
    files_skipped_dir = 0
    files_analyzed = 0

    # Scan all files in the repository
    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue

        files_checked += 1

        # Get relative path from repo root (not absolute path)
        try:
            relative_path = file_path.relative_to(repo_path)
        except ValueError:
            continue

        # Skip files inside hidden directories (like .git/, .github/)
        # But allow dotfiles at root or in normal directories (like .babelrc.js)
        # Check if any parent directory (not the file itself) starts with '.'
        if any(part.startswith('.') for part in relative_path.parts[:-1]):  # [:-1] excludes the filename
            files_skipped_hidden += 1
            if files_skipped_hidden <= 3:  # Show first 3
                print(f"[DEBUG] Skipped (hidden dir): {relative_path}", file=sys.stderr)
            continue
        if any(part in ['node_modules', '__pycache__', 'venv', '.venv', 'target', 'build', 'dist']
               for part in relative_path.parts):
            files_skipped_dir += 1
            continue

        try:
            analysis = SourceAnalysis.from_file(file_path, "pygount", encoding="utf-8")

            # Debug: show what pygount returns for first few files
            if files_checked <= 5:
                print(f"[DEBUG] File: {relative_path}", file=sys.stderr)
                print(f"[DEBUG]   State: {analysis.state}", file=sys.stderr)
                print(f"[DEBUG]   Language: {analysis.language}", file=sys.stderr)
                print(f"[DEBUG]   Code count: {analysis.code_count}", file=sys.stderr)

            if analysis.state == SourceState.analyzed and analysis.language:
                lang = analysis.language
                stats = stats_by_language[lang]

                stats.files += 1
                stats.code += analysis.code_count
                stats.comments += analysis.documentation_count + analysis.string_count
                stats.blanks += analysis.empty_count
                stats.lines += analysis.code_count + analysis.documentation_count + analysis.string_count + analysis.empty_count

                files_analyzed += 1
                if files_analyzed <= 3:  # Show first 3
                    print(f"[DEBUG] Analyzed: {relative_path} -> {lang}", file=sys.stderr)
            else:
                if files_checked <= 5:
                    print(f"[DEBUG]   -> SKIPPED (state={analysis.state}, lang={analysis.language})", file=sys.stderr)

        except Exception as e:
            # Skip files that can't be analyzed
            if files_checked <= 5:
                print(f"[DEBUG] Exception analyzing {relative_path}: {e}", file=sys.stderr)
            continue

    print(f"[DEBUG] Files checked: {files_checked}, skipped (hidden): {files_skipped_hidden}, skipped (dirs): {files_skipped_dir}, analyzed: {files_analyzed}", file=sys.stderr)
    print(f"[DEBUG] Languages found: {list(stats_by_language.keys())}", file=sys.stderr)

    return dict(stats_by_language)


def _sample_files_for_tokenization(repo_path: Path, stats_by_language: Dict[str, LanguageStats],
                                   sample_size: int = 50) -> Dict[str, List[Path]]:
    """
    Sample representative files from each language for accurate token counting.
    Returns a dict mapping language -> list of file paths to sample.
    """
    sampled_files = defaultdict(list)

    # Map common languages to file extensions
    extension_to_language = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.tsx': 'TypeScript',
        '.jsx': 'JavaScript',
        '.rs': 'Rust',
        '.go': 'Go',
        '.java': 'Java',
        '.c': 'C',
        '.cpp': 'C++',
        '.h': 'C',
        '.hpp': 'C++',
    }

    # Collect all files by language
    files_by_language = defaultdict(list)
    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue

        # Get relative path from repo root
        try:
            relative_path = file_path.relative_to(repo_path)
        except ValueError:
            continue

        # Skip files inside hidden directories (but allow dotfiles)
        if any(part.startswith('.') for part in relative_path.parts[:-1]):
            continue
        if any(part in ['node_modules', '__pycache__', 'venv', '.venv', 'target']
               for part in relative_path.parts):
            continue

        # Determine language from extension
        ext = file_path.suffix.lower()
        if ext in extension_to_language:
            lang = extension_to_language[ext]
            if lang in stats_by_language:  # Only sample languages we found
                files_by_language[lang].append(file_path)

    # Sample files from each language (stratified sampling)
    for lang, files in files_by_language.items():
        if not files:
            continue

        # Sample proportionally, but cap at sample_size per language
        num_samples = min(len(files), max(5, sample_size // len(files_by_language)))
        sampled_files[lang] = random.sample(files, num_samples)

    return dict(sampled_files)


def _count_tokens_with_tiktoken(file_paths: List[Path]) -> int:
    """
    Count actual tokens in given files using tiktoken.
    Returns total token count.
    """
    if not TIKTOKEN_AVAILABLE:
        raise ImportError("tiktoken is required for accurate token counting. Install with: pip install tiktoken")

    # Use cl100k_base encoding (GPT-4, GPT-3.5-turbo)
    encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                tokens = encoding.encode(content)
                total_tokens += len(tokens)
        except Exception:
            # Skip files that can't be read
            continue

    return total_tokens


def _estimate_tokens_accurate(repo_path: Path, stats_by_language: Dict[str, LanguageStats]) -> int:
    """
    Accurate token estimation by sampling files and measuring with tiktoken.
    Updates stats_by_language with estimated_tokens.
    Returns total estimated tokens.
    """
    print(f"[DEBUG] _estimate_tokens_accurate: TIKTOKEN_AVAILABLE = {TIKTOKEN_AVAILABLE}", file=sys.stderr)

    if not TIKTOKEN_AVAILABLE:
        # Fallback: use rough heuristic (0.5 tokens per line of code)
        print("Warning: Using fallback token estimation (0.5 tokens/line)", file=sys.stderr)
        total_tokens = 0
        for stats in stats_by_language.values():
            stats.estimated_tokens = int(stats.code * 0.5)
            total_tokens += stats.estimated_tokens
        print(f"[DEBUG] Fallback estimation: {total_tokens} tokens", file=sys.stderr)
        return total_tokens

    # Sample files for each language
    print(f"[DEBUG] Sampling files for tokenization...", file=sys.stderr)
    sampled_files = _sample_files_for_tokenization(repo_path, stats_by_language)
    print(f"[DEBUG] Sampled files: {[(lang, len(files)) for lang, files in sampled_files.items()]}", file=sys.stderr)

    total_estimated_tokens = 0

    for lang, stats in stats_by_language.items():
        if lang not in sampled_files or not sampled_files[lang]:
            # No samples - use fallback
            stats.estimated_tokens = int(stats.code * 0.5)
            total_estimated_tokens += stats.estimated_tokens
            continue

        # Count tokens in sampled files
        sample_tokens = _count_tokens_with_tiktoken(sampled_files[lang])

        # Calculate lines of code in sampled files
        sample_lines = 0
        for file_path in sampled_files[lang]:
            try:
                analysis = SourceAnalysis.from_file(file_path, "pygount", encoding="utf-8")
                if analysis.state == SourceState.analyzed:  # FIX: Use enum, not string
                    sample_lines += analysis.code_count
            except Exception:
                continue

        # Calculate tokens per line ratio
        if sample_lines > 0:
            tokens_per_line = sample_tokens / sample_lines
        else:
            tokens_per_line = 0.5  # Fallback

        # Apply ratio to full codebase
        stats.estimated_tokens = int(stats.code * tokens_per_line)
        total_estimated_tokens += stats.estimated_tokens

        print(f"[DEBUG] {lang}: {stats.code} LoC * {tokens_per_line:.2f} = {stats.estimated_tokens} tokens", file=sys.stderr)

    print(f"[DEBUG] Total estimated tokens: {total_estimated_tokens}", file=sys.stderr)
    return total_estimated_tokens


def _estimate_feature_overhead(config: dict, stats_by_language: Dict[str, LanguageStats],
                               base_tokens: int, num_files: int) -> Dict[str, FeatureEstimate]:
    """
    Estimate token and time overhead for each enabled pipeline feature.
    """
    features = {}

    # Parsing - always runs if enabled
    if config.get('run_parsing', True):
        time_sec = (num_files / BENCHMARKS['parsing_files_per_sec']) + BENCHMARKS['parsing_overhead_sec']
        features['run_parsing'] = FeatureEstimate(
            enabled=True,
            estimated_tokens=0,  # No token overhead
            estimated_time_seconds=time_sec,
            description=f"Parse {num_files} files into AST"
        )

    # Graph building
    if config.get('build_graphs', True):
        # Token overhead: ~10% for FQN resolution
        overhead_tokens = int(base_tokens * 0.1)
        time_sec = num_files / BENCHMARKS['graph_building_files_per_sec']
        features['build_graphs'] = FeatureEstimate(
            enabled=True,
            estimated_tokens=overhead_tokens,
            estimated_time_seconds=time_sec,
            description=f"Build import graph and name registry for {num_files} files"
        )
    else:
        features['build_graphs'] = FeatureEstimate(
            enabled=False, estimated_tokens=0, estimated_time_seconds=0,
            description="Skipped"
        )

    # Static analysis
    if config.get('run_analysis', True):
        # Only analyzes Python files
        python_files = stats_by_language.get('Python', LanguageStats()).files
        overhead_tokens = int(base_tokens * 0.05)
        time_sec = python_files / BENCHMARKS['analysis_files_per_sec'] if python_files > 0 else 0.1
        features['run_analysis'] = FeatureEstimate(
            enabled=True,
            estimated_tokens=overhead_tokens,
            estimated_time_seconds=time_sec,
            description=f"Run Ruff analysis on {python_files} Python files"
        )
    else:
        features['run_analysis'] = FeatureEstimate(
            enabled=False, estimated_tokens=0, estimated_time_seconds=0,
            description="Skipped"
        )

    # Embeddings - EXPENSIVE!
    if config.get('run_embeddings', True):
        # Rough estimate: ~10-15% of definitions are public APIs
        # Assume 1 definition per 50 lines of code
        estimated_apis = max(10, int(sum(s.code for s in stats_by_language.values()) / 50 * 0.12))
        # Each API gets embedded: ~100 tokens per API
        overhead_tokens = estimated_apis * 100
        time_sec = (estimated_apis / BENCHMARKS['embedding_apis_per_sec']) + BENCHMARKS['embedding_model_load_sec']
        features['run_embeddings'] = FeatureEstimate(
            enabled=True,
            estimated_tokens=overhead_tokens,
            estimated_time_seconds=time_sec,
            description=f"Generate embeddings for ~{estimated_apis} public APIs"
        )
    else:
        features['run_embeddings'] = FeatureEstimate(
            enabled=False, estimated_tokens=0, estimated_time_seconds=0,
            description="Skipped"
        )

    # Variable registry
    if config.get('build_variable_registry', False):
        overhead_tokens = int(base_tokens * 0.15)
        time_sec = num_files / BENCHMARKS['variable_registry_files_per_sec']
        features['build_variable_registry'] = FeatureEstimate(
            enabled=True,
            estimated_tokens=overhead_tokens,
            estimated_time_seconds=time_sec,
            description=f"Build variable registry for {num_files} files"
        )
    else:
        features['build_variable_registry'] = FeatureEstimate(
            enabled=False, estimated_tokens=0, estimated_time_seconds=0,
            description="Skipped"
        )

    # Store spans - no token overhead, just storage
    if config.get('store_spans', False):
        features['store_spans'] = FeatureEstimate(
            enabled=True,
            estimated_tokens=0,
            estimated_time_seconds=0.5,
            description="Store source code spans in ZIP"
        )
    else:
        features['store_spans'] = FeatureEstimate(
            enabled=False, estimated_tokens=0, estimated_time_seconds=0,
            description="Skipped"
        )

    return features


def estimate_tokens(repo_path: Path, config: dict) -> EstimateResult:
    """
    Fast, accurate token and time estimation for a repository.

    This function:
    1. Uses pygount to count lines of code by language
    2. Samples representative files and uses tiktoken for accurate token counting
    3. Estimates tokens and time for each pipeline feature
    4. Returns comprehensive estimates without running the full pipeline

    Args:
        repo_path: Path to the repository to analyze
        config: Pipeline configuration dict

    Returns:
        EstimateResult with detailed token and time estimates
    """
    print(f"[DEBUG] estimate_tokens called with repo_path: {repo_path}", file=sys.stderr)
    print(f"[DEBUG] repo_path exists: {repo_path.exists()}", file=sys.stderr)
    print(f"[DEBUG] repo_path is_dir: {repo_path.is_dir()}", file=sys.stderr)
    if repo_path.exists():
        file_count = len(list(repo_path.rglob("*")))
        print(f"[DEBUG] Total files found with rglob: {file_count}", file=sys.stderr)

    # Step 1: Count lines of code
    stats_by_language = _count_lines_with_pygount(repo_path)
    print(f"[DEBUG] stats_by_language result: {stats_by_language}", file=sys.stderr)

    if not stats_by_language:
        # Empty repository
        return EstimateResult(
            total_tokens=0,
            total_time_seconds=0.0,
            base_tokens=0,
            num_files=0,
            total_lines=0,
            language_breakdown={},
            feature_breakdown={},
            summary="Empty repository",
            warnings=[]
        )

    # Step 2: Estimate tokens accurately with tiktoken sampling
    base_tokens = _estimate_tokens_accurate(repo_path, stats_by_language)

    # Calculate totals
    num_files = sum(s.files for s in stats_by_language.values())
    total_lines = sum(s.lines for s in stats_by_language.values())

    # Step 3: Estimate feature overhead
    feature_breakdown = _estimate_feature_overhead(config, stats_by_language, base_tokens, num_files)

    # Step 4: Calculate totals
    total_tokens = base_tokens + sum(f.estimated_tokens for f in feature_breakdown.values())
    total_time = sum(f.estimated_time_seconds for f in feature_breakdown.values() if f.enabled)

    # Step 5: Generate warnings
    warnings = []
    if total_tokens > 100000:
        warnings.append("Large repository - token count exceeds 100K")
    if total_time > 60:
        warnings.append(f"Estimated time is {int(total_time/60)} minutes - consider disabling expensive features")
    if feature_breakdown.get('run_embeddings', FeatureEstimate(False, 0, 0, "")).enabled and num_files > 200:
        warnings.append("Embeddings on large repos can be slow - consider running without embeddings first")

    # Step 6: Generate summary
    summary = (
        f"Repository: {num_files} files, {total_lines:,} lines. "
        f"Estimated: {total_tokens:,} tokens, ~{total_time:.1f} seconds."
    )

    return EstimateResult(
        total_tokens=total_tokens,
        total_time_seconds=round(total_time, 2),
        base_tokens=base_tokens,
        num_files=num_files,
        total_lines=total_lines,
        language_breakdown=stats_by_language,
        feature_breakdown=feature_breakdown,
        summary=summary,
        warnings=warnings
    )
