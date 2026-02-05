"""
Per-Repo Fine-Tuning Pipeline for Learned Query Module.

This module provides automated data collection and quick fine-tuning
for adapting the learned query module to a specific codebase.

Workflow:
1. Load repo and create file embeddings (~30 sec)
2. Collect training data from real CCE spikes (~5 min)
3. Fine-tune the model on repo-specific data (~2-3 min)
4. Save and use forever

Usage:
    >>> from orchestrator.retrieval.learned_query.repo_tuning import RepoTuner
    >>> tuner = RepoTuner(repo_path="/path/to/repo")
    >>> tuner.collect_training_data(sample_queries, llm, tokenizer)
    >>> tuner.fine_tune(epochs=5)
    >>> tuner.save("repo_model.pt")
"""

import os
import re
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Callable
from pathlib import Path


@dataclass
class CCESpikeExample:
    """A single CCE spike training example."""
    confused_tokens: List[str]
    confused_probs: List[float]
    original_query: str
    generated_context: str
    relevant_files: List[str]
    spike_position: int = 0
    cce_value: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'CCESpikeExample':
        return cls(**d)


@dataclass
class RepoTrainingData:
    """Collection of training data for a repo."""
    repo_name: str
    repo_path: str
    file_paths: List[str]
    examples: List[CCESpikeExample] = field(default_factory=list)
    collection_time: float = 0.0

    def __len__(self):
        return len(self.examples)

    def save(self, path: str):
        """Save training data to JSON."""
        data = {
            'repo_name': self.repo_name,
            'repo_path': self.repo_path,
            'file_paths': self.file_paths,
            'examples': [ex.to_dict() for ex in self.examples],
            'collection_time': self.collection_time,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'RepoTrainingData':
        """Load training data from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            repo_name=data['repo_name'],
            repo_path=data['repo_path'],
            file_paths=data['file_paths'],
            examples=[CCESpikeExample.from_dict(ex) for ex in data['examples']],
            collection_time=data.get('collection_time', 0.0),
        )


class CCEDataCollector:
    """
    Collects training data from real CCE spikes during LLM generation.

    This captures the ACTUAL confused tokens that the model produces,
    not synthetic/artificial data. This is crucial for learning patterns
    that transfer to real usage.
    """

    # Stopwords and uninformative tokens to filter out
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
        'if', 'or', 'because', 'until', 'while', 'although', 'though',
        'this', 'that', 'these', 'those', 'it', 'its', 'you', 'your', 'we',
        'our', 'they', 'their', 'what', 'which', 'who', 'whom', 'whose',
        'use', 'using', 'used', 'make', 'made', 'get', 'got', 'set',
        # Common LLM response starters
        'yes', 'no', 'sure', 'here', 'well', 'okay', 'ok',
    }

    # Punctuation patterns to filter
    PUNCT_PATTERN = re.compile(r'^[\s\W]+$')

    def __init__(
        self,
        tokenizer,
        cce_threshold: float = 3.0,
        spike_cooldown: int = 15,
        top_k_tokens: int = 5,
        min_token_prob: float = 0.01,
    ):
        """
        Initialize data collector.

        Args:
            tokenizer: HuggingFace tokenizer for the LLM
            cce_threshold: Entropy threshold for spike detection
            spike_cooldown: Minimum tokens between spikes
            top_k_tokens: Number of confused tokens to capture per spike
            min_token_prob: Minimum probability for a token to be considered
        """
        self.tokenizer = tokenizer
        self.cce_threshold = cce_threshold
        self.spike_cooldown = spike_cooldown
        self.top_k_tokens = top_k_tokens
        self.min_token_prob = min_token_prob

        # Build code token indices for CCE computation
        self._build_code_indices()

    def _build_code_indices(self):
        """Build indices of code-related tokens."""
        self.code_indices = []
        vocab_size = min(self.tokenizer.vocab_size, 32000)

        for idx in range(vocab_size):
            try:
                token = self.tokenizer.decode([idx])
                # Match code-like patterns
                if re.search(r'[_A-Z]{2,}|\(|\)|\{|\}|def|class|import|return|async|await', token):
                    self.code_indices.append(idx)
            except:
                pass

        self.code_indices = torch.tensor(self.code_indices)

    def compute_cce(self, logits: torch.Tensor) -> float:
        """Compute Code-Conditional Entropy."""
        from scipy.stats import entropy as scipy_entropy

        # Convert to float32 to avoid bfloat16 issues with numpy
        logits = logits.float()
        probs = F.softmax(logits, dim=-1)
        device = probs.device

        if len(self.code_indices) > 0:
            code_indices = self.code_indices.to(device)
            code_probs = probs[code_indices].cpu().numpy()
            total = code_probs.sum()
            if total > 1e-10:
                code_probs = code_probs / total
                return float(scipy_entropy(code_probs + 1e-10, base=2))
        return 0.0

    # Common non-code English words to filter (beyond stopwords)
    GENERIC_WORDS = {
        'editor', 'professor', 'member', 'engine', 'product', 'powerful',
        'useful', 'effective', 'flexible', 'robust', 'simple', 'easy',
        'important', 'necessary', 'possible', 'available', 'different',
        'specific', 'common', 'general', 'standard', 'basic', 'main',
        'first', 'second', 'third', 'last', 'next', 'previous', 'current',
        'new', 'old', 'good', 'bad', 'best', 'better', 'great', 'large',
        'small', 'high', 'low', 'long', 'short', 'full', 'empty',
        'true', 'false', 'yes', 'also', 'however', 'therefore', 'example',
        'following', 'above', 'below', 'here', 'there', 'now', 'then',
    }

    def _is_code_like_token(self, token: str) -> bool:
        """Check if token looks like code (not generic English)."""
        clean = token.strip().lower()

        # Definitely code-like patterns
        if '_' in token:  # snake_case
            return True
        if re.search(r'[a-z][A-Z]', token):  # camelCase
            return True
        if token.isupper() and len(token) >= 2:  # CONSTANTS
            return True
        if re.match(r'^[A-Z][a-z]+[A-Z]', token):  # PascalCase
            return True

        # Programming keywords
        code_keywords = {
            'async', 'await', 'def', 'class', 'import', 'from', 'return',
            'yield', 'lambda', 'try', 'except', 'finally', 'raise', 'with',
            'self', 'cls', 'init', 'str', 'int', 'dict', 'list', 'tuple',
            'bool', 'none', 'http', 'https', 'url', 'uri', 'api', 'json',
            'xml', 'html', 'css', 'sql', 'get', 'post', 'put', 'delete',
            'patch', 'head', 'options', 'request', 'response', 'client',
            'server', 'socket', 'stream', 'buffer', 'encode', 'decode',
            'parse', 'serialize', 'config', 'timeout', 'retry', 'error',
            'exception', 'handler', 'callback', 'async', 'sync', 'pool',
            'connection', 'session', 'cookie', 'header', 'body', 'content',
            'auth', 'token', 'bearer', 'basic', 'oauth', 'ssl', 'tls',
            'cert', 'proxy', 'redirect', 'status', 'code', 'method',
        }
        if clean in code_keywords:
            return True

        # Filter generic English words
        if clean in self.GENERIC_WORDS:
            return False

        return False

    def _is_informative_token(self, token: str, prob: float) -> bool:
        """Check if a token is informative for file matching."""
        clean = token.strip().lower()

        # Filter by probability
        if prob < self.min_token_prob:
            return False

        # Filter too short
        if len(clean) < 2:
            return False

        # Filter stopwords
        if clean in self.STOPWORDS:
            return False

        # Filter pure punctuation/whitespace
        if self.PUNCT_PATTERN.match(token):
            return False

        # Filter quote patterns
        if token in ['"""', "'''", '``', "''", '""', '\\\\', '//', '##']:
            return False

        # Filter generic English words
        if clean in self.GENERIC_WORDS:
            return False

        # Prioritize code-like tokens
        if self._is_code_like_token(token):
            return True

        # Keep if alphanumeric and reasonable length (but less preferred)
        if token.isalnum() and len(token) >= 4:
            return True

        return False

    def get_confused_tokens(
        self,
        logits: torch.Tensor,
    ) -> Tuple[List[str], List[float]]:
        """Extract top confused tokens from logits, prioritizing code-like tokens."""
        # Convert to float32 to avoid bfloat16 issues
        logits = logits.float()
        probs = F.softmax(logits, dim=-1)

        # Strategy: First look for code-like tokens, then fall back to any informative token
        top_probs, top_indices = torch.topk(probs, min(100, probs.shape[-1]))

        code_tokens = []
        code_probs = []
        other_tokens = []
        other_probs = []

        for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
            try:
                token = self.tokenizer.decode([idx]).strip()
                if not token or len(token) < 2:
                    continue

                prob_val = float(prob)
                clean = token.lower()

                # Skip stopwords and punctuation
                if clean in self.STOPWORDS or self.PUNCT_PATTERN.match(token):
                    continue
                if token in ['"""', "'''", '``', "''", '""']:
                    continue
                if clean in self.GENERIC_WORDS:
                    continue

                # Categorize as code-like or other
                if self._is_code_like_token(token):
                    if len(code_tokens) < self.top_k_tokens:
                        code_tokens.append(token)
                        code_probs.append(prob_val)
                elif prob_val >= self.min_token_prob and len(token) >= 3:
                    if len(other_tokens) < self.top_k_tokens:
                        other_tokens.append(token)
                        other_probs.append(prob_val)

            except:
                pass

        # Prefer code tokens, fill with others if needed
        tokens = code_tokens[:self.top_k_tokens]
        token_probs = code_probs[:self.top_k_tokens]

        remaining = self.top_k_tokens - len(tokens)
        if remaining > 0:
            tokens.extend(other_tokens[:remaining])
            token_probs.extend(other_probs[:remaining])

        return tokens, token_probs

    def collect_from_generation(
        self,
        llm,
        query: str,
        file_labeler: Callable[[List[str], str], List[str]],
        max_tokens: int = 100,
        device: torch.device = None,
    ) -> List[CCESpikeExample]:
        """
        Collect CCE spike examples from a single generation.

        Args:
            llm: The language model (HuggingFace)
            query: The query to generate from
            file_labeler: Function that labels relevant files given tokens and query
            max_tokens: Maximum tokens to generate
            device: Torch device

        Returns:
            List of CCESpikeExample from this generation
        """
        if device is None:
            device = next(llm.parameters()).device

        examples = []
        last_spike = -self.spike_cooldown

        # Encode prompt
        prompt = f"Question: {query}\nAnswer:"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        generated_text = ""

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = llm(input_ids)
                logits = outputs.logits[:, -1, :]

            # Check for CCE spike
            cce = self.compute_cce(logits[0])
            is_spike = (cce > self.cce_threshold and
                       (step - last_spike) >= self.spike_cooldown)

            if is_spike:
                last_spike = step

                # Capture confused tokens
                tokens, probs = self.get_confused_tokens(logits[0])

                if tokens:
                    # Label relevant files using provided labeler
                    relevant_files = file_labeler(tokens, query)

                    if relevant_files:
                        examples.append(CCESpikeExample(
                            confused_tokens=tokens,
                            confused_probs=probs,
                            original_query=query,
                            generated_context=generated_text,
                            relevant_files=relevant_files,
                            spike_position=step,
                            cce_value=cce,
                        ))

            # Sample next token (convert to float32 for multinomial compatibility)
            probs = F.softmax(logits.float(), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_text += self.tokenizer.decode(next_token[0])

            # Stop on EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return examples


class HeuristicLabeler:
    """
    Labels files using keyword matching heuristic.

    This provides automatic labeling for training data collection.
    Uses weighted scoring with path names, class/function names, imports, and docstrings.
    """

    def __init__(self, file_paths: List[str], file_contents: Dict[str, str]):
        self.file_paths = file_paths
        self.file_contents = file_contents

        # Build comprehensive keyword index with weights
        self.file_keywords = {}  # path -> {keyword: weight}

        for path in file_paths:
            keywords = {}

            # Path keywords (high weight - filename is very indicative)
            parts = re.split(r'[/\\._]', path.lower())
            for p in parts:
                if len(p) >= 2 and p not in ('py', 'init'):
                    keywords[p] = keywords.get(p, 0) + 3.0

            content = file_contents.get(path, '')

            # Class names (high weight)
            for match in re.finditer(r'class\s+(\w+)', content):
                name = match.group(1).lower()
                keywords[name] = keywords.get(name, 0) + 2.5
                # Also add split camelCase
                for part in re.findall(r'[A-Z][a-z]+', match.group(1)):
                    keywords[part.lower()] = keywords.get(part.lower(), 0) + 1.0

            # Function names (high weight)
            for match in re.finditer(r'def\s+(\w+)', content):
                name = match.group(1).lower()
                if not name.startswith('_'):  # Skip private methods
                    keywords[name] = keywords.get(name, 0) + 2.0
                # Split snake_case
                for part in name.split('_'):
                    if len(part) >= 3:
                        keywords[part] = keywords.get(part, 0) + 0.5

            # Import names (medium weight - indicates dependencies)
            for match in re.finditer(r'(?:from|import)\s+([\w.]+)', content):
                parts = match.group(1).split('.')
                for part in parts:
                    if len(part) >= 3:
                        keywords[part.lower()] = keywords.get(part.lower(), 0) + 1.0

            # Docstring keywords (lower weight)
            for match in re.finditer(r'"""(.+?)"""', content, re.DOTALL):
                doc = match.group(1)
                for word in re.findall(r'\b([a-z]{4,})\b', doc.lower()):
                    if word not in CCEDataCollector.STOPWORDS:
                        keywords[word] = keywords.get(word, 0) + 0.3

            # Constants (medium weight)
            for match in re.finditer(r'\b([A-Z][A-Z_]{2,})\b', content):
                keywords[match.group(1).lower()] = keywords.get(match.group(1).lower(), 0) + 1.5

            self.file_keywords[path] = keywords

    def __call__(self, tokens: List[str], query: str, top_k: int = 3) -> List[str]:
        """Label relevant files for given tokens and query."""
        # Build query keywords with weights
        query_kw = {}

        # Tokens from CCE spike (higher weight)
        for t in tokens:
            clean = t.lower().strip()
            if len(clean) >= 2:
                query_kw[clean] = query_kw.get(clean, 0) + 2.0
                # Handle subword tokens (e.g., "▁auth" -> "auth")
                if clean.startswith('▁'):
                    query_kw[clean[1:]] = query_kw.get(clean[1:], 0) + 2.0

        # Query words (medium weight)
        for word in re.findall(r'\b([a-z]{2,})\b', query.lower()):
            if word not in CCEDataCollector.STOPWORDS:
                query_kw[word] = query_kw.get(word, 0) + 1.0

        # Score files using weighted matching
        scores = {}
        for path, file_kw in self.file_keywords.items():
            score = 0.0
            matched_keywords = []
            for qk, qw in query_kw.items():
                if qk in file_kw:
                    score += qw * file_kw[qk]
                    matched_keywords.append(qk)
                # Partial match for longer keywords
                elif len(qk) >= 4:
                    for fk, fw in file_kw.items():
                        if qk in fk or fk in qk:
                            score += qw * fw * 0.5
                            break

            scores[path] = score

        # Return top files with score > threshold
        sorted_files = sorted(scores.items(), key=lambda x: -x[1])
        min_score = 2.0  # Minimum score to be considered relevant
        return [f for f, s in sorted_files[:top_k] if s >= min_score]


class RepoTuner:
    """
    Main class for per-repo fine-tuning.

    Handles the complete workflow:
    1. Load repo and create file embeddings
    2. Collect training data from real CCE spikes
    3. Fine-tune the model
    4. Save/load tuned model

    Example:
        >>> tuner = RepoTuner("/path/to/httpx")
        >>> tuner.collect_training_data(queries, llm, tokenizer)
        >>> tuner.fine_tune(epochs=5)
        >>> tuner.save("httpx_tuned.pt")
    """

    def __init__(
        self,
        repo_path: str,
        src_folder: Optional[str] = None,
        model_config: Optional[Dict] = None,
    ):
        """
        Initialize repo tuner.

        Args:
            repo_path: Path to the repository
            src_folder: Subfolder containing source code (auto-detected if None)
            model_config: Optional config for LearnedQueryModule
        """
        self.repo_path = repo_path
        self.repo_name = os.path.basename(repo_path)
        self.src_folder = src_folder

        # Load codebase
        self.codebase = self._load_codebase()
        self.file_paths = list(self.codebase.keys())

        print(f"Loaded {len(self.file_paths)} files from {self.repo_name}")

        # Create model
        from .module import create_learned_query_module

        config = model_config or {}
        config.setdefault('scoring_method', 'hybrid')

        self.model = create_learned_query_module(config)
        self.model.set_available_files(self.file_paths, self.codebase)

        # Training data
        self.training_data: Optional[RepoTrainingData] = None

        # Labeler for automatic labeling
        self.labeler = HeuristicLabeler(self.file_paths, self.codebase)

    def _load_codebase(self) -> Dict[str, str]:
        """Load all Python files from the codebase."""
        documents = {}

        # Auto-detect source folder
        if self.src_folder:
            src_path = os.path.join(self.repo_path, self.src_folder)
        else:
            # Try common patterns
            for folder in [self.repo_name, 'src', 'lib', '.']:
                candidate = os.path.join(self.repo_path, folder)
                if os.path.isdir(candidate):
                    src_path = candidate
                    break
            else:
                src_path = self.repo_path

        for root, dirs, files in os.walk(src_path):
            # Skip test/hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')
                      and 'test' not in d.lower()
                      and d != '__pycache__']

            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, self.repo_path)

                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        if len(content) > 100:  # Skip tiny files
                            documents[rel_path] = content
                    except Exception as e:
                        print(f"Error reading {rel_path}: {e}")

        return documents

    def _filter_quality_examples(
        self,
        examples: List[CCESpikeExample],
        min_tokens: int = 2,
        min_files: int = 1,
        verbose: bool = True,
    ) -> List[CCESpikeExample]:
        """Filter out low-quality training examples."""
        filtered = []
        reasons = {'no_tokens': 0, 'no_files': 0, 'low_quality': 0}

        for ex in examples:
            # Must have enough tokens
            if len(ex.confused_tokens) < min_tokens:
                reasons['no_tokens'] += 1
                continue

            # Must have labeled files
            if len(ex.relevant_files) < min_files:
                reasons['no_files'] += 1
                continue

            # Check token quality - at least one token should have decent probability
            max_prob = max(ex.confused_probs) if ex.confused_probs else 0
            if max_prob < 0.01:
                reasons['low_quality'] += 1
                continue

            filtered.append(ex)

        if verbose and len(examples) > len(filtered):
            removed = len(examples) - len(filtered)
            print(f"  Filtered {removed} low-quality examples:")
            for reason, count in reasons.items():
                if count > 0:
                    print(f"    - {reason}: {count}")

        return filtered

    def collect_training_data(
        self,
        sample_queries: List[str],
        llm,
        tokenizer,
        max_tokens_per_query: int = 100,
        cce_threshold: float = 3.0,
        min_token_prob: float = 0.01,
        verbose: bool = True,
    ) -> RepoTrainingData:
        """
        Collect training data from real CCE spikes.

        Args:
            sample_queries: List of 15-20 sample queries about the codebase
            llm: HuggingFace language model
            tokenizer: HuggingFace tokenizer
            max_tokens_per_query: Max tokens to generate per query
            cce_threshold: CCE threshold for spike detection
            min_token_prob: Minimum probability for tokens to be considered
            verbose: Print progress

        Returns:
            RepoTrainingData with collected examples
        """
        start_time = time.time()

        if verbose:
            print(f"\nCollecting training data from {len(sample_queries)} queries...")

        # Create collector with quality filtering
        collector = CCEDataCollector(
            tokenizer=tokenizer,
            cce_threshold=cce_threshold,
            min_token_prob=min_token_prob,
        )

        all_examples = []

        for i, query in enumerate(sample_queries):
            if verbose:
                print(f"  [{i+1}/{len(sample_queries)}] {query[:50]}...")

            try:
                examples = collector.collect_from_generation(
                    llm=llm,
                    query=query,
                    file_labeler=self.labeler,
                    max_tokens=max_tokens_per_query,
                )
                all_examples.extend(examples)

                if verbose and examples:
                    print(f"    Collected {len(examples)} CCE spikes")

            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")

        # Filter low-quality examples
        if verbose:
            print(f"\nFiltering quality...")
        filtered_examples = self._filter_quality_examples(all_examples, verbose=verbose)

        elapsed = time.time() - start_time

        self.training_data = RepoTrainingData(
            repo_name=self.repo_name,
            repo_path=self.repo_path,
            file_paths=self.file_paths,
            examples=filtered_examples,
            collection_time=elapsed,
        )

        if verbose:
            print(f"\nCollected {len(filtered_examples)} quality examples in {elapsed:.1f}s")
            print(f"  (filtered from {len(all_examples)} raw examples)")

        return self.training_data

    def fine_tune(
        self,
        epochs: int = 10,
        learning_rate: float = 5e-4,
        batch_size: int = 1,
        freeze_encoders: bool = True,
        warmup_epochs: int = 2,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the model on collected training data.

        Args:
            epochs: Number of training epochs (default 10 for better convergence)
            learning_rate: Learning rate (default 5e-4 for faster learning)
            batch_size: Batch size (usually 1 for simplicity)
            freeze_encoders: Whether to freeze SentenceTransformer encoders
            warmup_epochs: Number of warmup epochs with lower LR
            verbose: Print progress

        Returns:
            Training history dict
        """
        if self.training_data is None or len(self.training_data) == 0:
            raise ValueError("No training data. Call collect_training_data first.")

        if verbose:
            print(f"\nFine-tuning on {len(self.training_data)} examples for {epochs} epochs...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.train()

        # Freeze encoders (they're already good from pre-training)
        if freeze_encoders:
            for param in self.model.token_encoder._encoder.parameters():
                param.requires_grad = False
            for param in self.model.context_encoder._encoder.parameters():
                param.requires_grad = False

        # Collect trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in trainable_params)
            print(f"  Trainable: {trainable:,} / {total_params:,} parameters")

        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return max(0.1, 1.0 - (epoch - warmup_epochs) / (epochs - warmup_epochs))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Loss function with stronger ranking signal
        from .loss import QueryPoolerLoss
        loss_fn = QueryPoolerLoss(
            ranking_weight=1.5,  # Increased for better file ranking
            contrastive_weight=0.5,
            attention_reg_weight=0.05,  # Reduced to allow more flexibility
        )

        # Build file index
        file_to_idx = {f: i for i, f in enumerate(self.file_paths)}

        # Get file embeddings
        if self.model.scoring_method == 'hybrid':
            file_embs = self.model.hybrid_scorer._file_embeddings.to(device)
        else:
            file_embs = self.model._file_embeddings.to(device)

        history = {'loss': [], 'epoch_loss': [], 'lr': []}
        start_time = time.time()
        best_loss = float('inf')

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_samples = 0

            # Shuffle examples each epoch
            import random
            shuffled_examples = list(self.training_data.examples)
            random.shuffle(shuffled_examples)

            for example in shuffled_examples:
                optimizer.zero_grad()

                # Forward pass
                result = self.model(
                    confused_tokens=example.confused_tokens,
                    confused_probs=example.confused_probs,
                    original_query=example.original_query,
                    generated_context=example.generated_context,
                    return_attention=True,
                )

                # Get relevant file indices
                relevant_indices = [
                    file_to_idx[f] for f in example.relevant_files
                    if f in file_to_idx
                ]

                if not relevant_indices:
                    continue

                # Compute loss
                if self.model.scoring_method == 'hybrid':
                    scores = self.model.hybrid_scorer.forward(
                        query_emb=None,
                        query_tokens=example.confused_tokens,
                        query_text=example.original_query,
                    )
                else:
                    scores = self.model.scorer(result.query_embedding, file_embs)

                attn = {
                    'token_attention': result.token_attention,
                    'context_attention': result.context_attention,
                }

                loss, loss_components = loss_fn(
                    result.query_embedding.unsqueeze(0) if result.query_embedding.dim() == 1 else result.query_embedding,
                    file_embs.unsqueeze(0),
                    scores.unsqueeze(0) if scores.dim() == 1 else scores,
                    relevant_indices,
                    attn,
                )

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_samples += 1
                history['loss'].append(loss.item())

            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)

            avg_loss = epoch_loss / max(n_samples, 1)
            history['epoch_loss'].append(avg_loss)

            # Track best loss
            if avg_loss < best_loss:
                best_loss = avg_loss

            if verbose:
                print(f"  Epoch {epoch+1}/{epochs}: loss = {avg_loss:.4f}, lr = {current_lr:.6f}")

        elapsed = time.time() - start_time

        if verbose:
            print(f"\nFine-tuning complete in {elapsed:.1f}s")
            print(f"  Best loss: {best_loss:.4f}")

        # Mark model as fine-tuned so it uses learned embeddings for scoring
        self.model._is_fine_tuned = True
        self.model.eval()
        return history

    def save(self, path: str, include_training_data: bool = True):
        """
        Save the fine-tuned model and optionally training data.

        Args:
            path: Path to save (e.g., "httpx_tuned.pt")
            include_training_data: Whether to save training data alongside
        """
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'repo_name': self.repo_name,
            'repo_path': self.repo_path,
            'file_paths': self.file_paths,
            'scoring_method': self.model.scoring_method,
            'is_fine_tuned': self.model._is_fine_tuned,
        }, path)

        print(f"Saved model to {path}")

        # Save training data
        if include_training_data and self.training_data:
            data_path = path.replace('.pt', '_data.json')
            self.training_data.save(data_path)
            print(f"Saved training data to {data_path}")

    def load(self, path: str):
        """
        Load a previously fine-tuned model.

        Args:
            path: Path to saved model
        """
        checkpoint = torch.load(path, map_location='cpu')

        # Load model state (compatible weights only)
        model_dict = self.model.state_dict()
        compatible = {
            k: v for k, v in checkpoint['model_state_dict'].items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(compatible)
        self.model.load_state_dict(model_dict, strict=False)

        print(f"Loaded {len(compatible)}/{len(checkpoint['model_state_dict'])} weights from {path}")

        # Restore fine-tuned flag
        self.model._is_fine_tuned = checkpoint.get('is_fine_tuned', True)
        if self.model._is_fine_tuned:
            print("Model marked as fine-tuned (will use learned embeddings)")

        # Load training data if exists
        data_path = path.replace('.pt', '_data.json')
        if os.path.exists(data_path):
            self.training_data = RepoTrainingData.load(data_path)
            print(f"Loaded {len(self.training_data)} training examples")

    def evaluate(
        self,
        test_queries: List[Dict],
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the model on test queries.

        Args:
            test_queries: List of dicts with 'query', 'tokens', 'ground_truth' keys
            verbose: Print per-query results

        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()

        results = []

        for q in test_queries:
            result = self.model(
                confused_tokens=q.get('tokens', []),
                original_query=q['query'],
            )

            predicted = result.matched_files
            ground_truth = q['ground_truth']

            # Compute F1
            pred_set = set(predicted)
            gt_set = set(ground_truth)
            hits = len(pred_set & gt_set)

            precision = hits / len(pred_set) if pred_set else 0
            recall = hits / len(gt_set) if gt_set else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results.append({
                'query': q['query'],
                'predicted': predicted,
                'ground_truth': ground_truth,
                'f1': f1,
            })

            if verbose:
                print(f"\n{q['query'][:50]}...")
                print(f"  GT: {ground_truth}")
                print(f"  Pred: {predicted}")
                print(f"  F1: {f1:.2f}")

        avg_f1 = np.mean([r['f1'] for r in results])

        if verbose:
            print(f"\n{'='*50}")
            print(f"Average F1: {avg_f1:.3f}")

        return {
            'avg_f1': avg_f1,
            'results': results,
        }


# Convenience function
def setup_repo(
    repo_path: str,
    sample_queries: List[str],
    llm,
    tokenizer,
    save_path: Optional[str] = None,
    epochs: int = 5,
) -> RepoTuner:
    """
    One-line function to set up a repo for learned query.

    Args:
        repo_path: Path to repository
        sample_queries: 15-20 sample queries about the codebase
        llm: Language model
        tokenizer: Tokenizer
        save_path: Optional path to save tuned model
        epochs: Number of fine-tuning epochs

    Returns:
        Configured and tuned RepoTuner
    """
    print(f"Setting up {repo_path}...")
    print("="*60)

    # Initialize
    tuner = RepoTuner(repo_path)

    # Collect data
    tuner.collect_training_data(sample_queries, llm, tokenizer)

    # Fine-tune
    tuner.fine_tune(epochs=epochs)

    # Save
    if save_path:
        tuner.save(save_path)

    print("\nSetup complete!")
    return tuner
