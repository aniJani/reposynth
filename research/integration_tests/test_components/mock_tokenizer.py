"""
Mock Tokenizer for Week 7 Integration Tests.

Simulates HuggingFace tokenizer behavior for testing without actual model.
"""

from typing import List, Dict, Optional
import hashlib
import re


class MockTokenizer:
    """
    Simulates HuggingFace tokenizer behavior.

    Features:
    - Realistic token-to-IDs mapping
    - Special tokens handling (</s>, <s>, etc.)
    - Token counting
    - Approximate subword tokenization
    """

    # Special tokens
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    # Special token IDs
    BOS_ID = 1
    EOS_ID = 2
    PAD_ID = 0
    UNK_ID = 3

    def __init__(self, vocab_size: int = 32000):
        """
        Initialize with simulated vocabulary.

        Args:
            vocab_size: Vocabulary size (CodeLlama: 32K)
        """
        self.vocab_size = vocab_size
        self.seed = 42

        # Build simulated vocabulary
        self._build_vocabulary()

    def _build_vocabulary(self):
        """Build a realistic simulated vocabulary."""
        import random

        random.seed(self.seed)

        # Start with special tokens
        self.vocab = {
            self.PAD_TOKEN: self.PAD_ID,
            self.BOS_TOKEN: self.BOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
            self.UNK_TOKEN: self.UNK_ID,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Add code keywords (first 100 IDs)
        code_keywords = [
            "def",
            "class",
            "import",
            "from",
            "return",
            "if",
            "else",
            "for",
            "while",
            "try",
            "except",
            "with",
            "as",
            "and",
            "or",
            "not",
            "True",
            "False",
            "None",
            "None",
            "async",
            "await",
            "yield",
            "raise",
            "assert",
            "pass",
            "break",
            "continue",
            "lambda",
            "global",
            "nonlocal",
            "del",
            "in",
            "is",
        ]

        for i, kw in enumerate(code_keywords):
            token_id = 4 + i
            self.vocab[kw] = token_id
            self.id_to_token[token_id] = kw

        # Add common code identifiers (next 500 IDs)
        common_identifiers = [
            "function",
            "variable",
            "param",
            "result",
            "data",
            "value",
            "response",
            "request",
            "config",
            "settings",
            "options",
            "handler",
            "callback",
            "error",
            "exception",
            "message",
            "string",
            "number",
            "boolean",
            "array",
            "object",
            "dict",
            "list",
            "tuple",
            "set",
            "file",
            "path",
            "name",
            "id",
            "type",
            "key",
            "value",
            "user",
            "auth",
            "token",
            "session",
            "model",
            "view",
            "controller",
            "service",
            "module",
            "package",
        ]

        for i, ident in enumerate(common_identifiers):
            token_id = 104 + i
            self.vocab[ident] = token_id
            self.id_to_token[token_id] = ident

        # Add library/framework names (next 100 IDs)
        library_names = [
            "pandas",
            "numpy",
            "torch",
            "tensorflow",
            "keras",
            "sklearn",
            "React",
            "Vue",
            "Angular",
            "jQuery",
            "axios",
            "lodash",
            "flask",
            "Django",
            "FastAPI",
            "Express",
            "Koa",
            "Firebase",
            "Auth",
            "Firestore",
            "Database",
            "SQLAlchemy",
            "pytest",
            "jest",
            "unittest",
            "mocha",
            "chai",
        ]

        for i, lib in enumerate(library_names):
            token_id = 604 + i
            self.vocab[lib] = token_id
            self.id_to_token[token_id] = lib

        # Add API/method names (next 500 IDs)
        api_names = [
            "read_csv",
            "read_json",
            "read_parquet",
            "DataFrame",
            "useState",
            "useEffect",
            "useCallback",
            "useMemo",
            "signInWithEmailAndPassword",
            "createUserWithEmailAndPassword",
            "signOut",
            "collection",
            "doc",
            "add",
            "update",
            "delete",
            "get",
            "get",
            "post",
            "put",
            "delete",
            "route",
            "router",
            "Depends",
            "HTTPException",
            "BackgroundTasks",
        ]

        for i, api in enumerate(api_names):
            token_id = 704 + i
            self.vocab[api] = token_id
            self.id_to_token[token_id] = api

        # Fill remaining IDs with random tokens
        next_id = 1204
        while next_id < self.vocab_size:
            # Generate random subword-like tokens
            token_len = random.randint(2, 6)
            token = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=token_len))
            token = "##" + token  # BPE-like marker

            if token not in self.vocab:
                self.vocab[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1
            else:
                next_id += 1

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        if not text:
            return []

        tokens = []
        remaining = text

        while remaining:
            # Try to find longest matching token
            longest_match = None
            longest_match_len = 0

            # Check common multi-word tokens first
            for token in sorted(self.vocab.keys(), key=len, reverse=True):
                if remaining.startswith(token):
                    if len(token) > longest_match_len:
                        longest_match = token
                        longest_match_len = len(token)

            if longest_match:
                tokens.append(self.vocab[longest_match])
                remaining = remaining[longest_match_len:]
            else:
                # No match, use UNK
                tokens.append(self.UNK_ID)
                remaining = remaining[1:]

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        text_parts = []

        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.UNK_TOKEN)

            # Clean up BPE markers
            token = token.replace("##", "")

            # Handle special tokens
            if token in [self.EOS_TOKEN, self.BOS_TOKEN, self.PAD_TOKEN]:
                # Add space between special tokens
                if text_parts and text_parts[-1] != " ":
                    text_parts.append(" ")
            else:
                text_parts.append(token)

        return "".join(text_parts)

    def __call__(self, text: str, return_tensors="pt"):
        """
        HuggingFace-compatible interface.

        Args:
            text: Input text
            return_tensors: Format to return ('pt' for PyTorch)

        Returns:
            Dict with 'input_ids' key
        """
        import torch

        token_ids = self.encode(text)

        if return_tensors == "pt":
            input_ids = torch.tensor([token_ids], dtype=torch.long)
        else:
            input_ids = [token_ids]

        return {"input_ids": input_ids}

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.encode(text))

    @property
    def eos_token_id(self) -> int:
        """Return EOS token ID."""
        return self.EOS_ID

    @property
    def bos_token_id(self) -> int:
        """Return BOS token ID."""
        return self.BOS_ID
