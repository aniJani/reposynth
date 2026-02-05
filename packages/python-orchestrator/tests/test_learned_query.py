"""
Unit Tests for Learned Query Pooling Module.

Tests for:
- ConfusedTokenEncoder
- ContextEncoder
- LearnedQueryPooler
- LearnedFileScorer
- LearnedQueryModule
- QueryPoolerLoss
- LearnedTopicInferrer (integration)
"""

import pytest
import numpy as np
import torch
from typing import List

# Skip if sentence-transformers not available
pytest.importorskip("sentence_transformers")


class TestConfusedTokenEncoder:
    """Tests for ConfusedTokenEncoder."""

    def test_encoder_initialization(self):
        """Test encoder initializes correctly."""
        from orchestrator.retrieval.learned_query.encoders import ConfusedTokenEncoder

        encoder = ConfusedTokenEncoder(embed_dim=384)
        assert encoder.embed_dim == 384
        assert encoder.max_tokens == 20

    def test_encode_tokens(self):
        """Test encoding a list of tokens."""
        from orchestrator.retrieval.learned_query.encoders import ConfusedTokenEncoder

        encoder = ConfusedTokenEncoder(embed_dim=384)
        tokens = ["auth", "jwt", "bearer", "token"]

        embeddings = encoder(tokens)

        assert embeddings.shape == (4, 384)
        assert embeddings.dtype == torch.float32

    def test_encode_with_probabilities(self):
        """Test encoding with probability weighting."""
        from orchestrator.retrieval.learned_query.encoders import ConfusedTokenEncoder

        encoder = ConfusedTokenEncoder(embed_dim=384, use_probability_weighting=True)
        tokens = ["auth", "jwt", "bearer"]
        probs = [0.3, 0.25, 0.2]

        embeddings, weights = encoder(tokens, probs, return_weights=True)

        assert embeddings.shape == (3, 384)
        assert weights.shape == (3,)

    def test_encode_empty_tokens(self):
        """Test encoding empty token list."""
        from orchestrator.retrieval.learned_query.encoders import ConfusedTokenEncoder

        encoder = ConfusedTokenEncoder(embed_dim=384)
        embeddings = encoder([])

        assert embeddings.shape == (1, 384)  # Returns placeholder

    def test_max_tokens_limit(self):
        """Test that max_tokens limit is respected."""
        from orchestrator.retrieval.learned_query.encoders import ConfusedTokenEncoder

        encoder = ConfusedTokenEncoder(embed_dim=384, max_tokens=5)
        tokens = ["a", "b", "c", "d", "e", "f", "g", "h"]

        embeddings = encoder(tokens)

        assert embeddings.shape == (5, 384)  # Limited to 5


class TestContextEncoder:
    """Tests for ContextEncoder."""

    def test_encoder_initialization(self):
        """Test encoder initializes correctly."""
        from orchestrator.retrieval.learned_query.encoders import ContextEncoder

        encoder = ContextEncoder(embed_dim=384)
        assert encoder.embed_dim == 384
        assert encoder.chunk_size == 128

    def test_encode_query_and_context(self):
        """Test encoding query and context together."""
        from orchestrator.retrieval.learned_query.encoders import ContextEncoder

        encoder = ContextEncoder(embed_dim=384)
        query = "How does authentication work?"
        context = "In our Flask app, authentication uses JWT tokens."

        embeddings = encoder(query, context)

        # Should have query + at least 1 context chunk
        assert embeddings.shape[0] >= 1
        assert embeddings.shape[1] == 384

    def test_encode_query_only(self):
        """Test encoding just the query."""
        from orchestrator.retrieval.learned_query.encoders import ContextEncoder

        encoder = ContextEncoder(embed_dim=384)
        query = "How does auth work?"

        embedding = encoder.encode_query_only(query)

        assert embedding.shape == (1, 384)

    def test_return_separate(self):
        """Test returning query and context separately."""
        from orchestrator.retrieval.learned_query.encoders import ContextEncoder

        encoder = ContextEncoder(embed_dim=384)
        query = "Test query"
        context = "Some context text here."

        query_emb, context_emb = encoder(query, context, return_separate=True)

        assert query_emb.shape == (1, 384)
        assert context_emb.shape[0] >= 1


class TestLearnedQueryPooler:
    """Tests for LearnedQueryPooler."""

    def test_pooler_initialization(self):
        """Test pooler initializes correctly."""
        from orchestrator.retrieval.learned_query.pooler import LearnedQueryPooler

        pooler = LearnedQueryPooler(
            embed_dim=384,
            num_query_vectors=4,
            num_heads=4,
        )

        assert pooler.embed_dim == 384
        assert pooler.num_query_vectors == 4
        assert pooler.query_vectors.shape == (4, 384)

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        from orchestrator.retrieval.learned_query.pooler import LearnedQueryPooler

        pooler = LearnedQueryPooler(embed_dim=384, num_query_vectors=4, num_heads=4)

        confused_embs = torch.randn(10, 384)  # 10 tokens
        context_embs = torch.randn(5, 384)    # 5 context chunks

        output, attn = pooler(confused_embs, context_embs, return_attention_weights=True)

        assert output.shape == (384,)
        assert 'token_attention' in attn
        assert 'context_attention' in attn
        assert attn['token_attention'].shape[2] == 10  # K dimension
        assert attn['context_attention'].shape[2] == 5  # L dimension

    def test_batched_forward(self):
        """Test forward pass with batch dimension."""
        from orchestrator.retrieval.learned_query.pooler import LearnedQueryPooler

        pooler = LearnedQueryPooler(embed_dim=384, num_query_vectors=4, num_heads=4)

        confused_embs = torch.randn(2, 10, 384)  # Batch of 2
        context_embs = torch.randn(2, 5, 384)

        output, _ = pooler(confused_embs, context_embs)

        assert output.shape == (2, 384)

    def test_fusion_methods(self):
        """Test different fusion methods."""
        from orchestrator.retrieval.learned_query.pooler import LearnedQueryPooler

        for method in ["concat", "add", "gated"]:
            pooler = LearnedQueryPooler(
                embed_dim=384,
                num_query_vectors=4,
                num_heads=4,
                fusion_method=method,
            )

            confused_embs = torch.randn(5, 384)
            context_embs = torch.randn(3, 384)

            output, _ = pooler(confused_embs, context_embs)

            assert output.shape == (384,)

    def test_attention_weights_stored(self):
        """Test that attention weights are stored for visualization."""
        from orchestrator.retrieval.learned_query.pooler import LearnedQueryPooler

        pooler = LearnedQueryPooler(embed_dim=384, num_query_vectors=4, num_heads=4)

        confused_embs = torch.randn(8, 384)
        context_embs = torch.randn(4, 384)

        pooler(confused_embs, context_embs)

        weights = pooler.get_attention_weights()
        assert weights['token_attention'] is not None
        assert weights['context_attention'] is not None


class TestLearnedFileScorer:
    """Tests for LearnedFileScorer."""

    def test_scorer_initialization(self):
        """Test scorer initializes correctly."""
        from orchestrator.retrieval.learned_query.scorer import LearnedFileScorer

        scorer = LearnedFileScorer(embed_dim=384, scoring_method="bilinear")
        assert scorer.embed_dim == 384
        assert scorer.scoring_method == "bilinear"

    def test_cosine_scoring(self):
        """Test cosine similarity scoring."""
        from orchestrator.retrieval.learned_query.scorer import LearnedFileScorer

        scorer = LearnedFileScorer(embed_dim=384, scoring_method="cosine")

        query_emb = torch.randn(384)
        file_embs = torch.randn(10, 384)

        scores = scorer(query_emb, file_embs)

        assert scores.shape == (10,)
        assert scores.min() >= -1.0
        assert scores.max() <= 1.0

    def test_bilinear_scoring(self):
        """Test bilinear scoring."""
        from orchestrator.retrieval.learned_query.scorer import LearnedFileScorer

        scorer = LearnedFileScorer(embed_dim=384, scoring_method="bilinear")

        query_emb = torch.randn(384)
        file_embs = torch.randn(10, 384)

        scores = scorer(query_emb, file_embs)

        assert scores.shape == (10,)

    def test_mlp_scoring(self):
        """Test MLP-based scoring."""
        from orchestrator.retrieval.learned_query.scorer import LearnedFileScorer

        scorer = LearnedFileScorer(embed_dim=384, scoring_method="mlp")

        query_emb = torch.randn(384)
        file_embs = torch.randn(10, 384)

        scores = scorer(query_emb, file_embs)

        assert scores.shape == (10,)

    def test_score_with_threshold(self):
        """Test scoring with threshold filtering."""
        from orchestrator.retrieval.learned_query.scorer import LearnedFileScorer

        scorer = LearnedFileScorer(embed_dim=384, scoring_method="cosine")

        query_emb = torch.randn(384)
        file_embs = torch.randn(5, 384)
        file_paths = ["a.py", "b.py", "c.py", "d.py", "e.py"]

        matched, scores_dict = scorer.score_with_threshold(
            query_emb, file_embs, file_paths,
            threshold=0.0,  # Low threshold to get some matches
            top_k=3,
        )

        assert isinstance(matched, list)
        assert len(matched) <= 3
        assert isinstance(scores_dict, dict)


class TestLearnedQueryModule:
    """Tests for LearnedQueryModule (main class)."""

    def test_module_initialization(self):
        """Test module initializes correctly."""
        from orchestrator.retrieval.learned_query.module import LearnedQueryModule

        module = LearnedQueryModule(
            embed_dim=384,
            num_query_vectors=4,
            num_heads=4,
        )

        assert module.embed_dim == 384

    def test_set_available_files(self):
        """Test setting available files."""
        from orchestrator.retrieval.learned_query.module import LearnedQueryModule

        module = LearnedQueryModule(embed_dim=384)
        files = ["src/auth.py", "src/api.py", "src/utils.py"]

        module.set_available_files(files)

        assert module._file_paths == files
        assert module._file_embeddings is not None
        assert module._file_embeddings.shape[0] == 3

    def test_forward_pass(self):
        """Test full forward pass."""
        from orchestrator.retrieval.learned_query.module import LearnedQueryModule

        module = LearnedQueryModule(embed_dim=384)
        module.set_available_files(["src/auth.py", "src/api.py"])

        result = module(
            confused_tokens=["auth", "jwt", "bearer"],
            confused_probs=[0.3, 0.25, 0.2],
            original_query="How does authentication work?",
            generated_context="In our Flask app, authentication uses",
        )

        assert result.query_embedding.shape == (384,)
        assert isinstance(result.matched_files, list)
        assert isinstance(result.file_scores, dict)
        assert 0.0 <= result.confidence <= 1.0

    def test_forward_empty_tokens(self):
        """Test forward with empty tokens."""
        from orchestrator.retrieval.learned_query.module import LearnedQueryModule

        module = LearnedQueryModule(embed_dim=384)
        result = module(confused_tokens=[], original_query="test")

        assert result.matched_files == []
        assert result.confidence == 0.0

    def test_explain(self):
        """Test explanation output."""
        from orchestrator.retrieval.learned_query.module import LearnedQueryModule

        module = LearnedQueryModule(embed_dim=384)
        module.set_available_files(["src/auth.py"])

        explanation = module.explain(
            confused_tokens=["auth", "token"],
            original_query="How does auth work?",
        )

        assert 'matched_files' in explanation
        assert 'confidence' in explanation
        assert 'token_contributions' in explanation


class TestQueryPoolerLoss:
    """Tests for QueryPoolerLoss."""

    def test_loss_initialization(self):
        """Test loss initializes correctly."""
        from orchestrator.retrieval.learned_query.loss import QueryPoolerLoss

        loss_fn = QueryPoolerLoss(
            ranking_weight=1.0,
            contrastive_weight=0.5,
            attention_reg_weight=0.1,
        )

        assert loss_fn.ranking_weight == 1.0
        assert loss_fn.contrastive_weight == 0.5

    def test_loss_computation(self):
        """Test loss computation produces valid output."""
        from orchestrator.retrieval.learned_query.loss import QueryPoolerLoss

        loss_fn = QueryPoolerLoss()

        query_emb = torch.randn(384)
        file_embs = torch.randn(10, 384)
        file_scores = torch.randn(10)
        relevant_indices = [0, 3]

        total_loss, components = loss_fn(
            query_emb=query_emb,
            file_embs=file_embs,
            file_scores=file_scores,
            relevant_indices=relevant_indices,
        )

        assert total_loss.dim() == 0  # Scalar
        assert 'total' in components
        assert 'ranking' in components
        assert 'contrastive' in components

    def test_loss_with_attention(self):
        """Test loss with attention regularization."""
        from orchestrator.retrieval.learned_query.loss import QueryPoolerLoss

        loss_fn = QueryPoolerLoss(attention_reg_weight=0.1)

        query_emb = torch.randn(384)
        file_embs = torch.randn(10, 384)
        file_scores = torch.randn(10)
        relevant_indices = [0, 2]

        attention_weights = {
            'token_attention': torch.softmax(torch.randn(1, 4, 8), dim=-1),
            'context_attention': torch.softmax(torch.randn(1, 4, 5), dim=-1),
        }

        total_loss, components = loss_fn(
            query_emb=query_emb,
            file_embs=file_embs,
            file_scores=file_scores,
            relevant_indices=relevant_indices,
            attention_weights=attention_weights,
        )

        assert components['attention_reg'] > 0


class TestLearnedTopicInferrer:
    """Tests for LearnedTopicInferrer (integration)."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        class MockTokenizer:
            def decode(self, ids):
                vocab = ["auth", "jwt", "bearer", "token", "user", "login"]
                if len(ids) == 1 and ids[0] < len(vocab):
                    return vocab[ids[0]]
                return "unknown"

        return MockTokenizer()

    def test_inferrer_initialization(self, mock_tokenizer):
        """Test inferrer initializes correctly."""
        from orchestrator.retrieval.learned_inferrer import LearnedTopicInferrer

        inferrer = LearnedTopicInferrer(
            tokenizer=mock_tokenizer,
            use_fallback=False,
        )

        assert inferrer.tokenizer is not None

    def test_set_available_files(self, mock_tokenizer):
        """Test setting available files."""
        from orchestrator.retrieval.learned_inferrer import LearnedTopicInferrer

        inferrer = LearnedTopicInferrer(
            tokenizer=mock_tokenizer,
            use_fallback=False,
        )

        files = ["src/auth.py", "src/api.py"]
        inferrer.set_available_files(files)

        assert inferrer.learned_module._file_paths == files

    def test_infer_topic(self, mock_tokenizer):
        """Test topic inference produces TopicResult."""
        from orchestrator.retrieval.learned_inferrer import LearnedTopicInferrer

        inferrer = LearnedTopicInferrer(
            tokenizer=mock_tokenizer,
            available_files=["src/auth.py"],
            use_fallback=False,
        )

        # Create mock logits with top predictions
        logits = np.zeros(100)
        logits[0] = 5.0  # "auth"
        logits[1] = 4.0  # "jwt"
        logits[2] = 3.0  # "bearer"

        result = inferrer.infer_topic(
            logits=logits,
            context="In our Flask app, authentication uses",
            position=42,
            original_query="How does auth work?",
        )

        assert result is not None
        assert hasattr(result, 'matched_files')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'query')

    def test_api_compatibility(self, mock_tokenizer):
        """Test API compatibility with TopicInferrer."""
        from orchestrator.retrieval.learned_inferrer import LearnedTopicInferrer
        from orchestrator.retrieval.topic_inference import TopicResult

        inferrer = LearnedTopicInferrer(
            tokenizer=mock_tokenizer,
            use_fallback=False,
        )

        logits = np.random.randn(100)

        result = inferrer.infer_topic(
            logits=logits,
            context="test context",
            position=0,
        )

        # Check it returns a TopicResult
        assert isinstance(result, TopicResult)
        assert hasattr(result, 'query')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'matched_files')
        assert hasattr(result, 'file_match_scores')

    def test_statistics(self, mock_tokenizer):
        """Test usage statistics tracking."""
        from orchestrator.retrieval.learned_inferrer import LearnedTopicInferrer

        inferrer = LearnedTopicInferrer(
            tokenizer=mock_tokenizer,
            use_fallback=False,
        )

        stats = inferrer.get_statistics()

        assert 'total_inferences' in stats
        assert 'learned_count' in stats
        assert 'fallback_count' in stats


class TestFileEmbeddingCache:
    """Tests for FileEmbeddingCache."""

    def test_cache_build(self):
        """Test building embedding cache."""
        from orchestrator.retrieval.learned_query.scorer import FileEmbeddingCache

        cache = FileEmbeddingCache()
        files = ["src/auth.py", "src/api/routes.py", "lib/utils.py"]

        embeddings = cache.build_cache(files)

        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension

    def test_cache_retrieval(self):
        """Test retrieving cached embeddings."""
        from orchestrator.retrieval.learned_query.scorer import FileEmbeddingCache

        cache = FileEmbeddingCache()
        files = ["a.py", "b.py", "c.py"]
        cache.build_cache(files)

        embeddings = cache.get_embeddings()

        assert embeddings.shape[0] == 3

    def test_cache_specific_files(self):
        """Test getting embeddings for specific files."""
        from orchestrator.retrieval.learned_query.scorer import FileEmbeddingCache

        cache = FileEmbeddingCache()
        files = ["a.py", "b.py", "c.py"]
        cache.build_cache(files)

        embeddings = cache.get_embeddings(["a.py", "c.py"])

        assert embeddings.shape[0] == 2


class TestTrainingComponents:
    """Tests for training utilities."""

    def test_training_sample_dataclass(self):
        """Test TrainingSample dataclass."""
        from orchestrator.retrieval.learned_query.training import TrainingSample

        sample = TrainingSample(
            confused_tokens=["auth", "jwt"],
            confused_probs=[0.3, 0.25],
            original_query="How does auth work?",
            generated_context="In our app...",
            relevant_files=["src/auth.py"],
        )

        assert len(sample.confused_tokens) == 2
        assert sample.relevant_files == ["src/auth.py"]

    def test_dataset_creation(self):
        """Test QueryPoolerDataset creation."""
        from orchestrator.retrieval.learned_query.training import (
            TrainingSample, QueryPoolerDataset
        )

        samples = [
            TrainingSample(
                confused_tokens=["auth"],
                confused_probs=[0.3],
                original_query="test",
                generated_context="",
                relevant_files=["a.py"],
            )
        ]
        all_files = ["a.py", "b.py", "c.py"]

        dataset = QueryPoolerDataset(samples, all_files)

        assert len(dataset) == 1

        item = dataset[0]
        assert 'confused_tokens' in item
        assert 'relevant_indices' in item


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
