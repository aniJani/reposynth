"""
Training Utilities for Learned Query Pooler.

This module provides:
- TrainingSample dataclass for training data
- DataLoader utilities
- Training loop
- Evaluation metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
from pathlib import Path

from .module import LearnedQueryModule
from .loss import QueryPoolerLoss


@dataclass
class TrainingSample:
    """
    Training sample for the learned query module.

    Collected from CCE spike data during generation.
    """
    # CCE spike data
    confused_tokens: List[str]          # Tokens from CCE spikes
    confused_probs: List[float]         # Token probabilities

    # Context
    original_query: str                 # User's question
    generated_context: str              # Text generated so far

    # Ground truth
    relevant_files: List[str]           # Files that should be retrieved
    irrelevant_files: List[str] = field(default_factory=list)  # Known irrelevant files

    # Metadata
    sample_id: str = ""
    spike_position: int = 0
    uncertainty_value: float = 0.0


class QueryPoolerDataset(Dataset):
    """
    Dataset for training the learned query module.
    """

    def __init__(
        self,
        samples: List[TrainingSample],
        all_files: List[str],
    ):
        """
        Initialize dataset.

        Args:
            samples: List of training samples
            all_files: All available file paths (for negative sampling)
        """
        self.samples = samples
        self.all_files = all_files
        self.file_to_idx = {f: i for i, f in enumerate(all_files)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Get relevant file indices
        relevant_indices = [
            self.file_to_idx[f]
            for f in sample.relevant_files
            if f in self.file_to_idx
        ]

        return {
            'confused_tokens': sample.confused_tokens,
            'confused_probs': sample.confused_probs,
            'original_query': sample.original_query,
            'generated_context': sample.generated_context,
            'relevant_indices': relevant_indices,
            'sample_id': sample.sample_id,
        }


def collate_samples(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.

    Since samples have variable-length token lists, we don't batch
    at the token level - each sample is processed individually.
    """
    return {
        'batch': batch,  # List of individual samples
        'batch_size': len(batch),
    }


class QueryPoolerTrainer:
    """
    Trainer for the learned query module.
    """

    def __init__(
        self,
        module: LearnedQueryModule,
        loss_fn: Optional[QueryPoolerLoss] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = 'cpu',
    ):
        """
        Initialize trainer.

        Args:
            module: LearnedQueryModule to train
            loss_fn: Loss function (default: QueryPoolerLoss)
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            device: Device to train on
        """
        self.module = module.to(device)
        self.loss_fn = loss_fn or QueryPoolerLoss()
        self.device = device

        # Optimizer (only learnable parameters)
        trainable_params = [
            p for p in module.parameters() if p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6,
        )

        # Training history
        self.history: List[Dict[str, float]] = []

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Dict with average loss components
        """
        self.module.train()

        epoch_losses = {
            'total': 0.0,
            'ranking': 0.0,
            'contrastive': 0.0,
            'attention_reg': 0.0,
        }
        num_samples = 0

        for batch in dataloader:
            batch_loss = torch.tensor(0.0, device=self.device)

            for sample in batch['batch']:
                # Forward pass
                result = self.module(
                    confused_tokens=sample['confused_tokens'],
                    confused_probs=sample['confused_probs'],
                    original_query=sample['original_query'],
                    generated_context=sample['generated_context'],
                    return_attention=True,
                )

                # Get file embeddings
                file_embs = self.module._file_embeddings
                if file_embs is None:
                    continue

                file_embs = file_embs.to(self.device)

                # Score files
                scores = self.module.scorer(
                    result.query_embedding,
                    file_embs,
                )

                # Compute loss
                attention_weights = {
                    'token_attention': result.token_attention,
                    'context_attention': result.context_attention,
                }

                loss, loss_components = self.loss_fn(
                    query_emb=result.query_embedding,
                    file_embs=file_embs,
                    file_scores=scores,
                    relevant_indices=sample['relevant_indices'],
                    attention_weights=attention_weights,
                )

                batch_loss = batch_loss + loss

                # Accumulate losses
                for key in epoch_losses:
                    epoch_losses[key] += loss_components[key]
                num_samples += 1

            # Backward pass
            if batch_loss.requires_grad:
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.module.parameters(), max_norm=1.0
                )
                self.optimizer.step()

        # Average losses
        if num_samples > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_samples

        self.scheduler.step()
        self.history.append(epoch_losses)

        return epoch_losses

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate the module.

        Args:
            dataloader: Validation data loader

        Returns:
            Dict with evaluation metrics
        """
        self.module.eval()

        metrics = {
            'precision_at_1': 0.0,
            'precision_at_3': 0.0,
            'recall_at_3': 0.0,
            'ndcg_at_3': 0.0,
            'mrr': 0.0,
        }
        num_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                for sample in batch['batch']:
                    result = self.module(
                        confused_tokens=sample['confused_tokens'],
                        confused_probs=sample['confused_probs'],
                        original_query=sample['original_query'],
                        generated_context=sample['generated_context'],
                    )

                    # Compute metrics
                    relevant_set = set(sample['relevant_indices'])
                    if not relevant_set:
                        continue

                    # Get predicted ranking
                    file_paths = self.module._file_paths
                    predicted_files = result.matched_files[:3]
                    predicted_indices = [
                        self.module.file_cache._file_paths.index(f)
                        for f in predicted_files
                        if f in self.module.file_cache._file_paths
                    ]

                    # Precision@1
                    if predicted_indices:
                        metrics['precision_at_1'] += (
                            1.0 if predicted_indices[0] in relevant_set else 0.0
                        )

                    # Precision@3
                    hits_at_3 = sum(
                        1 for idx in predicted_indices[:3]
                        if idx in relevant_set
                    )
                    metrics['precision_at_3'] += hits_at_3 / min(3, len(relevant_set))

                    # Recall@3
                    metrics['recall_at_3'] += hits_at_3 / len(relevant_set)

                    # MRR
                    for rank, idx in enumerate(predicted_indices, 1):
                        if idx in relevant_set:
                            metrics['mrr'] += 1.0 / rank
                            break

                    # NDCG@3
                    dcg = 0.0
                    for rank, idx in enumerate(predicted_indices[:3], 1):
                        if idx in relevant_set:
                            dcg += 1.0 / np.log2(rank + 1)

                    ideal_dcg = sum(
                        1.0 / np.log2(i + 2)
                        for i in range(min(3, len(relevant_set)))
                    )
                    if ideal_dcg > 0:
                        metrics['ndcg_at_3'] += dcg / ideal_dcg

                    num_samples += 1

        # Average metrics
        if num_samples > 0:
            for key in metrics:
                metrics[key] /= num_samples

        return metrics

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Optional evaluation metrics
        """
        checkpoint = {
            'epoch': epoch,
            'module_state_dict': self.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'metrics': metrics,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.module.load_state_dict(checkpoint['module_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', [])

        return checkpoint.get('epoch', 0)


def load_training_samples(
    json_path: str,
) -> List[TrainingSample]:
    """
    Load training samples from JSON file.

    Expected format:
    [
        {
            "confused_tokens": ["auth", "jwt"],
            "confused_probs": [0.3, 0.25],
            "original_query": "How does auth work?",
            "generated_context": "In our app...",
            "relevant_files": ["src/auth.py"],
            "sample_id": "sample_001"
        },
        ...
    ]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = TrainingSample(
            confused_tokens=item['confused_tokens'],
            confused_probs=item.get('confused_probs', [1.0] * len(item['confused_tokens'])),
            original_query=item.get('original_query', ''),
            generated_context=item.get('generated_context', ''),
            relevant_files=item['relevant_files'],
            irrelevant_files=item.get('irrelevant_files', []),
            sample_id=item.get('sample_id', ''),
            spike_position=item.get('spike_position', 0),
            uncertainty_value=item.get('uncertainty_value', 0.0),
        )
        samples.append(sample)

    return samples


def create_training_dataloader(
    samples: List[TrainingSample],
    all_files: List[str],
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader for training.

    Args:
        samples: Training samples
        all_files: All available file paths
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        DataLoader instance
    """
    dataset = QueryPoolerDataset(samples, all_files)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_samples,
    )
