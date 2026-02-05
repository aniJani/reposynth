"""
Training Loss Functions for Learned Query Pooler.

This module provides loss functions for training the learned query module:
- Ranking Loss: Relevant files should score higher than irrelevant
- Contrastive Loss: Query embedding close to relevant file embeddings
- Attention Regularization: Encourage focused attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class QueryPoolerLoss(nn.Module):
    """
    Combined loss for training the learned query pooler.

    Loss Components:
    1. Ranking Loss (margin-based): Ensure relevant files score higher
    2. Contrastive Loss (InfoNCE): Pull query toward relevant file embeddings
    3. Attention Regularization: Encourage sparse, focused attention

    Example:
        >>> loss_fn = QueryPoolerLoss()
        >>> loss = loss_fn(
        ...     query_emb=query_embedding,
        ...     file_embs=file_embeddings,
        ...     file_scores=predicted_scores,
        ...     relevant_indices=[0, 2],  # indices of relevant files
        ...     attention_weights=attn_weights,
        ... )
    """

    def __init__(
        self,
        ranking_margin: float = 0.5,
        ranking_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        attention_reg_weight: float = 0.1,
        temperature: float = 0.07,
        hard_negative_weight: float = 2.0,
    ):
        """
        Initialize loss function.

        Args:
            ranking_margin: Margin for ranking loss
            ranking_weight: Weight for ranking loss component
            contrastive_weight: Weight for contrastive loss component
            attention_reg_weight: Weight for attention regularization
            temperature: Temperature for contrastive loss
            hard_negative_weight: Extra weight for hard negatives
        """
        super().__init__()
        self.ranking_margin = ranking_margin
        self.ranking_weight = ranking_weight
        self.contrastive_weight = contrastive_weight
        self.attention_reg_weight = attention_reg_weight
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        query_emb: torch.Tensor,
        file_embs: torch.Tensor,
        file_scores: torch.Tensor,
        relevant_indices: List[int],
        attention_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            query_emb: Query embedding [D] or [B, D]
            file_embs: File embeddings [N, D] or [B, N, D]
            file_scores: Predicted file scores [N] or [B, N]
            relevant_indices: Indices of relevant files (ground truth)
            attention_weights: Optional attention weight dict

        Returns:
            total_loss: Combined loss scalar
            loss_components: Dict with individual loss values
        """
        # Ensure batch dimensions
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        if file_embs.dim() == 2:
            file_embs = file_embs.unsqueeze(0)
        if file_scores.dim() == 1:
            file_scores = file_scores.unsqueeze(0)

        batch_size = query_emb.shape[0]
        num_files = file_embs.shape[1]

        # Build relevance mask
        relevance_mask = torch.zeros(batch_size, num_files, device=query_emb.device)
        for idx in relevant_indices:
            if idx < num_files:
                relevance_mask[:, idx] = 1.0

        # Compute individual losses
        ranking_loss = self._ranking_loss(file_scores, relevance_mask)
        contrastive_loss = self._contrastive_loss(
            query_emb, file_embs, relevance_mask
        )

        attention_loss = torch.tensor(0.0, device=query_emb.device)
        if attention_weights is not None and self.attention_reg_weight > 0:
            attention_loss = self._attention_regularization(attention_weights)

        # Combine losses
        total_loss = (
            self.ranking_weight * ranking_loss +
            self.contrastive_weight * contrastive_loss +
            self.attention_reg_weight * attention_loss
        )

        loss_components = {
            'total': total_loss.item(),
            'ranking': ranking_loss.item(),
            'contrastive': contrastive_loss.item(),
            'attention_reg': attention_loss.item(),
        }

        return total_loss, loss_components

    def _ranking_loss(
        self,
        scores: torch.Tensor,  # [B, N]
        relevance_mask: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        """
        Compute margin-based ranking loss.

        For each relevant file, it should score higher than all irrelevant
        files by at least the margin.
        """
        batch_size = scores.shape[0]
        num_files = scores.shape[1]

        # Get positive and negative indices
        pos_mask = relevance_mask.bool()
        neg_mask = ~pos_mask

        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=scores.device)

        total_loss = torch.tensor(0.0, device=scores.device)
        num_pairs = 0

        for b in range(batch_size):
            pos_indices = pos_mask[b].nonzero(as_tuple=True)[0]
            neg_indices = neg_mask[b].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            # Get positive and negative scores
            pos_scores = scores[b, pos_indices]  # [P]
            neg_scores = scores[b, neg_indices]  # [N]

            # Margin loss: max(0, margin + neg - pos)
            # For each positive, compare against all negatives
            for pos_score in pos_scores:
                # Find hard negatives (highest scoring negatives)
                hard_neg_scores = neg_scores.topk(
                    min(5, len(neg_scores))
                ).values

                # Standard margin loss with hard negatives
                margins = self.ranking_margin + hard_neg_scores - pos_score
                loss = F.relu(margins)

                # Weight hard negatives more
                weights = torch.ones_like(loss)
                weights[0] = self.hard_negative_weight  # Hardest negative

                total_loss = total_loss + (loss * weights).mean()
                num_pairs += 1

        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        return total_loss

    def _contrastive_loss(
        self,
        query_emb: torch.Tensor,  # [B, D]
        file_embs: torch.Tensor,  # [B, N, D]
        relevance_mask: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.

        Pull query embedding toward relevant file embeddings,
        push away from irrelevant file embeddings.
        """
        batch_size = query_emb.shape[0]

        # Normalize embeddings
        query_norm = F.normalize(query_emb, p=2, dim=-1)  # [B, D]
        file_norm = F.normalize(file_embs, p=2, dim=-1)  # [B, N, D]

        # Compute similarities
        similarities = torch.bmm(
            query_norm.unsqueeze(1),  # [B, 1, D]
            file_norm.transpose(1, 2),  # [B, D, N]
        ).squeeze(1)  # [B, N]

        # Scale by temperature
        logits = similarities / self.temperature

        # InfoNCE: log softmax over positives
        # numerator: sum of exp(positive similarities)
        # denominator: sum of exp(all similarities)
        pos_mask = relevance_mask.bool()

        if not pos_mask.any():
            return torch.tensor(0.0, device=query_emb.device)

        # Mask out negatives with large negative value
        pos_logits = logits.clone()
        pos_logits[~pos_mask] = float('-inf')

        # Log-sum-exp for positives
        log_sum_pos = torch.logsumexp(pos_logits, dim=-1)

        # Log-sum-exp for all
        log_sum_all = torch.logsumexp(logits, dim=-1)

        # InfoNCE loss
        loss = -(log_sum_pos - log_sum_all)

        return loss.mean()

    def _attention_regularization(
        self,
        attention_weights: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Regularize attention to be focused (low entropy).

        Encourages the model to attend to specific tokens rather than
        spreading attention uniformly.
        """
        total_entropy = torch.tensor(0.0)
        count = 0

        for key, attn in attention_weights.items():
            if attn is None:
                continue

            # attn shape: [B, Q, K] or [Q, K]
            if attn.dim() == 2:
                attn = attn.unsqueeze(0)

            # Compute entropy
            attn_clamped = attn.clamp(min=1e-8)
            entropy = -(attn_clamped * torch.log(attn_clamped)).sum(dim=-1)

            # Average entropy across queries and batch
            total_entropy = total_entropy + entropy.mean()
            count += 1

        if count > 0:
            total_entropy = total_entropy / count

        return total_entropy


class RankingLoss(nn.Module):
    """
    Standalone ranking loss for simpler training scenarios.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        scores: torch.Tensor,
        relevant_indices: List[int],
    ) -> torch.Tensor:
        """
        Compute ranking loss.

        Args:
            scores: Predicted scores [N] or [B, N]
            relevant_indices: Indices of relevant items

        Returns:
            Ranking loss scalar
        """
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)

        batch_size = scores.shape[0]
        num_items = scores.shape[1]

        # Build mask
        relevance_mask = torch.zeros_like(scores, dtype=torch.bool)
        for idx in relevant_indices:
            if idx < num_items:
                relevance_mask[:, idx] = True

        # Get positive and negative scores
        pos_scores = scores[relevance_mask]
        neg_scores = scores[~relevance_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=scores.device)

        # All pairs margin loss
        pos_scores = pos_scores.unsqueeze(1)  # [P, 1]
        neg_scores = neg_scores.unsqueeze(0)  # [1, N]

        margins = self.margin + neg_scores - pos_scores  # [P, N]
        loss = F.relu(margins).mean()

        return loss


class ContrastiveLoss(nn.Module):
    """
    Standalone contrastive loss (InfoNCE).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            anchor: Anchor embedding [D]
            positives: Positive embeddings [P, D]
            negatives: Negative embeddings [N, D]

        Returns:
            Contrastive loss scalar
        """
        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0)  # [1, D]

        # Normalize
        anchor = F.normalize(anchor, p=2, dim=-1)
        positives = F.normalize(positives, p=2, dim=-1)
        negatives = F.normalize(negatives, p=2, dim=-1)

        # Positive similarities
        pos_sim = torch.mm(anchor, positives.T) / self.temperature  # [1, P]

        # Negative similarities
        neg_sim = torch.mm(anchor, negatives.T) / self.temperature  # [1, N]

        # All similarities
        all_sim = torch.cat([pos_sim, neg_sim], dim=-1)  # [1, P+N]

        # Labels (first P are positives)
        # InfoNCE: log(sum(exp(pos))) - log(sum(exp(all)))
        log_sum_pos = torch.logsumexp(pos_sim, dim=-1)
        log_sum_all = torch.logsumexp(all_sim, dim=-1)

        loss = -(log_sum_pos - log_sum_all)

        return loss.mean()
