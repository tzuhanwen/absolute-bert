import numpy as np
import torch
import torch.nn as nn

from .. import distances
from .mse import ComplexMse


class ComplexMultipletLoss(nn.Module):
    def __init__(
        self, model, sampling_word_size=10, margin=5, distance_metric="l2", **kwargs
    ) -> None:
        super().__init__()
        self.sampling_word_size = sampling_word_size
        self.distance_metric = distance_metric
        self.margin = margin
        self.model = model

    def __repr__(self) -> str:
        return f"multiplet_loss(margin={self.margin}, sampling_word_size={self.sampling_word_size})"

    def forward(self, predicts, targets):
        sampled_word_vecs = self.model.predictor.all_word_embeddings()[
            np.random.choice(self.model.vocab_size, size=self.sampling_word_size)
        ]

        pos_dist = distances.paired_distance(
            predicts, targets, metric=self.distance_metric
        )  # shape: [batch]
        sampled_dists = distances.pairwise_distance(
            targets, sampled_word_vecs, metric=self.distance_metric
        )
        triplets = (
            pos_dist[:, None] - sampled_dists + self.margin
        )  # shape: [batch, self.sampling_word_size]

        return {"multiplet_loss": torch.clamp(triplets, min=0).mean()}


class MseMultipletLoss(nn.Module):
    def __init__(
        self, model, sampling_word_size=10, margin=5, distance_metric="l2", **kwargs
    ) -> None:
        super().__init__()
        self.multiplet = ComplexMultipletLoss(model, sampling_word_size, margin, distance_metric)
        self.mse = ComplexMse()

    def __repr__(self) -> str:
        return f"{str(self.multiplet)}+{str(self.mse)}"

    def forward(self, predicts, targets):
        return {**self.multiplet(predicts, targets), **self.mse(predicts, targets)}


class CosMultipletLoss(nn.Module):
    def __init__(self, model, sampling_word_size=10, margin=5, eps=1e-8, **kwargs) -> None:
        super().__init__()
        self.multiplet = ComplexMultipletLoss(
            model, sampling_word_size, margin, distance_metric="cos"
        )
        self.eps = eps

    def __repr__(self) -> str:
        return f"{str(self.multiplet)}+cos_dist"

    def forward(self, predicts, targets):
        return {
            **self.multiplet(predicts, targets),
            "cos": distances.paired_distance(predicts, targets, metric="cos", eps=self.eps).mean(),
        }
