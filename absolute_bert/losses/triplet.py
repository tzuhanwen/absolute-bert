import numpy as np
import torch
import torch.nn as nn

from .. import distances
from .mse import ComplexMse


class ComplexTripletLoss(nn.Module):
    def __init__(
        self, model, sampling_word_size=10, margin=5, distance_metric="l2", **kwargs
    ) -> None:
        super().__init__()
        self.sampling_word_size = sampling_word_size
        self.distance_metric = distance_metric
        self.margin = margin
        self.model = model

    def __repr__(self) -> str:
        return f"triplet_loss(margin={self.margin})"

    def forward(self, predicts, targets):
        sampled_word_vecs = self.model.predictor.all_word_embeddings()[
            np.random.choice(self.model.vocab_size, size=self.sampling_word_size)
        ]

        pos_dist = distances.paired_distance(predicts, targets, metric=self.distance_metric)
        sampled_dists = distances.pairwise_distance(
            targets, sampled_word_vecs, metric=self.distance_metric
        )
        neg_dist = sampled_dists.min(dim=1).values

        return {"triplet_loss": torch.clamp(pos_dist - neg_dist + self.margin, min=0).mean()}


class ComplexMseTripletLoss(nn.Module):
    def __init__(
        self, model, sampling_word_size=10, margin=5, distance_metric="l2", **kwargs
    ) -> None:
        super().__init__()
        self.triplet = ComplexTripletLoss(model, sampling_word_size, margin, distance_metric)
        self.mse = ComplexMse()

    def __repr__(self) -> str:
        return f"{str(self.triplet)}+{str(self.mse)}"

    def forward(self, predicts, targets):
        return {**self.triplet(predicts, targets), **self.mse(predicts, targets)}


class ComplexMseSquaredTripletLoss(nn.Module):
    def __init__(
        self, model, sampling_word_size=10, margin=5, distance_metric="l2", **kwargs
    ) -> None:
        super().__init__()
        self.triplet = ComplexTripletLoss(model, sampling_word_size, margin, distance_metric)
        self.mse = ComplexMse()

    def __repr__(self) -> str:
        return f"sq{str(self.triplet)}+{str(self.mse)}"

    def forward(self, predicts, targets):
        return {
            "sqtriplet_loss": self.triplet(predicts, targets)["triplet_loss"] ** 2,
            **self.mse(predicts, targets),
        }
