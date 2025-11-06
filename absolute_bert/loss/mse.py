import torch
import torch.nn as nn


class ComplexMse(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "mse"

    def forward(self, predicts, targets):
        diffs = predicts - targets
        # diff_square_norms = diffs * diffs.conj()
        diff_square_norms = diffs.real**2 + diffs.imag**2
        return {"mse": diff_square_norms.real.sum(dim=[1]).mean()}


class ComplexMseWithInverseNorm(ComplexMse):
    def __init__(self, relax=0.25, min_dist=1, **kwargs) -> None:
        super().__init__()
        self.relax = relax
        self.min_dist = min_dist

    def __repr__(self) -> str:
        return f"mse_with_inverse_norm(relax={self.relax}, min_dist={self.min_dist})"

    def forward(self, predicts, targets):
        inverse_norm = torch.clamp(
            1 / (predicts.norm(dim=-1) + self.relax) - 1 / (self.min_dist + self.relax), min=0
        )
        return {**super().forward(predicts, targets), "inverse_norm": inverse_norm.mean()}
