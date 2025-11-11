from contextlib import contextmanager
from collections.abc import Generator

import torch
from torch import nn
from torch.types import Device
from torch.optim import Optimizer

class UpdateRatioTracker:

    def __init__(self, model: nn.Module, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer

    @contextmanager
    def track(
        self, device: Device = "cpu"
    ) -> Generator[None, None, None]:

        for p in self.model.parameters():
            if p.requires_grad:
                self.optimizer.state[p]['prev'] = p.data.detach().to(device, copy=True)

        yield

        # after optimizer.step()
        # calculate Δθ / θ = update_ratio
        d2 = torch.tensor(0., device=device)
        p2 = torch.tensor(0., device=device)
        for p in self.model.parameters():
            if p.requires_grad:
                prev = self.optimizer.state[p]['prev']
                now = p.data.detach().to(device)
                d2 += (now - prev).float().pow(2).sum()
                p2 += prev.float().pow(2).sum()

        update_ratio = (d2.sqrt() / (p2.sqrt() + 1e-12)).item()
        self.ratio = update_ratio

