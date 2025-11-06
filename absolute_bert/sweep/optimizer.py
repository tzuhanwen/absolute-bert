from torch import nn
from typing import TypedDict, Iterable
from torch import nn, optim

from dataclasses import dataclass
from absolute_bert.base_types import Config


@dataclass
class OptimizerConfig(Config):
    lr: float = 1e-4
    no_decay: tuple[str, ...] = ("bias", "LayerNorm.weight")
    weight_decay: float = 0.01


class ParamGroupConfig(TypedDict):
    params: list[nn.Parameter]
    weight_decay: float


def _make_parameter_groups(model: nn.Module, config: OptimizerConfig) -> list[ParamGroupConfig]:

    param_groups = {
        "decay": [],
        "no_decay": [],
    }
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        group = "no_decay" if any(nd in n for nd in config.no_decay) else "decay"
        param_groups[group].append(p)

    return [
        {"params": param_groups["decay"], "weight_decay": config.weight_decay},
        {"params": param_groups["no_decay"], "weight_decay": 0.0},
    ]


def make_adamw(model: nn.Module, config: OptimizerConfig):

    optimizer_grouped_parameters = _make_parameter_groups(model, config)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.lr)
    return optimizer
