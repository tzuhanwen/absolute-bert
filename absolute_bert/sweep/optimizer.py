from typing import TypeAlias

from torch import nn, optim

from dataclasses import dataclass
from absolute_bert.base_types import Config

import logging
logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig(Config):
    lr: float = 1e-4
    no_decay: tuple[str, ...] = ("bias", "layer_norm.weight")
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.998)


ParamName: TypeAlias = str
NamedParam: TypeAlias = tuple[ParamName, nn.Parameter]

@dataclass
class ParamGroupConfig:
    named_params: list[NamedParam]
    weight_decay: float

    def __repr__(self) -> str:
        param_infos = ",".join([f"({name}: {param.shape})" for name, param in self.named_params])
        return f"<ParamGroupConfig: [{param_infos}]>"

    def to_optimizer_group(self) -> list[dict[str, nn.Parameter | float]]:
        return {
            "params": [param for _, param in self.named_params],
            "weight_decay": self.weight_decay,
        }


def _make_parameter_groups(model: nn.Module, config: OptimizerConfig) -> list[ParamGroupConfig]:

    param_groups: dict[str, list[NamedParam]] = {
        "decay": [],
        "no_decay": [],
    }
    for name, param in model.named_parameters():
        group = "no_decay" if any(nd in name for nd in config.no_decay) else "decay"
        param_groups[group].append((name, param))

    return [
        ParamGroupConfig(named_params=param_groups["decay"], weight_decay=config.weight_decay),
        ParamGroupConfig(named_params=param_groups["no_decay"], weight_decay=0.0),
    ]


def make_adamw(model: nn.Module, config: OptimizerConfig):

    param_group_config = _make_parameter_groups(model, config)
    logger.info(f"setting weight decay: {param_group_config}")
    
    optimizer = optim.AdamW(
        [group.to_optimizer_group() for group in param_group_config], 
        lr=config.lr,
        betas=config.betas
    )
    logger.info(f"{optimizer=}")
    
    return optimizer
