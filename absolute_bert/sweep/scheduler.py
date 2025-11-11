from dataclasses import dataclass
from typing import Literal, Protocol, Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from absolute_bert.base_types import Config

import logging
logger = logging.getLogger(__name__)

@dataclass
class SchedulerConfig(Config):
    num_epochs: int
    type: Literal["cosine", "linear", "no_op"] = "cosine"
    warmup_ratio: float = 0.1
    # step_size: int | None = None
    # gamma: float | None = None
    # T_max: int | None = None


class OptimizerGetter(Protocol):

    def __call__(
        self, optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int
    ) -> LRScheduler:
        ...


def get_noop_scheduler(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int) -> LambdaLR:
    return LambdaLR(optimizer, lr_lambda=[lambda step: 1.0]*len(optimizer.param_groups))


GETTERS: dict[str, OptimizerGetter] = {
    "cosine": get_cosine_schedule_with_warmup,
    "linear": get_linear_schedule_with_warmup,
    "no_op": get_noop_scheduler,
}


def make_scheduler(optimizer, effective_num_batches: int, config: SchedulerConfig) -> LRScheduler:
    logger.debug(f"make_scheduler, {optimizer=}, {effective_num_batches=}, {config=}")

    scheduler = GETTERS[config.type](
        optimizer,
        num_warmup_steps=int(effective_num_batches * config.num_epochs * config.warmup_ratio),
        num_training_steps=effective_num_batches * config.num_epochs,
    )
    
    return scheduler
