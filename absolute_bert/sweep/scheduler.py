from dataclasses import dataclass
from typing import Literal
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from absolute_bert.base_types import Config

import logging
logger = logging.getLogger(__name__)

@dataclass
class SchedulerConfig(Config):
    num_epochs: int
    type: Literal["cosine", "linear"] = "cosine"
    warmup_ratio: float = 0.1
    # step_size: int | None = None
    # gamma: float | None = None
    # T_max: int | None = None


GETTERS = {
    "cosine": get_cosine_schedule_with_warmup,
    "linear": get_linear_schedule_with_warmup,
}


def make_scheduler(optimizer, num_batches: int, config: SchedulerConfig) -> _LRScheduler:
    logger.debug(f"make_scheduler, {optimizer=}, {num_batches=}, {config=}")

    scheduler = GETTERS[config.type](
        optimizer,
        num_warmup_steps=num_batches * config.num_epochs * config.warmup_ratio,
        num_training_steps=num_batches * config.num_epochs,
    )

    return scheduler
