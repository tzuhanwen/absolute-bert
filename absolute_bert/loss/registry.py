from absolute_bert.utils import Registry
from enum import StrEnum
from typing import Any
from .loss_types import LossForLM, LossForLMConfig

class LossType(StrEnum):
    CROSS_ENTROPY = "cross_entropy"
    SAMPLED_SOFTMAX_CROSS_ENTROPY = "sampled_softmax_cross_entropy"


loss_registry = Registry[LossType, LossForLM]()
loss_config_registry = Registry[LossType, LossForLMConfig]()