from enum import StrEnum
from typing import Protocol

from torch.types import Device
from transformers import PreTrainedTokenizerBase

from absolute_bert.base_types import LanguageModel, LanguageModelConfig
from absolute_bert.utils import Registry

import logging

logger = logging.getLogger(__name__)
logger.debug("start of model registry")


class LanguageModelType(StrEnum):
    ABSOLUTE_BERT = "absolute_bert"
    ROFORMER = "roformer"


class SummaryGenerator(Protocol):
    
    def __init__(self, lm: LanguageModel, tokenizer: PreTrainedTokenizerBase) -> None:
        ...

    def __call__(
        self, device: Device = "cpu",
    ) -> dict[str, str]:
        ...


lm_registry = Registry[LanguageModelType, type[LanguageModel]]()

lm_config_registry = Registry[LanguageModelType, type[LanguageModelConfig]]()

lm_summary_generator_registry = Registry[LanguageModelType, type[SummaryGenerator]]()


def ensure_model_registered():
    from . import _auto_import


ensure_model_registered()

logger.debug("end of model registry")
