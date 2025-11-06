from enum import StrEnum

from absolute_bert.base_types import LanguageModel, LanguageModelConfig
from absolute_bert.utils import Registry

import logging

logger = logging.getLogger(__name__)
logger.debug("start of model registry")

class LanguageModelType(StrEnum):
    ABSOLUTE_BERT = "absolute_bert"


lm_registry = Registry[LanguageModelType, type[LanguageModel]]()

lm_config_registry = Registry[LanguageModelType, type[LanguageModelConfig]]()


def ensure_model_registered():
    from . import _auto_import

ensure_model_registered()

logger.debug("end of model registry")