from enum import StrEnum
from absolute_bert.utils import Registry
from absolute_bert.base_types import LanguageModelConfig, LanguageModel

class LanguageModelType(StrEnum):
    ABSOLUTE_BERT = "absolute_bert"


lm_registry = Registry[LanguageModelType, LanguageModel]()

lm_config_registry = Registry[LanguageModelType, LanguageModelConfig]()