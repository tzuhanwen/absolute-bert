from typing import Any
from dataclasses import dataclass

from absolute_bert.base_types import Config, LanguageModelConfig

from ..registry import lm_config_registry, LanguageModelType


# @dataclass
# class RoformerConfig:


@lm_config_registry.register(LanguageModelType.ROFORMER)
@dataclass
class RoformerConfig(LanguageModelConfig):
    vocab_size: int
    dim: int = 512
    num_heads: int = 8
    activation_dim: int = 3 * 512
    depth: int = 8
    max_position_embeddings: int = 3072

    def to_transformer_config_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.dim,
            "num_hidden_layers": self.depth,
            "num_attention_heads": self.num_heads,
            "intermediate_size": self.activation_dim,
            "max_position_embeddings": self.max_position_embeddings,
        }
