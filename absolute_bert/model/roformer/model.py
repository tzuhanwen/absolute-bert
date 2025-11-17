import torch
from transformers import RoFormerModel, RoFormerConfig
from torch import nn

from absolute_bert.base_types import EncoderInputs, States, Labels, WordEmbeddings, WordBiases
from ..registry import LanguageModelType, lm_registry
from .config import RoformerConfig


@lm_registry.register(LanguageModelType.ROFORMER)
class RoformerLM(nn.Module):
    def __init__(self, config: RoformerConfig) -> None:
        super().__init__()
        transformer_config = RoFormerConfig(**config.to_transformer_config_dict())
        self.base_model = RoFormerModel(transformer_config)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, inputs: EncoderInputs) -> tuple[States, Labels | None]:
        states = self.base_model(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
        ).last_hidden_state
        return states, inputs.labels

    @property
    def embed(self) -> nn.Embedding:
        return self.base_model.embeddings.word_embeddings

    @property
    def word_embeddings(self) -> WordEmbeddings:
        return self.base_model.embeddings.word_embeddings.weight

    @property
    def word_biases(self) -> WordBiases:
        return self.bias
