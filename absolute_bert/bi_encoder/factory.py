from collections.abc import Callable, Sequence
from typing import Self, TypeAlias

import torch.nn as nn
from torch.types import Device
from transformers import PreTrainedTokenizerBase

from absolute_bert.base_types import EncoderInputs, Encoder, Embeddable, Sequences, States

from .bases import SemiSiameseBiEncoder
from .encoder_methods import EncoderPool, PoolMethodType

BertLikeTokenizeFn: TypeAlias = Callable[[Sequence[str]], States]


class _EncoderTokenizer:

    def __init__(self, tokenizer: PreTrainedTokenizerBase, device: Device) -> None:
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, texts: Sequences) -> EncoderInputs:
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, return_special_tokens_mask=True
        )
        return inputs.to(self.device)


class _EncoderMixin:

    @classmethod
    def _pure_text_token_vectors(cls, output: tuple[States], inputs: EncoderInputs) -> States:  
        """special_token and padding_token will map to zero vector."""
        if inputs.special_tokens_mask is None:
            raise AttributeError
        mask = (inputs.attention_mask * (1 - inputs.special_tokens_mask)).bool()
        return output.masked_fill(~mask[:, :, None], 0.0)

    @classmethod
    def _get_tokenize_fn(
        cls, tokenizer: PreTrainedTokenizerBase, device: Device
    ) -> BertLikeTokenizeFn:
        return _EncoderTokenizer(tokenizer, device)


class EncoderOutputPool(SemiSiameseBiEncoder[States], _EncoderMixin):

    @classmethod
    def from_model(
        cls,
        model: Encoder,
        tokenzier: PreTrainedTokenizerBase,
        device: Device,
        pool_method: PoolMethodType = "mean",
        using_corpus_part: str = "text",
    ) -> Self:

        model_output_method = EncoderPool(
            model.to(device),
            tokenize_fn=cls._get_tokenize_fn(tokenzier, device),
            common_post_fn=cls._pure_text_token_vectors,
            pool_method=pool_method,
        )

        return cls(model_output_method, using_corpus_part=using_corpus_part)


class EncoderEmbeddingPool(SemiSiameseBiEncoder[States], _EncoderMixin):

    @classmethod
    def from_model(
        cls,
        model: Encoder,
        tokenzier: PreTrainedTokenizerBase,
        device: Device,
        pool_method: PoolMethodType = "mean",
        using_corpus_part: str = "text",
    ) -> Self:

        static_embeddings_method = EncoderPool(
            _EncoderEmbedder(model).to(device),
            tokenize_fn=cls._get_tokenize_fn(tokenzier, device),
            common_post_fn=cls._pure_text_token_vectors,
            pool_method=pool_method,
        )

        return cls(static_embeddings_method, using_corpus_part)


class _EncoderEmbedder(nn.Module):
    def __init__(self, model: Embeddable):
        super().__init__()
        self.model = model

    def forward(self, inputs: EncoderInputs) -> States:
        return self.model.embed(inputs.input_ids)
