from collections.abc import Callable, Sequence
from typing import Generic, Literal, TypeAlias, TypeVar
from enum import StrEnum
import torch
from jaxtyping import Float

from absolute_bert.base_types import (
    EncoderInputs,
    Encoder,
    Encodings,
    EncoderOutputT,
    States,
)

from .bases import SemiSiameseBiEncodeMethod

import logging
logger = logging.getLogger(__name__)

EncodingsT = TypeVar("EncodingsT", bound=Encodings)
AggregateMethod: TypeAlias = Callable[[States], Float[torch.Tensor, "b D"]]


class PoolMethodType(StrEnum):
    MEAN = "mean"
    SUM = "sum"


class EncoderPool(SemiSiameseBiEncodeMethod, Generic[EncodingsT, EncoderOutputT]):
    """actually siamese"""

    pool_methods: dict[PoolMethodType, AggregateMethod] = {
        PoolMethodType.MEAN: lambda x: torch.mean(x, dim=-2),
        PoolMethodType.SUM: lambda x: torch.sum(x, dim=-2),
    }

    def __init__(
        self,
        model: Encoder[EncoderOutputT],
        tokenize_fn: Callable[[Sequence[str]], EncodingsT],
        common_post_fn: Callable[[EncoderOutputT, EncodingsT], States],
        pool_method: PoolMethodType,
    ) -> None:
        self.model = model
        self.tokenize_fn = tokenize_fn
        self.common_post_fn = common_post_fn
        self.pool_method = pool_method

    def common_base(self, texts: Sequence[str]) -> States:
        model_inputs = self.tokenize_fn(texts)

        with torch.no_grad():
            inputs = EncoderInputs.from_mapping(model_inputs)
            result = self.model(inputs)
            
            if self.common_post_fn is not None:
                result = self.common_post_fn(result, model_inputs)

        return result

    def query_aggregate_fn(self, embeddings: States) -> Float[torch.Tensor, "b D"]:
        return self._aggregate_fn(embeddings)

    def corpus_aggregate_fn(self, embeddings: States) -> Float[torch.Tensor, "b D"]:
        return self._aggregate_fn(embeddings)

    def _aggregate_fn(self, embeddings: States) -> Float[torch.Tensor, "b D"]:
        return self.pool_methods[self.pool_method](embeddings)
