from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Generic, Protocol, TypeVar

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import trange

CommonBaseOutputT = TypeVar("CommonBaseOutputT")


class BiEncoder(Protocol):
    """Match the usage in DenseRetrievalExactSearch of beir package."""

    def encode_queries(
        self,
        queries: dict[str, str],
        batch_size: int,
        show_progress_bar: bool,
        convert_to_tensor: bool,
    ) -> Float[Tensor, "B D"]: ...

    def encode_corpus(
        self,
        corpus: Iterable[dict[str, str]],
        batch_size: int,
        show_progress_bar: bool,
        convert_to_tensor: bool,
    ) -> Float[Tensor, "B D"]: ...


class SemiSiameseBiEncodeMethod(ABC, Generic[CommonBaseOutputT]):

    @abstractmethod
    def common_base(self, texts: Iterable[str]) -> CommonBaseOutputT: ...

    @abstractmethod
    def query_aggregate_fn(
        self, common_base_output: CommonBaseOutputT, convert_to_tensor: bool = True
    ): ...

    @abstractmethod
    def corpus_aggregate_fn(
CommonBaseOutputT    ): ...


class SemiSiameseBiEncoder(BiEncoder, Generic[CommonBaseOutputT]):

    def __init__(
        self,
        bi_encode_method: SemiSiameseBiEncodeMethod[CommonBaseOutputT],
        using_corpus_part="text",
    ):
        self.bi_encode_method = bi_encode_method
        self.using_corpus_part = using_corpus_part

    def encode_queries(
        self,
        queries: dict[str, str],
        batch_size: int,
        show_progress_bar: bool,
        convert_to_tensor: bool,
    ):
        return self._encode(
            queries, self.bi_encode_method.query_aggregate_fn, batch_size, show_progress_bar
        )

    def encode_corpus(
        self,
        corpus: Iterable[dict[str, str]],
        batch_size: int,
        show_progress_bar: bool,
        convert_to_tensor: bool,
    ) -> Float[Tensor, "B D"]:
        return self._encode(
            [doc[self.using_corpus_part] for doc in corpus],
            self.bi_encode_method.corpus_aggregate_fn,
            batch_size,
            show_progress_bar,
        )

    def _encode(
        self,
        texts: Sequence[str],
        aggregate_fn: Callable[[CommonBaseOutputT], Float[Tensor, "b D"]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> Float[Tensor, "B D"]:
        results = []
        itr = self._get_batch_start_num_iterator(show_progress_bar)
        for batch_start_idx in itr(0, len(texts), batch_size):

            batch_end_idx = min(batch_start_idx + batch_size, len(texts))
            batch = texts[batch_start_idx:batch_end_idx]

            result = aggregate_fn(self.bi_encode_method.common_base(batch))
            results.append(result)
        return torch.cat(results)

    def _get_batch_start_num_iterator(
        self, show_progress_bar: bool
    ) -> Callable[[int, int, int], Iterable[int]]:
        if show_progress_bar:
            return partial(trange, leave=False)
        else:
            return range
