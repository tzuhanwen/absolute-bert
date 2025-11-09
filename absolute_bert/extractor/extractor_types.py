from collections.abc import Sequence, Iterable
from dataclasses import dataclass
from typing import Any, Self, TypeAlias, Callable
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from torch.types import Tensor, Number


@dataclass
class HistogramData:
    bins: Sequence[float]
    counts: Sequence[int]
    min: float
    max: float
    num_vals: int

    @classmethod
    def from_array(cls, array: NDArray[Any], bins: int = 50) -> Self:
        seq = array.reshape(-1)
        counts, bin_edges = np.histogram(seq, bins=bins)
        return HistogramData(
            bins=bin_edges.tolist(),
            counts=counts.tolist(),
            min=float(seq.min()),
            max=float(seq.max()),
            num_vals=seq.size,
        )


Statistic: TypeAlias = HistogramData | Number
Extractor: TypeAlias = Callable[[Tensor], Statistic]


class ExtractionType(StrEnum):
    MODULE_NORM = "module_norm"
    PARAM_MEAN = "param_mean"
    PARAM_DISTRIBUTION = "param_dist"
    NORM_DIST_ALONG_LAST_DIM = "norm_dist_along_last_dim"


@dataclass
class ModuleExtractingRule:
    include: Sequence[str] = ()
    exclude: Sequence[str] = ()
    methods: Sequence[ExtractionType] = ()