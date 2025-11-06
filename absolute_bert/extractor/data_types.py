from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray


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
