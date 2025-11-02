from abc import abstractmethod, ABC
from torch import nn, Tensor
from jaxtyping import Float, Int
from typing import TypeAlias

Sequences: TypeAlias = Int[Tensor, "B T"]
States: TypeAlias = Float[Tensor, "B T D"]
Hiddens: TypeAlias = Float[Tensor, "B T H Dh"]
WordBiases: TypeAlias = Float[Tensor, "V"]
WordEmbeddings: TypeAlias = Float[Tensor, "V D"]


class ModelForMaskedLM(ABC):

    @abstractmethod
    def word_embeddings(self) -> WordEmbeddings: ...

    @abstractmethod
    def word_biases(self) -> WordBiases: ...
