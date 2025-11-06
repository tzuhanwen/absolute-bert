from absolute_bert.base_types import LanguageModelingCapable, Config, States, Labels
from dataclasses import dataclass
from torch import nn, Tensor
from jaxtyping import Float
from typing import Protocol, TypeAlias

Loss: TypeAlias = Float[Tensor, "1"]


@dataclass
class LossForLMConfig(Config):
    pass


class LossForLM(Protocol):
    
    def forward(self, output: States, labels: Labels) -> Loss: ...

    def __call__(self, output: States, labels: Labels) -> Loss: ...