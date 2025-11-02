from collections.abc import Generator, Sequence
from dataclasses import dataclass

import torch


@dataclass
class AbsoluteAttentionConfig:
    dim: int
    num_heads: int
    hidden_dim: int
    time_dim: int
    k_temperature: float

    def __post_init__(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError

        # assert self.hidden_dim % 2 == 0


@dataclass
class ActivationLayerConfig:
    dim: int
    hidden_dim: int


@dataclass
class AbsoluteBertLayerConfig:
    dim: int
    num_heads: int
    hidden_dim: int
    time_dim: int
    k_temperature: float
    activation_dim: int

    def get_attention_config(self) -> AbsoluteAttentionConfig:
        return AbsoluteAttentionConfig(
            dim=self.dim,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            time_dim=self.time_dim,
            k_temperature=self.k_temperature,
        )

    def get_activation_config(self) -> ActivationLayerConfig:
        return ActivationLayerConfig(dim=self.dim, hidden_dim=self.activation_dim)


@dataclass
class AbsoluteBertConfig:
    vocab_size: int
    dim: int = 512
    num_heads: int = 8
    hidden_dim: int | None = None
    time_dim: int = 64
    activation_dim: int = 3 * 512
    depth: int = 8
    log_granularity: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6)
    k_temperature: float = 0.5
    dtype: torch.dtype = torch.float

    def __post_init__(self) -> None:
        if len(self.log_granularity) != self.depth:
            raise AttributeError("log_granularity is about setting hidden_dim of each layer")

    def iter_layer_configs(self) -> Generator[AbsoluteBertLayerConfig, None, None]:
        for gran in self.log_granularity:
            yield AbsoluteBertLayerConfig(
                self.dim,
                num_heads=self.dim // (2**gran),
                hidden_dim=2**gran,
                time_dim=self.time_dim,
                k_temperature=self.k_temperature,
                activation_dim=self.activation_dim,
            )
