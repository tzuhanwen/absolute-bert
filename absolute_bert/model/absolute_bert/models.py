from dataclasses import dataclass

from jaxtyping import Float, Int
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn


from absolute_bert.base_types import (
    Hiddens,
    Labels,
    States,
    WordBiases,
    WordEmbeddings,
    EncoderInputs,
)
from .config import (
    AbsoluteAttentionConfig,
    AbsoluteBertConfig,
    AbsoluteBertLayerConfig,
    ActivationLayerConfig,
)
from ..registry import LanguageModelType, lm_registry


class AbsoluteAttention(nn.Module):
    def __init__(self, config: AbsoluteAttentionConfig) -> None:
        super().__init__()

        self.config = config
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads

        # self.time_embedding = time_embedding
        self.time_angle: Float[Tensor, "H Dt"] = nn.Parameter(
            torch.rand(config.num_heads, config.time_dim) ** 10 + 1e-8
        )
        self.head_time_delta: Float[Tensor, "H"] = nn.Parameter(torch.rand(self.num_heads))

        self.Q = nn.Linear(config.dim, config.num_heads * config.hidden_dim, bias=False)
        self.K = nn.Linear(config.dim, config.num_heads * config.hidden_dim)
        self.V = nn.Linear(config.dim, config.num_heads * config.hidden_dim)
        self.O = nn.Linear(config.num_heads * config.hidden_dim, config.dim)

        self.q_bias: Float[Tensor, "H_Dh"] = nn.Parameter(
            torch.zeros(config.num_heads * config.hidden_dim)
        )
        self.k_temperature = config.k_temperature

        with torch.no_grad():
            self.V.bias.copy_(torch.zeros_like(self.V.bias))
            self.O.bias.copy_(torch.zeros_like(self.O.bias))

        self.dropout = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(config.dim)

    def forward(self, states: States, attention_mask: Int[Tensor, "B T"]) -> States:
        batch_length = states.shape[:2]

        q_attentioned: Float[Tensor, "B T T H"] = self._get_q_attentioned(states, attention_mask)
        kv: Hiddens = self._get_kv(states, attention_mask)
        
        loading: Hiddens = torch.einsum("btlh,blhd->bthd", q_attentioned, kv)
        loading_flat: Float[Tensor, "B T H_Dh"] = loading.reshape(*batch_length, self.dim)

        attention_output: States = self.O(loading_flat)
        intermediate: States = self.layer_norm(states + self.dropout(attention_output))

        return intermediate

    def _get_q_attentioned(self, states: States, attention_mask: Int[Tensor, "B T"]) -> Float[Tensor, "B T T H"]:
        batch_length = states.shape[:2]

        q_flat: Float[Tensor, "B T H_Dh"] = self.Q(states) - self.q_bias.exp()
        q: Hiddens = q_flat.view(*batch_length, self.num_heads, self.hidden_dim)
        # q = (q + attention_mask[..., None, None])
        q = (q.sigmoid() / self.hidden_dim) * attention_mask[..., None, None]

        time: Float[Tensor, "T H 2*Dt"] = self._get_time(batch_length[1], True)
        q_timed: Float[Tensor, "B T H 2*Dt"] = q.sum(-1, keepdim=True) * time

        k_time: Float[Tensor, "T H 2*Dt"] = self._get_time(batch_length[1], False)
        q_attentioned: Float[Tensor, "B T T H"] = torch.einsum("bthd,lhd->btlh", q_timed, k_time)

        return q_attentioned

    def _get_time(self, length: int, with_time_delta: bool) -> Float[Tensor, "T H 2*Dt"]:
        word_positions: Int[Tensor, "T"] = torch.arange(length).to(self.head_time_delta.device)

        time_delta: Float[Tensor, "T 1 1"] = word_positions[:, None, None]
        if with_time_delta:
            time_delta: Float[Tensor, "T H 1"]  = time_delta + self.head_time_delta[None, :, None]

        time_angles: Float[Tensor, "T H Dt"] = time_delta * self.time_angle

        cosines, sines = time_angles.cos(), time_angles.sin()
        time: Float[Tensor, "T H 2*Dt"] = torch.cat(
            [cosines + sines, cosines - sines], dim=-1
        ) / np.sqrt(self.hidden_dim)

        return time
    
    def _get_kv(self, states: States, attention_mask: Int[Tensor, "B T"]) -> Hiddens:
        batch_length = states.shape[:2]

        k_flat: Float[Tensor, "B T H_Dh"] = self.K(states)
        k: Hiddens = k_flat.view(*batch_length, self.num_heads, self.hidden_dim)
        k_masked = (k / self.k_temperature) * attention_mask[..., None, None]
        k_softmax: Hiddens = k_masked.softmax(dim=-1)

        v_flat: Float[Tensor, "B T H_Dh"] = self.V(states)
        v: Hiddens = v_flat.view(*batch_length, self.num_heads, self.hidden_dim)

        kv: Hiddens = k_softmax * v

        return kv


class ActivationLayer(nn.Module):
    def __init__(self, config: ActivationLayerConfig) -> None:
        super().__init__()
        self.config = config

        self.linear_in = nn.Linear(config.dim, config.hidden_dim)
        self.act_fn = nn.GELU(approximate="tanh")
        self.linear_out = nn.Linear(config.hidden_dim, config.dim)
        self.layer_norm = nn.LayerNorm(config.dim)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, states: States) -> States:
        activated: Float[Tensor, "B T Da"] = self.act_fn(self.linear_in(states))
        lin_output: States = self.dropout(self.linear_out(activated))

        return self.layer_norm(states + lin_output)


class AbsoluteBertLayer(nn.Module):
    def __init__(
        self,
        config: AbsoluteBertLayerConfig,
    ) -> None:
        super().__init__()
        self.attention = AbsoluteAttention(config.get_attention_config())
        self.activation = ActivationLayer(config.get_activation_config())

    def forward(self, states: States, attention_mask: Int[Tensor, "B T"]) -> States:
        return self.activation(self.attention(states, attention_mask))


class AbsoluteBert(nn.Module):
    def __init__(
        self,
        config: AbsoluteBertConfig,
    ) -> None:
        super().__init__()
        self.config = config

        weight = torch.nn.init.xavier_normal_(torch.ones([config.vocab_size, config.dim]))
        self.embedding = nn.Embedding(config.vocab_size, config.dim, _weight=weight)
        # self.embedding = nn.Embedding(
        #     config.vocab_size,
        #     config.dim,
        #     _weight=torch.rand(config.vocab_size, config.dim) / np.sqrt(config.dim)
        # )

        self.layers = nn.ModuleList(
            [AbsoluteBertLayer(layer_config) for layer_config in config.iter_layer_configs()]
        )

    def forward(self, inputs: EncoderInputs) -> States:
        # extended_attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        # extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        states = self.embedding(inputs.input_ids)

        for layer in self.layers:
            states = layer(states, inputs.attention_mask)

        return states
    
    @property
    def embed(self) -> nn.Embedding:
        return self.embedding


@lm_registry.register(LanguageModelType.ABSOLUTE_BERT)
class AbsoluteBertLM(nn.Module):
    def __init__(self, config: AbsoluteBertConfig) -> None:
        super().__init__()
        self.base_model = AbsoluteBert(config)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, inputs: EncoderInputs) -> tuple[States, Labels | None]:
        states = self.base_model(inputs)
        return states, inputs.labels

    @property
    def embed(self) -> nn.Embedding:
        return self.base_model.embedding

    @property
    def word_embeddings(self) -> WordEmbeddings:
        return self.base_model.embedding.weight

    @property
    def word_biases(self) -> WordBiases:
        return self.bias
