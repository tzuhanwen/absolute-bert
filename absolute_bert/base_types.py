import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields, asdict
from typing import Annotated, Any, Generic, Protocol, Self, TypeAlias, TypeVar, get_args, overload

import dacite
from dacite.exceptions import MissingValueError
from jaxtyping import Float, Int
from torch import Tensor, nn

logger = logging.getLogger(__name__)

_Tokens: TypeAlias = list[list[str]]
Sequences: TypeAlias = Int[Tensor, "B T"]
Labels: TypeAlias = Sequences
States: TypeAlias = Float[Tensor, "B T D"]
Projection: TypeAlias = Float[Tensor, "Dh D"]
Hiddens: TypeAlias = Float[Tensor, "B T H Dh"]
WordBiases: TypeAlias = Float[Tensor, "V"]
WordEmbeddings: TypeAlias = Float[Tensor, "V D"]

Encodings: TypeAlias = Mapping[str, Tensor]
NestedMetricDict: TypeAlias = dict[str, dict[int, float]]

EncoderOutputT = TypeVar("EncoderOutputT")


class Tokens:
    @classmethod
    def __class_getitem__(cls, shape: str) -> Any:
        return Annotated[_Tokens, {"shape": shape}]


@dataclass
class _ConfigBase:

    @classmethod
    def from_dict(cls, d: dict[Any, Any], type_hooks: dict[Any, Any] | None = None) -> Self:
        try:
            if type_hooks is None:
                return dacite.from_dict(cls, d)
            return dacite.from_dict(cls, d, config=dacite.Config(type_hooks=type_hooks))

        except MissingValueError as e:
            # missing_attr = ".".join(e.field_path)
            raise ValueError(f"missing {e.field_path}") from e

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Config(_ConfigBase):
    pass


LanguageModelConfig: TypeAlias = Config

configT = TypeVar("configT", bound=Config)


@dataclass
class ConfigUnresolved(_ConfigBase, Generic[configT]):

    @property
    def _resolve_target(self) -> type[configT]:
        orig_bases = getattr(self, "__orig_bases__", None)
        if orig_bases is None:

            raise TypeError(
                "Missing generic type info, "
                f"use `{self.__class__.__name__}(ConfigUnresolved[T])` on subclassing "
                "(_resolve_target only available on direct inherited subclass)"
            )
        target_class = get_args(orig_bases[0])[0]
        return target_class

    def resolve(self, **kwargs: Any) -> configT:
        logger.debug(f"resolving configs `{self.__class__.__name__}`, {type(self)=}, {kwargs=}")

        overrides = self._collect_overrides()
        return self._resolve_target(**(overrides | kwargs))

    def _collect_overrides(self) -> dict[Any, Any]:
        overrides = {}
        for f in fields(self):
            config_entry = getattr(self, f.name)
            if config_entry is not None:
                overrides[f.name] = config_entry

        return overrides


@dataclass
class EncoderInputsBase:
    # @classmethod
    # def from_mapping(cls, batch: EncoderInputsType) -> Self:
    #     return cls(
    #         batch["input_ids"],
    #         batch["attention_mask"],
    #         batch.get("labels"),
    #         batch.get("special_tokens_mask"),
    #     )

    @classmethod
    def from_mapping(cls, batch: Encodings) -> Self:
        return dacite.from_dict(cls, batch)


@dataclass
class EncoderInputsWithLabels(EncoderInputsBase):
    input_ids: Sequences
    attention_mask: Sequences
    labels: Sequences
    special_tokens_mask: Sequences | None = None


@dataclass
class EncoderInputs(EncoderInputsBase):
    input_ids: Sequences
    attention_mask: Sequences
    labels: Sequences | None = None
    special_tokens_mask: Sequences | None = None


class Encoder(Protocol[EncoderOutputT]):

    def train(self, mode: bool = True) -> Self: ...

    def eval(self) -> None: ...

    def forward(self, inputs: EncoderInputsBase) -> EncoderOutputT: ...

    def __call__(self, inputs: EncoderInputsBase) -> EncoderOutputT: ...


class LanguageModelingCapable(Protocol):

    def forward(self, *args: Any, **kwargs: Any) -> tuple[States, Labels | None]: ...

    @property
    def word_embeddings(self) -> WordEmbeddings: ...

    @property
    def word_biases(self) -> WordBiases: ...


class LanguageModel(Encoder[tuple[States, Sequences | None]], LanguageModelingCapable, Protocol):

    @overload
    def __call__(self, inputs: EncoderInputsWithLabels) -> tuple[States, Labels]: ...
    @overload
    def __call__(self, inputs: EncoderInputs) -> tuple[States, Labels | None]: ...
    def __call__(self, inputs: EncoderInputsBase) -> tuple[States, Labels | None]: ...


class Embeddable(Protocol):

    @property
    def embed(self) -> nn.Embedding: ...


class SequenceEncoder(Encoder[States], Embeddable, Protocol):
    pass
