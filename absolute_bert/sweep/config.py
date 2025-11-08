import logging
from dataclasses import dataclass, field, fields
from typing import Literal, Any, Self, Sequence

from absolute_bert.base_types import Config, ConfigUnresolved, LanguageModelConfig, _ConfigBase
from absolute_bert.extractor import ExtractingModuleRule
from absolute_bert.loss import LossForLMConfig
from absolute_bert.model import LanguageModelType, lm_config_registry

from .optimizer import OptimizerConfig
from .scheduler import SchedulerConfig

logger = logging.getLogger(__name__)


@dataclass
class WandbSession(Config):
    entity: str
    project: str
    job_type: Literal["train", "sweep"]
    # tags: Sequence[str] = ()
    # group: str | None = None


@dataclass
class TrainingArgs(Config):
    model_type: LanguageModelType 
    num_epochs: int = 1
    masking_probability: float = 0.15
    max_length: int = 256
    random_seed: int = 42
    val_ratio: float = 0.1
    max_steps: int = -1
    clip_loss: float = 50


@dataclass
class BatchSizes(Config):
    train: int
    val: int
    ir: int


@dataclass(frozen=True)
class Logging(Config):
    every_n_steps: int


@dataclass
class ParamLogging(Config):
    every_n_steps: int = 500
    rules: Sequence[ExtractingModuleRule] = ()


@dataclass
class LoggingConfig(Config):
    params: ParamLogging = field(default_factory=ParamLogging)
    train: Logging = field(default=Logging(10))
    val: Logging = field(default=Logging(500))
    ir: Logging = field(default=Logging(2000))


@dataclass
class SchedulerUnresolved(ConfigUnresolved[SchedulerConfig]):
    type: Literal["cosine", "linear"] | None = None
    warmup_ratio: float | None = None


class LanguageModelUnresolved(ConfigUnresolved[LanguageModelConfig]):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def resolve(self, model_type: LanguageModelType, vocab_size: int) -> LanguageModelConfig:
        logger.debug(f"start of LanguageModelUnresolved.resolve, {model_type=}, {lm_config_registry=}")
        return lm_config_registry[model_type](**self.kwargs | {"vocab_size": vocab_size})


@dataclass
class ExperimentConfig(Config):
    wandb: WandbSession
    model: LanguageModelConfig
    train: TrainingArgs
    batch_sizes: BatchSizes
    loss: LossForLMConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    logging: LoggingConfig


@dataclass
class ExperimentUnresolved(_ConfigBase):
    model: LanguageModelUnresolved = field(default_factory=LanguageModelUnresolved)
    wandb: WandbSession = field(default_factory=WandbSession)
    train: TrainingArgs = field(default_factory=TrainingArgs)
    batch_sizes: BatchSizes = field(default_factory=BatchSizes)
    loss: LossForLMConfig = field(default_factory=LossForLMConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerUnresolved = field(default_factory=SchedulerUnresolved)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, d: dict[Any, Any]) -> Self:
        type_hooks = {LanguageModelType: LanguageModelType}
        return super().from_dict(d, type_hooks=type_hooks)

    def resolve(self, vocab_size: int) -> ExperimentConfig:
        configs = {}
        for f in fields(self):
            config = getattr(self, f.name)

            if isinstance(config, Config):
                configs[f.name] = config
                continue

            if isinstance(config, SchedulerUnresolved):
                configs[f.name] = config.resolve(num_epochs=self.train.num_epochs)
            elif isinstance(config, LanguageModelUnresolved):
                configs[f.name] = config.resolve(self.train.model_type, vocab_size)

        return ExperimentConfig(**configs)
