from dataclasses import dataclass, field, fields
from typing import Literal, Self, Any, Iterable, Sequence
import dacite
from absolute_bert.base_types import Config, ConfigUnresolved, _ConfigBase
from .optimizer import OptimizerConfig
from .scheduler import SchedulerConfig
from absolute_bert.extractor import ExtractingModuleRule
from absolute_bert.loss import LossForLMConfig



@dataclass
class WandbSession(Config):
    entity: str
    project: str
    job_type: Literal["train", "sweep"]
    # tags: Sequence[str] = ()
    # group: str | None = None
    



@dataclass
class TrainingArgs(Config):
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
    

@dataclass(frozen=True)
class ParamLogging(Config):
    every_n_steps: int = 500
    rules: tuple[ExtractingModuleRule] = ()


@dataclass
class LoggingConfig(Config):
    params: ParamLogging = field(default=ParamLogging())
    train: Logging = field(default=Logging(10))
    val: Logging = field(default=Logging(500))
    ir: Logging = field(default=Logging(2000))




@dataclass
class SchedulerUnresolved(ConfigUnresolved[SchedulerConfig]):
    type: Literal["cosine", "linear"] | None = None
    warmup_ratio: int | None = None




@dataclass
class ExperimentConfig(Config):
    wandb: WandbSession
    train: TrainingArgs
    batch_sizes: BatchSizes
    loss: LossForLMConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    logging: LoggingConfig


# model overrides
#     embedding_initialize_method: 'rand'
#     attention_type: Absolute_global_attention
#     depth: 12
#     num_heads: 12
#     dim: 768
#     k_temperature: 0.5

@dataclass
class ExperimentUnresolved(_ConfigBase):
    wandb: WandbSession = field(default_factory=WandbSession)
    train: TrainingArgs = field(default_factory=TrainingArgs)
    batch_sizes: BatchSizes = field(default_factory=BatchSizes)
    loss: LossForLMConfig = field(default_factory=LossForLMConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerUnresolved = field(default_factory=SchedulerUnresolved)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def resolve(self) -> ExperimentConfig:
        configs = {}
        for f in fields(self):
            config = getattr(self, f.name)
            if isinstance(config, Config):
                configs[f.name] = config
            elif isinstance(config, ConfigUnresolved):
                configs[f.name] = config.resolve()

        return ExperimentConfig(**configs)