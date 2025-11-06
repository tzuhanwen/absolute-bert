import fnmatch
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

from torch.nn import Module, Parameter
from torch.types import Number

from ._module_name_resolver import ModuleNameResolver
from .data_types import HistogramData

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ExtractingModuleRule:
    include: Sequence[str] = ()
    exclude: Sequence[str] = ()
    log_norm: bool = True
    log_distribution: bool = True


class ParamStats(TypedDict):
    norm: dict[str, Number]
    dist: dict[str, HistogramData]


class ModuleParamStatsExtractor:
    """
    Extracts and logs model parameter statistics (e.g. norm, distribution)
    based on config-driven rules.

    Supports multiple inclusion/exclusion rules with glob-style pattern matching.
    Also supports suffix-based indexing via '[-1]' to reference the last module or parameter
    in repeated structures like encoder layers.

    Example config:
        [
            {
                "include": ["encoder.layer[-1].*"],
                "exclude": ["*.bias"],
                "log_distribution": True,
                "log_norm": True
            },
            {
                "include": ["embeddings.*"],
                "log_distribution": False,
                "log_norm": True
            }
        ]

    Usage:
        logger = ParamLogger(model)
        log_dict = logger.extract_stats(config["param_logging"])
        wandb.log(log_dict, step=global_step)
    """

    def __init__(self, model: Module, rules: Iterable[ExtractingModuleRule]):
        self.model = model
        self.named_modules = dict(model.named_modules())
        self.named_parameters = dict(model.named_parameters())
        self.rules = rules
        self.resolved_logging_name_type_pairs: dict[
            tuple[str, Literal["norm", "dist"]], Parameter
        ] = {}

        module_names = list(self.named_modules.keys())
        for rule in rules:
            include = [
                ModuleNameResolver.resolve_suffix_indices(r, module_names)
                for r in rule.include
            ]
            exclude = [
                ModuleNameResolver.resolve_suffix_indices(r, module_names)
                for r in rule.exclude
            ]
            log_distribution = rule.log_distribution
            log_norm = rule.log_norm

            logger.info(f"Rule solved, `{include=}`, `{exclude=}`.")

            for name, param in self.named_parameters.items():
                if param.numel() == 0:
                    continue
                if not self._match_name(name, include):
                    continue
                if self._match_name(name, exclude):
                    continue

                if log_distribution:
                    self.resolved_logging_name_type_pairs[(name, "dist")] = param
                if log_norm:
                    self.resolved_logging_name_type_pairs[(name, "norm")] = param

    def _match_name(self, name: str, patterns: list[str]) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in patterns)

    def extract_stats(self) -> ParamStats:
        stats = {}

        for (name, log_type), param in self.resolved_logging_name_type_pairs.items():

            if log_type == "dist":
                stats["dist"][name] = HistogramData.from_array(param.detach().cpu().numpy())

            elif log_type == "norm":
                stats["norm"][name] = param.detach().norm().item()

        return stats
