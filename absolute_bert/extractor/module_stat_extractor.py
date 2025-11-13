import fnmatch
import logging
from collections.abc import Iterable
from typing import TypeAlias
from torch.nn import Module, Parameter
from torch.types import Number

import numpy as np

from ._module_name_resolver import ModuleNameResolver
from .extractor_types import HistogramData, ModuleExtractingRule, ExtractionType, Extractor

ModuleName: TypeAlias = str


logger = logging.getLogger(__name__)


def get_histogram(array: np.typing.NDArray) -> HistogramData:
    logger.info(f"{array.min(), array.max()}")
    return HistogramData.from_array(array)


EXTRACTORS: dict[ExtractionType, Extractor] = {
    ExtractionType.MODULE_NORM: lambda param: param.detach().norm().item(),
    ExtractionType.PARAM_MEAN: lambda param: param.detach().mean().item(),
    # ExtractionType.PARAM_DISTRIBUTION: lambda param: HistogramData.from_array(param.detach().cpu().numpy()),
    ExtractionType.PARAM_DISTRIBUTION: lambda param: get_histogram(param.detach().cpu().numpy()),
    ExtractionType.LOG_PARAM_DISTRIBUTION: lambda param: HistogramData.from_array(np.log10(param.detach().cpu().numpy())),
    ExtractionType.NORM_DIST_ALONG_LAST_DIM: lambda param: HistogramData.from_array(param.detach().norm(dim=-1).cpu().numpy())
}


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

    def __init__(self, model: Module, rules: Iterable[ModuleExtractingRule]):
        self.model = model
        self.named_modules = dict(model.named_modules())
        self.named_parameters = dict(model.named_parameters())
        self.rules = rules
        self.resolved_logging_name_type_pairs: dict[
            tuple[ModuleName, ExtractionType], Parameter
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

            logger.info(f"Rule solved, `{include=}`, `{exclude=}`.")

            for name, param in self.named_parameters.items():
                if param.numel() == 0:
                    continue
                if not self._match_name(name, include):
                    continue
                if self._match_name(name, exclude):
                    continue

                for method in set(rule.methods):
                    key = (name, ExtractionType(method))
                    self.resolved_logging_name_type_pairs[key] = param

    def _match_name(self, name: str, patterns: list[str]) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in patterns)

    def extract_stats(self) -> dict[str, HistogramData | Number]:
        stats = {}

        for (module_name, log_type), param in self.resolved_logging_name_type_pairs.items():
            if log_type == ExtractionType.PARAM_DISTRIBUTION:
                logger.debug(f"logging `{log_type}/{module_name}`")
            stats[f"{log_type}/{module_name}"] = EXTRACTORS[log_type](param)

        return stats
