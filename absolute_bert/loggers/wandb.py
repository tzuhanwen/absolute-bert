from typing import Iterable, Any
import wandb

from absolute_bert.extractor import ParamStats
from absolute_bert.formatter import to_all_metrics_and_highlights
from absolute_bert.base_types import NestedMetricDict


class WandbLogger:

    def log_dict_without_commit(self, dict_: dict[Any, Any], global_step: int) -> None:
        wandb.log(dict_, step=global_step)

    def log_beir_metrics_without_commit(
        self,
        nested_metric_dicts: Iterable[tuple[str, NestedMetricDict]],
        global_step: int,
        corpus_name: str,
        epoch_num: int | None = None,
    ) -> None:
        for name, nested_metric_dict in nested_metric_dicts:
            mixed_dict = to_all_metrics_and_highlights(nested_metric_dict, f"{corpus_name}-{name}")

            logging_dict = mixed_dict
            if epoch_num is not None:
                logging_dict |= {"epoch": epoch_num}

            wandb.log(logging_dict, step=global_step, commit=False)

    def log_param_stats_without_commit(self, param_stats: ParamStats, global_step: int) -> None:

        logging_dict = {}

        norm_prefix = "params/module_norm"
        for module_name, norm in param_stats["norm"].items():
            logging_dict[f"{norm_prefix}/{module_name}"] = norm

        dist_prefix = "params/module_dist"
        for module_name, histogram_data in param_stats["dist"].items():
            histogram = wandb.Histogram(np_histogram=(histogram_data.counts, histogram_data.bins))
            logging_dict[f"{dist_prefix}/{module_name}"] = histogram

        wandb.log(logging_dict, step=global_step, commit=False)
