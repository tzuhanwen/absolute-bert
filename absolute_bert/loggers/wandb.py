from typing import Iterable, Any
import wandb

from absolute_bert.extractor import HistogramData, Statistic
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

    def log_param_stats_without_commit(self, param_stats: dict[str, Statistic], global_step: int) -> None:

        logging_dict = {}

        for stat_name, stat in param_stats.items():
            if isinstance(stat, HistogramData):
                histogram = wandb.Histogram(np_histogram=(stat.counts, stat.bins))
                logging_dict[f"params/{stat_name}"] = histogram
                continue

            logging_dict[f"params/{stat_name}"] = stat

        wandb.log(logging_dict, step=global_step, commit=False)
