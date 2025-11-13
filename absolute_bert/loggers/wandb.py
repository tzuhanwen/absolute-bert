import tempfile
import wandb
from pathlib import Path
from typing import Iterable, Any, TypeAlias

from absolute_bert.base_types import NestedMetricDict
from absolute_bert.extractor import HistogramData, Statistic
from absolute_bert.formatter.metric import to_all_metrics_and_highlights

FileName: TypeAlias = str
FileContent: TypeAlias = str
Files: TypeAlias = dict[FileName, FileContent]


class WandbLogger:

    def log_dict(self, dict_: dict[Any, Any], global_step: int) -> None:
        wandb.log(dict_, step=global_step, commit=False)

    def log_beir_metrics(
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

    def log_param_stats(self, param_stats: dict[str, Statistic], global_step: int) -> None:

        logging_dict = {}

        for stat_name, stat in param_stats.items():
            if isinstance(stat, HistogramData):
                histogram = wandb.Histogram(np_histogram=(stat.counts, stat.bins))
                logging_dict[f"params/{stat_name}"] = histogram
                continue

            logging_dict[f"params/{stat_name}"] = stat

        wandb.log(logging_dict, step=global_step, commit=False)

    def dump_strings(self, files: Files, category_name: str, global_step: int) -> None:

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            for file_name, content in files.items():
                (path / file_name).write_text(content)

            artifact = wandb.Artifact(
                name=f"{category_name}-{wandb.run.id}",
                type=category_name,
                metadata={"global_step": global_step},
            )
            artifact.add_dir(path)
            wandb.log_artifact(artifact, aliases=[f"step_{global_step}", "latest"])
