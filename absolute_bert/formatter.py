from typing import TypeAlias
from absolute_bert.base_types import NestedMetricDict

MetricDict: TypeAlias = dict[str, float]  # name@k -> value
MetricDicts: TypeAlias = tuple[MetricDict, MetricDict, MetricDict, MetricDict]


def nest_a_metric_dict_tuple(metric_dicts: MetricDicts) -> NestedMetricDict:

    nested_dict: dict[str, dict[int, float]] = {}  # ex: NDCG -> "1" -> value of NDCG@1
    for metric_type, metric_dict in zip(["NDCG", "MAP", "Recall", "P"], metric_dicts, strict=True):

        inner_dict: dict[int, float] = {}
        for full_name, value in metric_dict.items():
            at_value = int(full_name.strip(f"{metric_type}@"))
            inner_dict[at_value] = value

        nested_dict |= {metric_type: inner_dict}

    return nested_dict


def to_all_metrics_and_highlights(metric_dict: NestedMetricDict, category_name: str):
    all_metrics = {f"{category_name}/{k}": v for k, v in metric_dict.items()}
    highlights = {f"highlight/{category_name}/{k}": {"10": v[10]} for k, v in metric_dict.items()}
    return all_metrics | highlights
