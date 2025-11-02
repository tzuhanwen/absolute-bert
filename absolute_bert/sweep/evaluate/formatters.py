from typing import TypeAlias

MetricDict: TypeAlias = dict[str, float]  # name@k -> value
MetricDicts: TypeAlias = tuple[MetricDict, MetricDict, MetricDict, MetricDict]
NestedMetricDict: TypeAlias = dict[str, dict[int, float]]


def nest_a_metric_dict_tuple(metric_dicts: MetricDicts) -> NestedMetricDict:

    nested_dict: dict[str, dict[int, float]] = {}  # NDCG -> "1" -> value of NDCG@1
    for metric_type, metric_dict in zip(["NDCG", "MAP", "Recall", "P"], metric_dicts):

        inner_dict: dict[int, float] = {}
        for full_name, value in metric_dict.items():
            at_value = int(full_name.strip(f"{metric_type}@"))
            inner_dict[at_value] = value

        nested_dict |= {metric_type: inner_dict}

    return nested_dict
