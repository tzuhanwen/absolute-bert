from collections import OrderedDict


def beir_metrics_to_markdown_table(
    ndcg: dict[str, float],
    map_: dict[str, float],
    recall: dict[str, float],
    precision: dict[str, float],
) -> str:

    metrics: dict[str, dict[str, float]] = OrderedDict(
        {"NDCG": ndcg, "MAP": map_, "Recall": recall, "P": precision}
    )

    k_values = [key.split("@")[1] for key in ndcg.keys()]  # extract the @ values of metrics

    column_name_line = "||" + "|".join(metrics.keys()) + "|"
    column_format_line = "|-" + "|-" * len(metrics) + "|"

    lines = [column_name_line, column_format_line]

    for k in k_values:
        metric_values: list[str] = []
        for metric_name, metric_dict in metrics.items():
            metric_value = metric_dict[f"{metric_name}@{k}"]
            metric_values.append(f"{metric_value:.4f}")

        line = f"|@{k}|" + "|".join(metric_values) + "|"
        lines.append(line)

    return "\n".join(lines)
