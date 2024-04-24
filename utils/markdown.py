from collections import OrderedDict

def beir_metrics_to_markdown_table(ndcg, map_, recall, precision):
    metrics = OrderedDict({
        'NDCG': ndcg,
        'MAP': map_,
        'Recall': recall,
        'P': precision
    })
    k_values = [key.split('@')[1] for key in ndcg.keys()]  # extract the @ values of metrics

    header = (f"""||{'|'.join(metrics.keys())}|\n"""
              f"""|-{'|-'*len(metrics)}|\n""")
    rows = [
        f'|@{k}|' + "|".join([f"{metric_dict[f'{metric_name}@{k}']:.4f}"
                             for metric_name, metric_dict in metrics.items()])
        for k in k_values]
    rows = '|\n'.join(rows) + '|'
    
    return header + rows