from collections import OrderedDict
import numpy as np

import wandb
import torch

from torch.nn import Module
from typing import Sequence

def extract_param_stats(modules: Sequence[Module], model: Module, prefix: str = "params", log_distribution: bool = True, log_norm: bool = True):
    name_to_module = {mod: name for name, mod in model.named_modules()}
    log_dict = {}

    for i, mod in enumerate(modules):
        mod_name = name_to_module.get(mod, f"unnamed_module_{i}")

        for param_name, param in mod.named_parameters():
            if param.numel() == 0:
                continue
            full_name = f"{prefix}/{mod_name}.{param_name}"
            try:
                if log_distribution:
                    log_dict[f"{full_name}_dist"] = wandb.Histogram(param.detach().cpu().numpy())
                if log_norm:
                    log_dict[f"{full_name}_norm"] = param.detach().norm().item()
            except Exception as e:
                print(f"[warn] Cannot log {full_name}: {e}")

    return log_dict


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