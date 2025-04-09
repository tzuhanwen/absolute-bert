from collections import OrderedDict
from contextlib import contextmanager
import time
import logging

import numpy as np
import wandb
import torch

from torch.nn import Module
from typing import Sequence


logger = logging.getLogger(__name__)


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



@contextmanager
def log_step(step: int = None, tag: str = None):
    """
    Context manager for safe and traceable wandb logging steps.

    Ensures that:
      - Long-running computation (e.g. IR evaluation, validation) completes before logging
      - Log order stays strictly increasing (avoids wandb step desync warning)
      - Prints runtime duration for performance debugging
      - Catches and reports exceptions within the context block

    Args:
        step (int): The wandb step to log under. Make sure this matches global_step at the time of logging.
        tag (str): Optional tag for printing debug info (e.g. "IR_eval", "val", "epoch_1_end").

    Example usage:
        with log_step(step=global_step, tag="IR_eval"):
            wandb.log(get_beir_log_dict(benchmark.run(model_output_method), "model_output"), step=global_step)
            wandb.log(get_beir_log_dict(benchmark.run(static_embeddings_method), "static_embeddings"), step=global_step)

    Note:
        This is especially useful for logging long-running metrics like validation accuracy or IR metrics,
        where step desync can cause wandb to drop logs or raise warnings.

    """
    start = time.time()
    try:
        yield
    except Exception as e:
        logger.exception(f"[log_step] ERROR at step={step} tag={tag or ''}")
        raise
    else:
        elapsed = time.time() - start
        logger.info(f"[log_step] ✅ step={step} tag={tag or ''} took {elapsed:.2f}s")