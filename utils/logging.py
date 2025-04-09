from typing import Sequence, List, Dict, Optional
from collections import defaultdict, OrderedDict
import time
from contextlib import contextmanager
import fnmatch
import re
import logging

import numpy as np
import wandb
import torch
from torch.nn import Module


logger = logging.getLogger(__name__)

class ParamLogger:
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

    def __init__(self, model: Module, config_rules: List[Dict], prefix: str = "params"):
      self.model = model
      self.prefix = prefix
      self.named_modules = dict(model.named_modules())
      self.named_parameters = dict(model.named_parameters())
      self.config_rules = config_rules
      self.resolved_logging_name_type_pairs = {}

      module_keys = list(self.named_modules.keys())
      for rule in config_rules:
        include = [self._resolve_suffix_index(r, module_keys) for r in rule.get("include", ["*"])]
        exclude = [self._resolve_suffix_index(r, module_keys) for r in rule.get("exclude", [])]
        log_distribution = rule.get("log_distribution", True)
        log_norm = rule.get("log_norm", True)
        print("include", include)
        print("exclude", exclude)
        for name, param in self.named_parameters.items():
          if param.numel() == 0:
            continue
          if not self._match_name(name, include):
            continue
          if self._match_name(name, exclude):
            continue

          if log_distribution:
            self.resolved_logging_name_type_pairs[(name, "dist")] = param
          if log_norm:
            self.resolved_logging_name_type_pairs[(name, "norm")] = param


    def _match_name(self, name: str, patterns: List[str]) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in patterns)

    def _resolve_suffix_index(self, target: str, named_dict: list[str], recursive_root=True) -> str:
        class PrefixError(ValueError): pass
        class IndexError(ValueError): pass

        m = re.search(r"\[(-?\d+)\](?:\.(.*))?", target)
        if not m:
            return target

        prefix_len, _ = m.span()
        prefix = target[:prefix_len]
        index = int(m.group(1))
        suffix = m.group(2)

        same_prefix_module_suffixes = [name[prefix_len:] for name in named_dict if name.startswith(prefix)]

        if not same_prefix_module_suffixes:
            raise PrefixError(prefix)

        suffix_pattern = re.compile(rf"(?:\.(\d+)(\.[^\.]+)?$)?")  # 一定要 ".0[.xxx]" 或是空的，數字在 group1，後續在 group2
        module_indices = set()
        suffixes = set()
        for candidate_module_name in same_prefix_module_suffixes:
            module_index_re = suffix_pattern.match(candidate_module_name)
            
            if not suffix_pattern:
                raise IndexError(prefix)

            module_index = module_index_re.group(1)
            more_suffix = module_index_re.group(2)
            if not more_suffix:
                if module_index is not None:
                  module_indices.add(module_index)
            else:
                suffixes.add(more_suffix)

        resolved_index = sorted(module_indices)[index]

        if suffix:
            try:
                suffix = self._resolve_suffix_index(suffix, suffixes, recursive_root=False)

            except IndexError as e:
                if recursive_root:
                    raise ValueError(f"Can't use [index] syntax in module name: {prefix}{e.args[0]}. "
                        f"There may be some module having same prefix, while not in the same module list.")
                else:
                    raise IndexError(f"{prefix}{e.args[0]}")

            except PrefixError as e:
                if recursive_root:
                    raise ValueError(f"No module name with prefix: {prefix}{e.args[0]}.")
                else: 
                    raise PrefixError(f"{prefix}{e.args[0]}")
            
            return f"{prefix}.{resolved_index}.{suffix}"

        return f"{prefix}.{resolved_index}"


    def extract_stats(self) -> Dict[str, object]:
        log_dict = {}

        for (name, log_type), param in self.resolved_logging_name_type_pairs.items():
            full_key = f"{self.prefix}/{name}"
            try:
                if log_type == "dist":
                    log_dict[f"{full_key}_dist"] = wandb.Histogram(param.detach().cpu().numpy())
                elif log_type == "norm":
                    log_dict[f"{full_key}_norm"] = param.detach().norm().item()

            except Exception as e:
                logger.exception(f"[param_log] Cannot log {alias_name}: {e}")

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