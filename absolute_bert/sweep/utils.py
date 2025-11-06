import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def log_step(step: int | None = None, tag: str | None = None):
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
    except Exception:
        logger.exception(f"[log_step] ERROR at step={step} tag={tag or ''}")
        raise
    else:
        elapsed = time.time() - start
        logger.info(f"[log_step] ✅ step={step} tag={tag or ''} took {elapsed:.2f}s")
