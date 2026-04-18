import os

from .base_logger import BaseLogger
from typing import Dict, Any, List
import numpy as np


def is_basic(x):
    return isinstance(x, str) or isinstance(x, int) or isinstance(x, float) or isinstance(x, bool)


def convert_no_basic_to_str(sub_dict: Dict[str, Any]):
    return {k: v if is_basic(v)
    else str(v) if not isinstance(v, dict) else convert_no_basic_to_str(v)
            for k, v in sub_dict.items()}


def convert_no_basic_to_str_from_any(p: Any):
    if is_basic(p):
        return p
    elif isinstance(p, dict):
        return convert_no_basic_to_str(p)
    else:
        return str(p)


def _to_native(v):
    """Convert numpy scalars to Python native types for WandB compatibility."""
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


class WandbLogger(BaseLogger):
    """WandB logger with automatic batching.

    Consecutive ``log()`` calls that share the same (step, step_key) are
    accumulated into a single ``wandb.log()`` call.  The buffer is flushed
    automatically when the step changes, or when ``log_metrics``,
    ``log_file``, ``_log_fig``, or ``stop`` are called.
    """

    def __init__(self, entity="azencot-group", project="ts_corrupted", *args, **kwargs):
        super(WandbLogger, self).__init__(*args, **kwargs)
        import wandb
        self.wandb = wandb

        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)

        self.run = wandb.init(entity=entity, project=project)

        wandb.define_metric("custom_step", hidden=True)
        wandb.define_metric("train_step", hidden=True)

        # Explicit patterns — the bare "*" glob has a known bug (wandb #5059)
        # where it fails to match metric names containing "/".  Listing
        # concrete prefixes works on every wandb version.
        wandb.define_metric("em/*", step_metric="custom_step")
        wandb.define_metric("test/*", step_metric="custom_step")
        wandb.define_metric("eval/*", step_metric="custom_step")
        wandb.define_metric("test/memorization/*", step_metric="custom_step")
        wandb.define_metric("warmstart/*", step_metric="custom_step")
        wandb.define_metric("*", step_metric="custom_step")

        wandb.define_metric("em/m_step_loss", step_metric="train_step")

        self._train_step_metrics = {"em/m_step_loss"}
        self._pending: Dict[str, Any] = {}
        self._pending_step = None
        self._pending_step_key = None

    def _flush_pending(self):
        if not self._pending:
            return
        log_dict = dict(self._pending)
        if self._pending_step is not None:
            log_dict[self._pending_step_key] = int(self._pending_step)
        self.run.log(log_dict)
        self._pending = {}
        self._pending_step = None
        self._pending_step_key = None

    def stop(self, exit_code=0):
        self._flush_pending()
        self.run.finish(exit_code=exit_code)

    def log(self, name: str, data: Any, step=None):
        step_key = "train_step" if name in self._train_step_metrics else "custom_step"
        flush_key = (step, step_key)
        current_key = (self._pending_step, self._pending_step_key)

        if self._pending and flush_key != current_key:
            self._flush_pending()

        self._pending[name] = _to_native(data)
        # Duplicate discriminative eval metrics under eval/* for dedicated charts (same step as test/*).
        if name in ("test/disc_mean", "test/disc_std"):
            self._pending["eval/" + name.split("/", 1)[1]] = _to_native(data)
        self._pending_step = step
        self._pending_step_key = step_key

    def log_metrics(self, metrics: Dict[str, Any], step=None):
        """Log all metrics in a single wandb.log() call so they share one row."""
        self._flush_pending()
        log_dict = {k: _to_native(v) for k, v in metrics.items()}
        for k in ("test/disc_mean", "test/disc_std"):
            if k in log_dict:
                log_dict["eval/" + k.split("/", 1)[1]] = log_dict[k]
        if step is not None:
            has_train = any(k in self._train_step_metrics for k in metrics)
            if has_train:
                log_dict["train_step"] = int(step)
            else:
                log_dict["custom_step"] = int(step)
        self.run.log(log_dict)

    def _log_fig(self, name: str, fig: Any):
        self._flush_pending()
        if isinstance(fig, np.ndarray):
            from PIL import Image
            fig = Image.fromarray(fig)
        self.run.log({name: self.wandb.Image(fig)})

    def log_file(self, name: str, file_path: str, step=None):
        self._flush_pending()
        # Intentionally omit custom_step: images don't need a custom x-axis,
        # and duplicating the same custom_step across multiple wandb.log()
        # calls can confuse chart rendering for numeric metrics.
        self.run.log({name: self.wandb.Image(file_path)})

    def log_hparams(self, params: Dict[str, Any]):
        params = convert_no_basic_to_str(params)
        self.wandb.config.update({"hyperparameters": params}, allow_val_change=True)

    def log_params(self, params: Dict[str, Any]):
        params = convert_no_basic_to_str(params)
        self.wandb.config.update({"parameters": params}, allow_val_change=True)

    def add_tags(self, tags: List[str]):
        current = list(self.run.tags) if self.run.tags else []
        self.run.tags = list(set(current + list(tags)))

    def log_name_params(self, name: str, params: Any):
        params = convert_no_basic_to_str_from_any(params)
        self.wandb.config.update({name: params}, allow_val_change=True)
