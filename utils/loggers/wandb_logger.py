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


class WandbLogger(BaseLogger):

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
        wandb.define_metric("em/m_step_loss", step_metric="train_step")
        wandb.define_metric("*", step_metric="custom_step")

        self._train_step_metrics = {"em/m_step_loss"}

    def stop(self, exit_code=0):
        self.run.finish(exit_code=exit_code)

    def log(self, name: str, data: Any, step=None):
        log_dict = {name: data}
        if step is not None:
            step_key = "train_step" if name in self._train_step_metrics else "custom_step"
            log_dict[step_key] = step
        self.run.log(log_dict)

    def log_metrics(self, metrics: Dict[str, Any], step=None):
        """Log all metrics in a single wandb.log() call so they share one row."""
        log_dict = dict(metrics)
        if step is not None:
            has_train = any(k in self._train_step_metrics for k in metrics)
            if has_train:
                log_dict["train_step"] = step
            else:
                log_dict["custom_step"] = step
        self.run.log(log_dict)

    def _log_fig(self, name: str, fig: Any):
        if isinstance(fig, np.ndarray):
            from PIL import Image
            fig = Image.fromarray(fig)
        self.run.log({name: self.wandb.Image(fig)})

    def log_file(self, name: str, file_path: str, step=None):
        log_dict = {name: self.wandb.Image(file_path)}
        if step is not None:
            log_dict["custom_step"] = step
        self.run.log(log_dict)

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
