# -*- coding: utf-8 -*-
"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""
"""Shared utilities used by the StrokeTimer training runner."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path_like) -> Path:
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return repo_root() / path


def source_import(file_path: str):
    path = resolve_repo_path(file_path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_value(config_value, override_value):
    return override_value if override_value is not None else config_value


def print_write(print_list: Iterable, log_file: str | None = None):
    text = "".join(str(item) for item in print_list)
    print(text)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")


def torch2numpy(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mic_acc_cal(preds, labels):
    preds_np = torch2numpy(preds).reshape(-1)
    labels_np = torch2numpy(labels).reshape(-1)
    if labels_np.size == 0:
        return 0.0
    return float((preds_np == labels_np).sum() / labels_np.size)


def weighted_mic_acc_cal(preds, labels, weights):
    preds_np = torch2numpy(preds).reshape(-1)
    labels_np = torch2numpy(labels).reshape(-1)
    weights_np = torch2numpy(weights).reshape(-1).astype(np.float64)
    if labels_np.size == 0 or weights_np.sum() <= 0:
        return 0.0
    return float(((preds_np == labels_np).astype(np.float64) * weights_np).sum() / weights_np.sum())


def class_count(data_loader) -> list[int]:
    dataset = getattr(data_loader, "dataset", data_loader)
    if hasattr(dataset, "get_cls_num_list"):
        return list(dataset.get_cls_num_list())
    if hasattr(dataset, "labels"):
        labels = list(map(int, dataset.labels))
    elif hasattr(dataset, "annotations"):
        labels = [int(a["category_id"]) for a in dataset.annotations]
    elif hasattr(dataset, "get_annotations"):
        labels = [int(a["category_id"]) for a in dataset.get_annotations()]
    else:
        return []
    n = max(labels) + 1 if labels else 0
    return [labels.count(i) for i in range(n)]


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    preds_np = torch2numpy(preds).reshape(-1).astype(int)
    labels_np = torch2numpy(labels).reshape(-1).astype(int)
    train_counts = class_count(train_data)
    num_classes = max(len(train_counts), int(labels_np.max()) + 1 if labels_np.size else 0)

    cls_accs = []
    many, median, low = [], [], []
    for c in range(num_classes):
        mask = labels_np == c
        acc = float((preds_np[mask] == labels_np[mask]).mean()) if mask.any() else 0.0
        cls_accs.append(acc)
        train_n = train_counts[c] if c < len(train_counts) else 0
        if train_n > many_shot_thr:
            many.append(acc)
        elif train_n < low_shot_thr:
            low.append(acc)
        else:
            median.append(acc)

    out = (
        float(np.mean(many)) if many else 0.0,
        float(np.mean(median)) if median else 0.0,
        float(np.mean(low)) if low else 0.0,
    )
    return (*out, cls_accs) if acc_per_cls else out


def weighted_shot_acc(preds, labels, weights, train_data, many_shot_thr=100, low_shot_thr=20):
    preds_np = torch2numpy(preds).reshape(-1).astype(int)
    labels_np = torch2numpy(labels).reshape(-1).astype(int)
    weights_np = torch2numpy(weights).reshape(-1).astype(np.float64)
    train_counts = class_count(train_data)
    num_classes = max(len(train_counts), int(labels_np.max()) + 1 if labels_np.size else 0)

    many, median, low = [], [], []
    for c in range(num_classes):
        mask = labels_np == c
        if mask.any() and weights_np[mask].sum() > 0:
            acc = float(((preds_np[mask] == labels_np[mask]).astype(np.float64) * weights_np[mask]).sum() / weights_np[mask].sum())
        else:
            acc = 0.0
        train_n = train_counts[c] if c < len(train_counts) else 0
        if train_n > many_shot_thr:
            many.append(acc)
        elif train_n < low_shot_thr:
            low.append(acc)
        else:
            median.append(acc)
    return (
        float(np.mean(many)) if many else 0.0,
        float(np.mean(median)) if median else 0.0,
        float(np.mean(low)) if low else 0.0,
    )


def get_priority(ptype: str, logits: torch.Tensor, labels: torch.Tensor):
    import torch
    if logits is None:
        return torch.ones_like(labels, dtype=torch.float32)
    probs = torch.softmax(logits.detach(), dim=1)
    labels = labels.detach().long()
    conf = probs.gather(1, labels.view(-1, 1)).squeeze(1)
    if ptype in {"score", "confidence"}:
        return conf
    if ptype in {"hard", "difficulty", "error"}:
        return 1.0 - conf
    return torch.ones_like(conf)


def init_weights(model, weights_path: str | None = None, classifier: bool = False):
    import torch
    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    else:
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(module.weight)
                if getattr(module, "bias", None) is not None:
                    torch.nn.init.zeros_(module.bias)
    return model


def CosineAnnealingLRWarmup(optimizer, T_max, eta_min=0.0, warmup_epochs=0, base_lr=0.0, warmup_lr=0.0, last_epoch=-1):
    """Epoch-level cosine scheduler with linear warmup."""
    import torch

    class _CosineAnnealingLRWarmup(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self):
            self.T_max = max(1, int(T_max))
            self.eta_min = float(eta_min)
            self.warmup_epochs = int(warmup_epochs)
            self.base_lr = float(base_lr) if base_lr else None
            self.warmup_lr = float(warmup_lr)
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            epoch = max(0, self.last_epoch)
            lrs = []
            for base_lr_item in self.base_lrs:
                peak_lr = self.base_lr if self.base_lr is not None else base_lr_item
                if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                    alpha = float(epoch + 1) / float(self.warmup_epochs)
                    lr = self.warmup_lr + alpha * (peak_lr - self.warmup_lr)
                else:
                    denom = max(1, self.T_max - self.warmup_epochs)
                    progress = min(1.0, max(0.0, (epoch - self.warmup_epochs) / denom))
                    lr = self.eta_min + 0.5 * (peak_lr - self.eta_min) * (1.0 + math.cos(math.pi * progress))
                lrs.append(lr)
            return lrs

    return _CosineAnnealingLRWarmup()
