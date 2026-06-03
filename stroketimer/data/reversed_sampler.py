# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Iterable, List, Dict
import math
import numpy as np
import torch
from torch.utils.data import Sampler

__all__ = ["ClassAwareReversedSampler", "get_sampler"]

def _get_targets(dataset, targets_attr: str = "targets") -> np.ndarray:
    if hasattr(dataset, targets_attr):
        arr = getattr(dataset, targets_attr)
    elif hasattr(dataset, "labels"):
        arr = getattr(dataset, "labels")
    else:
        raise AttributeError(f"Dataset has no '{targets_attr}' or 'labels' attribute.")
    arr = np.asarray(arr).reshape(-1)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64)
    return arr

def _safe_bincount(labels: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    if labels.size == 0:
        return np.zeros(0, dtype=np.int64)
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    bc = np.bincount(labels, minlength=num_classes)
    if bc.shape[0] < num_classes:
        bc = np.pad(bc, (0, num_classes - bc.shape[0]), constant_values=0)
    return bc

class ClassAwareReversedSampler(Sampler[int]):
    """
    Class-aware two-stage reverse sampler:
      1) sample classes with probability proportional to (count_c + eps)^(-power) with optional temperature;
      2) given the class, sample instance uniformly from that class.

    This matches BBN's "Reverse branch" spirit better than per-instance multinomial.
    """

    def __init__(
        self,
        dataset,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        power: float = 1.0,
        temperature: float = 1.0,
        targets_attr: str = "targets",
        num_classes: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.labels = _get_targets(dataset, targets_attr)
        self.num_classes = int(num_classes) if num_classes is not None else (int(self.labels.max()) + 1)
        self.replacement = bool(replacement)
        self.power = float(power)
        self.temperature = float(temperature)
        self.num_samples = int(num_samples) if num_samples is not None else int(len(self.labels))
        if self.num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        counts = _safe_bincount(self.labels, self.num_classes).astype(np.float64)
        eps = 1e-9
        # inverse-frequency weights
        class_w = (counts + eps) ** (-self.power)

        if not math.isclose(self.temperature, 1.0):
            logw = np.log(np.maximum(class_w, eps)) / max(self.temperature, eps)
            logw = logw - np.max(logw)  # stabilize
            class_w = np.exp(logw)

        # normalize
        class_w = class_w / np.maximum(class_w.sum(), eps)
        self.class_w_t = torch.as_tensor(class_w, dtype=torch.double)

        # build class -> indices map
        self.cls_to_idx: List[List[int]] = [[] for _ in range(self.num_classes)]
        for idx, y in enumerate(self.labels.tolist()):
            if 0 <= y < self.num_classes:
                self.cls_to_idx[y].append(idx)

        self._base_seed = int(seed) if seed is not None else None
        self._epoch = 0
        self._rng = None  # torch.Generator

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        self._rng = None

    def _get_generator(self) -> Optional[torch.Generator]:
        if self._base_seed is None:
            return None
        if self._rng is None:
            g = torch.Generator()
            g.manual_seed(self._base_seed + self._epoch)
            self._rng = g
        return self._rng

    def __iter__(self) -> Iterable[int]:
        # Stage 1: sample classes
        cls_ids = torch.multinomial(
            self.class_w_t,
            num_samples=self.num_samples,
            replacement=True,  # class draws should allow repetition
            generator=self._get_generator(),
        ).tolist()

        # Stage 2: for each class, draw an instance uniformly from that class
        out_idx: List[int] = []
        g = self._get_generator()
        for c in cls_ids:
            pool = self.cls_to_idx[c]
            if not pool:
                # if a class has 0 instances (shouldn't happen if num_classes correct), fallback random
                j = torch.randint(low=0, high=len(self.labels), size=(1,), generator=g).item()
                out_idx.append(j)
            else:
                if self.replacement:
                    # uniform with replacement
                    j = torch.randint(low=0, high=len(pool), size=(1,), generator=g).item()
                    out_idx.append(pool[j])
                else:
                    # without replacement per epoch per class: rotate an offset
                    # simple approach: random choose then remove
                    j = torch.randint(low=0, high=len(pool), size=(1,), generator=g).item()
                    out_idx.append(pool.pop(j))
                    if not pool:  # if emptied, rebuild from original indices to keep length
                        # rebuild class pool
                        pool.extend([i for i, y in enumerate(self.labels.tolist()) if y == c])

        return iter(out_idx)

    def __len__(self) -> int:
        return self.num_samples

def get_sampler():
    return ClassAwareReversedSampler
