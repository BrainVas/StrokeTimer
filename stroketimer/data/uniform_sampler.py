# ./data/UniformSampler.py
# -*- coding: utf-8 -*-
"""
UniformSampler: instance-level uniform sampling over the whole dataset.
- Keeps the original class distribution (each index has equal probability).
- Compatible with your loader: get_sampler() returns a Sampler class.
"""

from __future__ import annotations
from typing import Optional, Iterable
import torch
from torch.utils.data import Sampler

__all__ = ["UniformSampler", "get_sampler"]


class UniformSampler(Sampler[int]):
    """
    Instance-uniform sampler (like RandomSampler).
    Args:
        dataset: any Dataset (only requires __len__)
        num_samples: number of indices per epoch (default: len(dataset))
        replacement: sample with replacement (default: False)
        seed: base RNG seed (optional, call set_epoch for epoch-determinism)
    """
    def __init__(self, dataset, num_samples: Optional[int] = None,
                 replacement: bool = False, seed: Optional[int] = None):
        self.dataset = dataset
        self.N = len(dataset)
        self.num_samples = int(num_samples) if num_samples is not None else self.N
        self.replacement = bool(replacement)
        self._base_seed = int(seed) if seed is not None else None
        self._epoch = 0
        self._rng = None  # torch.Generator, lazy init

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        self._rng = None

    def _gen(self) -> Optional[torch.Generator]:
        if self._base_seed is None:
            return None
        if self._rng is None:
            g = torch.Generator()
            g.manual_seed(self._base_seed + self._epoch)
            self._rng = g
        return self._rng

    def __iter__(self) -> Iterable[int]:
        if self.replacement:
            idx = torch.randint(low=0, high=self.N, size=(self.num_samples,),
                                generator=self._gen())
        else:
            # draw a random permutation; if num_samples < N, take the first K
            perm = torch.randperm(self.N, generator=self._gen())
            idx = perm[: self.num_samples]
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples


def get_sampler():
    return UniformSampler
