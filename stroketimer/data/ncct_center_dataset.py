# -*- coding: utf-8 -*-
"""
NCCTDataset-3D for long-tail classification with center-stratified meta CSV.

Functionality:
- Uses a meta CSV (e.g., meta_with_phase_center_stratified.csv)
  that must contain at least: patient_id, bucket_age, phase.
  If pid_canon is not present, it is computed from patient_id.
- root contains center folders / vendor / model / kernel / *.npy or *.nii.gz.
- Train split keeps the original long-tailed distribution.
- Val/Test splits can be strictly balanced (per-class down-sampling to min count).

Return:
    (x3d, label, index)
    x3d: torch.float32, shape (1, 48, 256, 256)
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import re
import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ----------------- global constants -----------------

# Map bucket_age strings to integer labels
_BUCKET_MAP = {"4.5<": 0, "<4.5": 0, "4.5-6": 1, ">6": 2}
_NUM_CLASSES = 3

# Expected (D, H, W)
_EXPECTED_DHW = (48, 256, 256)
_EXPECTED_DEPTH = _EXPECTED_DHW[0]

# Patient ID head: Rxxxx or 3xxxx
_PID_HEAD = re.compile(r"^(R\d+|3\d+)", re.IGNORECASE)


# ----------------- helper functions -----------------

def _map_bucket_to_int(s: str) -> int:
    """Map bucket_age string to integer label."""
    s = str(s).strip()
    if s not in _BUCKET_MAP:
        raise ValueError(f"bucket_age '{s}' not in {_BUCKET_MAP.keys()}.")
    return _BUCKET_MAP[s]


def _canonical_pid(pid: str) -> str:
    """
    Canonicalize patient_id:
    - If starts with R or r: keep 'R' + integer part, e.g. an R-style case id.
    - If starts with '3': keep the longest 3xxxx prefix.
    - Otherwise, return as-is (string).
    """
    pid = str(pid).strip()
    if not pid:
        return pid

    # R prefix
    if pid[0] in ("R", "r"):
        nums = "".join(ch for ch in pid[1:] if ch.isdigit())
        return "R" + str(int(nums)) if nums != "" else "R"

    # 3xxxx pattern
    if pid[0] == "3":
        m = re.match(r"3\d+", pid)
        return m.group(0) if m else pid

    return pid


def _safe_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Robust z-score normalization with NaN/Inf handling."""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    std = float(x.std())
    mean = float(x.mean())
    std = max(std, eps)
    x = (x - mean) / std
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _ensure_dhw(vol: np.ndarray) -> np.ndarray:
    """
    Ensure the volume is float32 and shaped (D, H, W).
    Handles possible channels or (H, W, D) arrangements.
    """
    v = vol
    # Squeeze trivial 4D shapes
    if v.ndim == 4:
        if v.shape[0] == 1:
            v = v[0]
        elif v.shape[-1] == 1:
            v = v[..., 0]
        else:
            raise ValueError(f"Unsupported 4D shape: {v.shape}")

    if v.ndim != 3:
        raise ValueError(f"Volume must be 3D after squeeze, got {v.shape}.")

    # Try to move depth axis to position 0
    if _EXPECTED_DEPTH in v.shape and v.shape[0] != _EXPECTED_DEPTH:
        depth_axis = int(list(v.shape).index(_EXPECTED_DEPTH))
        v = np.moveaxis(v, depth_axis, 0)

    # Heuristic: if depth seems to be last axis
    if v.shape[0] != _EXPECTED_DEPTH and v.shape[-1] == _EXPECTED_DEPTH:
        v = np.moveaxis(v, -1, 0)

    # Another heuristic: if first axis is huge compared to last
    if v.shape[0] >= 128 and v.shape[-1] < v.shape[0]:
        v = np.moveaxis(v, -1, 0)

    return v.astype(np.float32, copy=False)


def _center_crop_depth(v: np.ndarray, target_d: int) -> np.ndarray:
    """Center-crop along depth (axis 0) to target_d."""
    d = v.shape[0]
    if d <= target_d:
        return v
    start = (d - target_d) // 2
    return v[start:start + target_d]


def _pad_depth(v: np.ndarray, target_d: int) -> np.ndarray:
    """Symmetric zero-padding along depth (axis 0) to target_d."""
    d = v.shape[0]
    if d >= target_d:
        return v
    pad_total = target_d - d
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return np.pad(v, ((pad_before, pad_after), (0, 0), (0, 0)),
                  mode="constant", constant_values=0.0)


def _ensure_depth_48(v: np.ndarray) -> np.ndarray:
    """Adjust depth to 48 by center-crop or padding."""
    d = v.shape[0]
    if d == _EXPECTED_DEPTH:
        return v
    if d > _EXPECTED_DEPTH:
        return _center_crop_depth(v, _EXPECTED_DEPTH)
    return _pad_depth(v, _EXPECTED_DEPTH)


# ----------------- main Dataset class -----------------

class NCCTDataset(Dataset):
    """
    3D-only dataset:
      - returns (x3d, label, index)
      - x3d is torch.float32, shape (1, 48, 256, 256)

    Arguments:
        root:       root directory with volume files (center/vendor/model/kernel).
        phase:      'train' | 'val' | 'test'
        csv_path:   meta CSV with at least columns: patient_id, bucket_age, phase.
                    If 'pid_canon' is absent, it will be computed.
        balance_eval: if True, val/test will be strictly down-sampled so that
                      each class has the same number of samples (min over classes).
    """
    def __init__(
        self,
        root: str,
        phase: str,                                 # 'train' | 'val' | 'test'
        transform=None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),  # kept for API, unused if CSV has 'phase'
        seed: int = 42,
        verbose: bool = True,
        csv_path: Optional[str] = None,
        fix_depth: bool = True,
        zscore: bool = True,
        balance_eval: bool = True,
    ):
        super().__init__()
        assert phase in {"train", "val", "test"}
        self.root = Path(root)
        self.phase = phase
        self.transform = transform
        self.verbose = verbose
        self.fix_depth = fix_depth
        self.zscore = zscore
        self.balance_eval = balance_eval
        self._rng_seed = int(seed)

        # --------- 1. resolve CSV ---------
        if csv_path is None:
            raise FileNotFoundError(
                "NCCTDataset (center-stratified) requires an explicit csv_path "
                "(e.g., /path/to/meta_with_phase_center_stratified.csv)."
            )
        csv_file = Path(csv_path)
        if not csv_file.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_file}")

        df = pd.read_csv(csv_file)

        if "phase" not in df.columns:
            raise ValueError("Meta CSV must contain a 'phase' column (train/val/test).")

        # --------- 2. canonical patient id ---------
        df["patient_id"] = df["patient_id"].astype(str)
        if "pid_canon" not in df.columns:
            df["pid_canon"] = df["patient_id"].map(_canonical_pid)
        else:
            df["pid_canon"] = df["pid_canon"].astype(str).map(_canonical_pid)

        # --------- 3. filter by phase ---------
        df_phase = df[df["phase"].astype(str).str.lower() == self.phase].copy()
        if df_phase.empty:
            raise ValueError(f"No rows with phase='{self.phase}' in {csv_file}")

        # --------- 4. build label map (pid_canon -> int label) ---------
        if "bucket_age" not in df_phase.columns:
            raise ValueError("Meta CSV must contain 'bucket_age' in {'<4.5','4.5-6','>6','4.5<'}.")

        label_map: Dict[str, int] = {}
        for _, row in df_phase.iterrows():
            try:
                y = _map_bucket_to_int(row["bucket_age"])
            except Exception:
                continue
            pid = str(row["pid_canon"])
            label_map[pid] = int(y)

        # --------- 5. collect volume paths ---------
        self.samples = self._collect_samples(self.root, label_map)
        if len(self.samples) == 0:
            raise ValueError(f"No valid volumes found under {self.root} for phase='{self.phase}'.")

        # --------- 6. strict balancing for val/test (optional) ---------
        if (self.phase in {"val", "test"}) and self.balance_eval:
            before = len(self.samples)
            self.samples = self._balance_downsample(self.samples, seed=self._rng_seed)
            after = len(self.samples)
            if self.verbose:
                print(f"[balance] {self.phase}: {before} -> {after} rows (strict per-class down-sampling).")

        # --------- 7. cache labels and class counts ---------
        self.labels: List[int] = [int(s["label"]) for s in self.samples]
        self._num_per_cls_dict = self._count_per_class(self.labels, _NUM_CLASSES)

        if self.verbose:
            counts = [self._num_per_cls_dict.get(c, 0) for c in range(_NUM_CLASSES)]
            print(f"[NCCTDataset-3D] phase={self.phase} | N={len(self.samples)} | class_counts={counts}")

    # ----------------- balancing helpers -----------------

    @staticmethod
    def _balance_downsample(samples: List[Dict], seed: int) -> List[Dict]:
        """
        Strictly down-sample each class to K = min_c n_c.
        Deterministic per seed.
        """
        from collections import defaultdict
        import random as pyrand

        per_cls = defaultdict(list)
        for i, s in enumerate(samples):
            per_cls[int(s["label"])].append(i)

        if len(per_cls) == 0:
            return samples

        sizes = [len(v) for v in per_cls.values()]
        K = min(sizes)
        if K <= 0:
            return samples

        rng = pyrand.Random(seed)
        keep_idx: List[int] = []
        for c, idxs in per_cls.items():
            idxs = idxs.copy()
            rng.shuffle(idxs)
            keep_idx.extend(idxs[:K])

        keep_idx = sorted(keep_idx)
        return [samples[i] for i in keep_idx]

    # ----------------- filesystem helpers -----------------

    def _collect_samples(self, root: Path, label_map: Dict[str, int]) -> List[Dict]:
        """
        Scan for .npy or .nii.gz files whose filename starts with R#### or 3####;
        match canonicalized pid to label_map.
        """
        samples: List[Dict] = []
        for p in root.rglob("*"):
            p = Path(p)
            if not p.is_file():
                continue

            is_npy = (p.suffix == ".npy")
            is_nii = ("".join(p.suffixes[-2:]) == ".nii.gz")
            if not (is_npy or is_nii):
                continue

            stem = p.stem.replace(".nii", "")
            m = _PID_HEAD.match(stem)
            if not m:
                continue
            pid = _canonical_pid(m.group(1))
            if pid not in label_map:
                continue

            samples.append({
                "path": str(p),
                "pid": pid,
                "label": int(label_map[pid]),
            })
        return samples

    @staticmethod
    def _count_per_class(labels: List[int], num_classes: int) -> Dict[int, int]:
        d: Dict[int, int] = {i: 0 for i in range(num_classes)}
        for y in labels:
            if 0 <= y < num_classes:
                d[y] += 1
        return d

    def _load_vol(self, path: str) -> np.ndarray:
        """Load .npy or .nii.gz and normalize layout/shape."""
        if path.endswith(".npy"):
            arr = np.load(path)
            vol = _ensure_dhw(arr)
        else:
            img = nib.load(path).get_fdata().astype(np.float32)
            vol = _ensure_dhw(img)
        if self.fix_depth:
            vol = _ensure_depth_48(vol)
        return vol

    # ----------------- PyTorch Dataset API -----------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        vol = self._load_vol(s["path"])          # (D, H, W), float32

        if self.zscore:
            vol = _safe_zscore(vol)

        x = torch.from_numpy(vol).unsqueeze(0)   # (1, D, H, W)
        if self.transform is not None:
            x = self.transform(x)

        if not torch.is_floating_point(x):
            x = x.float()

        assert x.dim() == 4, f"Expected 4D tensor (C,D,H,W), got {tuple(x.shape)}"
        c, d, h, w = x.shape
        assert c == 1, f"Expected C=1, got C={c}"

        if (d, h, w) != _EXPECTED_DHW:
            x = self._final_fix_shape(x, _EXPECTED_DHW)

        label = int(s["label"])
        return x, label, index

    @staticmethod
    def _final_fix_shape(x: torch.Tensor, target_dhw: tuple) -> torch.Tensor:
        """
        Center-crop / pad tensor x (1,D,H,W) to target shape (1, 48, 256, 256).
        """
        _, d, h, w = x.shape
        td, th, tw = target_dhw

        # Depth
        if d > td:
            start = (d - td) // 2
            x = x[:, start:start + td]
        elif d < td:
            pad_before = (td - d) // 2
            pad_after = td - d - pad_before
            x = F.pad(x, (0, 0, 0, 0, 0, 0, pad_before, pad_after))

        # Height
        _, d, h, w = x.shape
        if h > th:
            start = (h - th) // 2
            x = x[:, :, start:start + th, :]
        elif h < th:
            pad_before = (th - h) // 2
            pad_after = th - h - pad_before
            x = F.pad(x, (0, 0, pad_before, pad_after, 0, 0, 0, 0))

        # Width
        _, d, h, w = x.shape
        if w > tw:
            start = (w - tw) // 2
            x = x[:, :, :, start:start + tw]
        elif w < tw:
            pad_before = (tw - w) // 2
            pad_after = tw - w - pad_before
            x = F.pad(x, (pad_before, pad_after, 0, 0, 0, 0, 0, 0))

        return x

    # ----------------- helpers for samplers -----------------

    def get_annotations(self):
        return [{'category_id': int(y)} for y in self.labels]

    def get_num_classes(self) -> int:
        return _NUM_CLASSES

    def get_cls_num_list(self) -> List[int]:
        return [self._num_per_cls_dict.get(i, 0) for i in range(_NUM_CLASSES)]
