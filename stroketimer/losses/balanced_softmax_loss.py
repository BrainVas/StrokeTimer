# -*- coding: utf-8 -*-
# Balanced Softmax Cross-Entropy (logit adjustment by class frequency)
# Robust to YAML strings; supports counts as list/tensor/path or deferred init.
#
# Forward signature (runner-compatible):
#   forward(logits, y, features=None, classifier=None)
#
# References:
#   - "Three-Headed Bias in Long-Tailed Recognition" (logit adjustment)

import os
import csv
import json
from typing import Optional, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_float(x, name: str) -> float:
    """Cast YAML-loaded numeric (possibly string) to float with clear error."""
    try:
        return float(x)
    except Exception:
        raise ValueError(f"{name} must be a float-like value; got {type(x)}={x!r}")


def _load_counts_from_path(p: str) -> torch.Tensor:
    """Load class counts from JSON or CSV. Supports:
       - JSON list: [c0, c1, ...]
       - JSON dict (flat): {"0": 123, "1": 45, ...}
       - JSON nested per-class:
         {"0": {"class_name": "...", "num_samples": 1546, "freq": 0.90}, ...}
       - CSV/TXT: one number per line (or take last column)
    """
    if not os.path.exists(p):
        raise FileNotFoundError(f"class_counts path not found: {p}")

    ext = os.path.splitext(p)[1].lower()
    if ext in [".json", ".jsn"]:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # 1) list -> counts
        if isinstance(obj, list):
            vals = [float(v) for v in obj]
            return torch.tensor(vals, dtype=torch.float32)

        # 2) dict
        if isinstance(obj, dict):
            # Sort keys numerically when possible
            try:
                keys = sorted(obj.keys(), key=lambda k: int(k))
            except Exception:
                keys = list(obj.keys())

            # 2a) flat dict: {"0": 1546, ...}
            if all(isinstance(obj[k], (int, float)) for k in keys):
                vals = [float(obj[k]) for k in keys]
                return torch.tensor(vals, dtype=torch.float32)

            # 2b) nested dict per class
            vals = []
            has_num = False
            freqs = []
            for k in keys:
                v = obj[k]
                if not isinstance(v, dict):
                    raise ValueError(f"JSON value for class {k} is not dict nor number")
                if "num_samples" in v:
                    vals.append(float(v["num_samples"]))
                    has_num = True
                elif "freq" in v:
                    freqs.append(float(v["freq"]))
                    vals.append(None)  # placeholder; fill later
                else:
                    raise ValueError(f"Nested JSON for class {k} missing 'num_samples' or 'freq'")

            if has_num:
                # If some classes use num_samples, prefer them; fill the rest using freq * total_num
                total_num = sum([x for x in vals if x is not None])
                if total_num <= 0 and len(freqs) > 0:
                    total_num = 1.0  # fallback to avoid zeros
                out = []
                for k in keys:
                    v = obj[k]
                    if "num_samples" in v:
                        out.append(float(v["num_samples"]))
                    else:
                        out.append(float(v["freq"]) * total_num)
                return torch.tensor(out, dtype=torch.float32)
            else:
                # All classes only provide freq: normalize to sum=1 then scale by class_count
                s = sum(freqs) if len(freqs) > 0 else 1.0
                out = [(float(obj[k]["freq"]) / max(s, 1e-12)) * len(keys) for k in keys]
                return torch.tensor(out, dtype=torch.float32)

        raise ValueError(f"Unsupported JSON structure for counts: {type(obj)}")

    elif ext in [".csv", ".txt"]:
        vals = []
        with open(p, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    vals.append(float(row[0]))
                except Exception:
                    vals.append(float(row[-1]))
        if len(vals) == 0:
            raise ValueError(f"No numeric counts parsed from CSV: {p}")
        return torch.tensor(vals, dtype=torch.float32)

    else:
        raise ValueError(f"Unsupported counts file extension: {ext}")


def _counts_from_any(x: Union[str, Sequence[float], torch.Tensor]) -> Optional[torch.Tensor]:
    """Accept list/tensor/path and return float tensor. Return None for 'from_loader' etc."""
    if isinstance(x, torch.Tensor):
        return x.detach().float().clone()
    if isinstance(x, (list, tuple)):
        return torch.tensor([float(v) for v in x], dtype=torch.float32)
    if isinstance(x, str):
        # file path or sentinel flag like "from_loader"
        if os.path.sep in x or x.lower().endswith((".json", ".csv", ".txt")):
            return _load_counts_from_path(x)
        if x.lower() in ("auto", "from_loader", "defer", "none"):
            return None
        raise ValueError(
            f"class_counts='{x}' not recognized. Provide a list/tensor, a JSON/CSV path, or 'from_loader'."
        )
    raise ValueError(f"Unsupported class_counts type: {type(x)}")


class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax:
      L = CE( logits + log(counts) , y )
    where counts are per-class sample frequencies collected from the training set.
    """

    def __init__(
        self,
        class_counts: Optional[Union[str, Sequence[float], torch.Tensor]] = None,
        temperature: float = 1.0,
        min_count: float = 1e-6,
        normalize_counts: bool = False,
    ):
        super().__init__()
        # Robust cast for possible YAML strings
        self.temperature = _as_float(temperature, "temperature")
        self.min_count = _as_float(min_count, "min_count")
        self.normalize_counts = bool(normalize_counts)

        # counts_buffer will be registered once we know the vector length
        self.register_buffer("log_counts", None, persistent=False)
        self.needs_counts = False  # flag to let runner know we need loader stats

        if class_counts is None:
            # Defer to runner to inject counts
            self.needs_counts = True
        else:
            cc = _counts_from_any(class_counts)
            if cc is None:
                self.needs_counts = True
            else:
                self._set_counts_tensor(cc)

    @torch.no_grad()
    def _set_counts_tensor(self, counts: torch.Tensor):
        """Validate and set internal log-counts buffer."""
        if counts.dim() != 1:
            counts = counts.view(-1)
        # Optional normalize to sum=1 then rescale by total n (no-op; here just clamp)
        if self.normalize_counts:
            s = counts.sum().clamp_min(1.0)
            counts = counts / s * s
        counts = counts.clamp_min(self.min_count)
        logc = counts.log()
        self.register_buffer("log_counts", logc, persistent=False)
        self.needs_counts = False

    @torch.no_grad()
    def set_class_counts(self, counts: Union[Sequence[float], torch.Tensor]):
        """Public setter used by the runner once it has loader stats."""
        if not isinstance(counts, torch.Tensor):
            counts = torch.tensor([float(v) for v in counts], dtype=torch.float32)
        self._set_counts_tensor(counts)

    def forward(self, logits, y, features=None, classifier=None):
        """
        Args:
            logits: FloatTensor (B, C)
            y:      LongTensor (B,)
            features, classifier: unused (kept for runner API)
        """
        if self.log_counts is None:
            raise RuntimeError(
                "BalancedSoftmaxLoss: class_counts not set yet. "
                "Please call set_class_counts() before the first forward, "
                "or pass class_counts in create_loss()."
            )
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # add log-counts to logits (broadcast across batch)
        if self.log_counts.device != logits.device:
            logc = self.log_counts.to(logits.device)
        else:
            logc = self.log_counts

        logits_adj = logits + logc.unsqueeze(0)
        return F.cross_entropy(logits_adj, y, reduction="mean")


def create_loss(
    class_counts: Optional[Union[str, Sequence[float], torch.Tensor]] = None,
    temperature: float = 1.0,
    min_count: float = 1e-6,
    normalize_counts: bool = False,
    **kwargs
):
    """
    Factory for YAML. Examples:
      loss_params:
        class_counts: "from_loader"   # let runner compute from data
        temperature: 1.0
        min_count: 1e-6
    or:
      loss_params:
        class_counts: [1200, 130, 55] # direct vector
    or:
      loss_params:
        class_counts: "/path/to/counts.json"  # JSON/CSV with counts
    """
    print("Loading BalancedSoftmaxLoss.")
    return BalancedSoftmaxLoss(
        class_counts=class_counts,
        temperature=temperature,
        min_count=min_count,
        normalize_counts=normalize_counts,
    )
