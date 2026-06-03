# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, Dict
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyGuidedSupCMSLoss(nn.Module):


    def __init__(
        self,
        feat_dim: int = 256,
        num_classes: Optional[int] = None,
        tau_ms: float = 0.2,
        tau_nce: float = 0.07,
        use_queue: bool = False,
        queue_size: int = 32768,
        dim: int = 256,
        cosine_bank: bool = True,
        class_freq_json: Optional[str] = None,
        freq_gamma: float = 0.5,
        proto_momentum: float = 0.9,
        energy_weight: float = 1.0,
        temperature: Optional[float] = None,
        # NEW: strength of logit-adjust; can be passed via loss_params["logit_adjust_beta"]
        logit_adjust_beta: float = 0.0,
        # NEW: strength of tail prototype extra update; >0 increases tail movement
        tail_update_gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        # Feature dimension (contrastive space)
        self.feat_dim = int(feat_dim)
        # Temperature for supervised contrastive logits
        self.tau = float(temperature if temperature is not None else tau_nce)
        # Bandwidth for mean-shift weights
        self.tau_ms = float(tau_ms)
        # EMA base momentum
        self.proto_momentum = float(proto_momentum)
        # Energy-guided weighting strength
        self.energy_weight = float(energy_weight)
        self.freq_gamma = float(freq_gamma)
        # Logit-adjust strength; 0.0 means disabled
        self.logit_adjust_beta = float(logit_adjust_beta)
        # Tail prototype update scaling
        self.tail_update_gamma = float(tail_update_gamma)

        # ---------- Class frequencies / weights ----------
        if class_freq_json is not None:
            with open(class_freq_json, "r") as f:
                freq_dict = json.load(f)
            # Expect a mapping "0" -> {...}, "1" -> {...}, ...
            class_keys = sorted(freq_dict.keys(), key=lambda x: int(x))
            counts = []
            freqs = []
            for k in class_keys:
                v = freq_dict[k]
                # Be robust: try num_samples then fallback to freq*total
                num = v.get("num_samples", None)
                if num is None and "freq" in v:
                    # frequency is relative; we only care about relative scales
                    num = v["freq"]
                counts.append(float(num))
                freqs.append(float(v.get("freq", 1.0)))
            counts_tensor = torch.tensor(counts, dtype=torch.float32)
            freq_tensor = torch.tensor(freqs, dtype=torch.float32)
            inferred_num_classes = counts_tensor.numel()
            if num_classes is None:
                num_classes = inferred_num_classes
        else:
            # No frequency JSON; use uniform counts
            if num_classes is None:
                raise ValueError(
                    "num_classes must be provided when class_freq_json is None "
                    "for EnergyGuidedSupCMSLoss."
                )
            counts_tensor = torch.ones(num_classes, dtype=torch.float32)
            freq_tensor = counts_tensor / counts_tensor.sum()

        self.num_classes = int(num_classes)

        # Resize tensors if JSON had more entries than num_classes
        if counts_tensor.numel() != self.num_classes:
            counts_tensor = counts_tensor[: self.num_classes]
        if freq_tensor.numel() != self.num_classes:
            freq_tensor = freq_tensor[: self.num_classes]

        # Normalize frequency to sum 1 for convenience
        freq_tensor = freq_tensor / freq_tensor.sum().clamp_min(1e-12)
        self.register_buffer("class_counts", counts_tensor)
        self.register_buffer("class_freq", freq_tensor)

        # ---------- Prototype initialization ----------
        # Prototypes live on the unit sphere; start from random normal.
        proto = torch.randn(self.num_classes, self.feat_dim)
        proto = F.normalize(proto, dim=1)
        self.register_buffer("prototypes", proto)

        # ---------- Per-class loss weight (energy-guided) ----------
        # class_weight[c] ¡Ø (1 / freq_c)^gamma, normalized by mean
        inv_freq = 1.0 / self.class_freq.clamp_min(1e-12)
        weight_raw = inv_freq.pow(self.freq_gamma)
        weight_raw = weight_raw / weight_raw.mean().clamp_min(1e-12)
        self.register_buffer("class_weight", weight_raw)

        # ---------- Per-class prototype momentum ----------
        # Use class_weight to derive w_norm in [0, 1].
        # We set:
        #   - higher w_norm (tail)   -> larger momentum (more stable)
        #   - lower w_norm (head)   -> smaller momentum (more adaptive)
        with torch.no_grad():
            w = self.class_weight
            w_norm = (w - w.min()) / (w.max() - w.min() + 1e-12)
            proto_m = self.proto_momentum + 0.2 * w_norm
            proto_m = proto_m.clamp(0.0, 0.999)
        self.register_buffer("class_proto_momentum", proto_m)

        # ---------- Per-class prototype update scale (tail moves more) ----------
        # proto_update_scale[c] ¡Ø (1 / freq_c)^tail_update_gamma, normalized
        with torch.no_grad():
            upd_raw = inv_freq.pow(self.tail_update_gamma)
            upd_raw = upd_raw / upd_raw.mean().clamp_min(1e-12)
        self.register_buffer("proto_update_scale", upd_raw)

    @staticmethod
    def _l2_normalize(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """L2-normalize along feature dimension."""
        return z / (z.norm(p=2, dim=1, keepdim=True).clamp_min(eps))

    @torch.no_grad()
    def _mean_shift_update_prototypes(self, z: torch.Tensor, labels: torch.Tensor):

        if z.numel() == 0:
            return

        B, D = z.shape
        device = z.device
        # Ensure buffers live on the same device
        self.prototypes = self.prototypes.to(device)
        self.class_proto_momentum = self.class_proto_momentum.to(device)
        self.proto_update_scale = self.proto_update_scale.to(device)

        for c in range(self.num_classes):
            mask = (labels == c)
            if not mask.any():
                continue
            z_c = z[mask]  # (B_c, D)

            # Current prototype
            p_c = self.prototypes[c : c + 1]  # (1, D)

            # Cosine similarity with current prototype p_c
            sim = (z_c * p_c).sum(dim=1)      # (B_c,)

            # Mean-shift kernel weights; higher sim => larger weight
            w = torch.exp(sim / self.tau_ms)
            w = w / w.sum().clamp_min(1e-12)  # normalize over the class subset

            # Weighted center
            m_c = (w.unsqueeze(1) * z_c).sum(dim=0, keepdim=True)  # (1, D)
            m_c = F.normalize(m_c, dim=1)

            # EMA update with per-class momentum AND tail-biased update scale
            mom = float(self.class_proto_momentum[c].item())       # in [0, ~0.999]
            scale = float(self.proto_update_scale[c].item())       # tail > 1, head < 1 (on average ~1)

            # Effective update factor alpha_c:
            #   larger for tails, smaller for heads
            alpha = (1.0 - mom) * scale
            # Avoid exploding updates
            alpha = max(0.0, min(alpha, 1.0))

            new_p = (1.0 - alpha) * p_c + alpha * m_c
            new_p = F.normalize(new_p, dim=1)
            self.prototypes[c : c + 1] = new_p

    def forward(self, feats: torch.Tensor, labels=None, extras: Optional[Dict] = None):

        if labels is None:
            raise ValueError("EnergyGuidedSupCMSLoss requires labels for supervised contrastive learning.")

        if feats.ndim != 2:
            raise ValueError(f"Expected feats shape (B, D), got {feats.shape}")
        if feats.size(1) != self.feat_dim:
            raise ValueError(
                f"feat_dim mismatch: got feats with D={feats.size(1)}, "
                f"but loss was initialized with feat_dim={self.feat_dim}"
            )

        device = feats.device

        # Ensure prototypes are on the same device and normalized
        self.prototypes = F.normalize(self.prototypes.to(device), dim=1)

        # Optional safety: re-normalize feats in case upstream did not
        feats = self._l2_normalize(feats)

        # ----------------- 1) Mean-shift prototype update (no grad) -----------------
        with torch.no_grad():
            self._mean_shift_update_prototypes(feats.detach(), labels.detach())

        # ----------------- 2) Supervised prototype contrastive logits -----------------
        # logits_{i,c} = sim(z_i, p_c) / tau
        logits = torch.matmul(feats, self.prototypes.t())  # (B, C)
        logits = logits / self.tau

        # --------- 2.1) Logit adjustment based on class frequency ----------
        #       logits_{i,c} <- logits_{i,c} - beta * log(freq_c)
        # This suppresses head classes (larger freq) and relatively boosts tails.
        if self.logit_adjust_beta > 0.0:
            log_freq = torch.log(self.class_freq.clamp_min(1e-12)).to(device)  # (C,)
            bias = -self.logit_adjust_beta * log_freq                          # (C,)
            logits = logits + bias                                             # broadcast to (B, C)

        # Per-sample base loss
        ce = F.cross_entropy(logits, labels, reduction="none")  # (B,)

        # ----------------- 3) Frequency / energy guided reweighting -----------------
        if self.energy_weight > 0.0:
            # Map labels -> per-sample class weight
            # class_weight[c] already encodes (1 / freq_c)^gamma normalized by mean.
            class_w = self.class_weight.to(device)
            w = class_w[labels]  # (B,)
            # Optionally scale by energy_weight; and normalize mean to 1.0
            #   w = 1 + energy_weight * (w - 1)
            # so energy_weight controls how strongly we emphasize tails.
            w = 1.0 + self.energy_weight * (w - 1.0)
            loss = (ce * w).mean()
        else:
            loss = ce.mean()

        return loss


def create_loss(
    k=20,
    tau_ms=0.2,
    tau_nce=0.07,
    use_queue=False,
    queue_size=32768,
    feat_dim=256,
    dim=None,
    cosine_bank=True,
    num_classes: Optional[int] = None,
    class_freq_json: Optional[str] = None,
    freq_gamma: float = 0.5,
    proto_momentum: float = 0.9,
    energy_weight: float = 1.0,
    temperature: Optional[float] = None,
    logit_adjust_beta: float = 0.0,
    tail_update_gamma: float = 1.0,
    **kwargs,
):

    return EnergyGuidedSupCMSLoss(
        feat_dim=feat_dim if dim is None else dim,
        num_classes=num_classes,
        tau_ms=tau_ms,
        tau_nce=tau_nce,
        use_queue=use_queue,
        queue_size=queue_size,
        dim=dim if dim is not None else feat_dim,
        cosine_bank=cosine_bank,
        class_freq_json=class_freq_json,
        freq_gamma=freq_gamma,
        proto_momentum=proto_momentum,
        energy_weight=energy_weight,
        temperature=temperature,
        logit_adjust_beta=logit_adjust_beta,
        tail_update_gamma=tail_update_gamma,
        **kwargs,
    )
