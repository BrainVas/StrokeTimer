#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization for NCCT pipeline (after training) with a single model.

Model:
- feat_model (e.g. ResNeXt3DFeature)
- classifier (e.g. DotProduct / Causal Norm / MARC)

Functions:
1) Load test data and one trained checkpoint (ONLY from ncct.csv_path in the cfg).
2) For each case, compute 3D Grad-CAM++ regardless of whether it is
   correctly classified.
   - Save brain-masked 3D heatmap as .npy.
   - Save 6x8 slice grid overlay PNG.
   - Save the original CT volume and model input as .npy.
3) Extract logits on the full test set.
4) Compute classification metrics (accuracy, precision, recall, F1-micro/macro)
   and ROC curves (one-vs-rest) on the base test set, and save plots.
   Also compute Precision-Recall curves and AP (micro/macro & per-class).
   Additionally, compute:
     - Sensitivity and Specificity (multiclass one-vs-rest).
     - A p-value vs random guessing (binomial test).

Usage:
  python -m stroketimer.visualize \
      --cfg configs/stroketimer_best.yaml \
      --ckpt checkpoints/best/final_model_checkpoint.pth \
      --out_dir outputs/visualization \
      --gradcam-target pred
"""

from __future__ import annotations
import os
import argparse
import csv
import inspect
from typing import Dict, List, Tuple, Any

import yaml
from yaml import Loader

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# sklearn metrics for evaluation, ROC, and PR
try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        classification_report,
        roc_curve,
        auc,
        roc_auc_score,
        precision_recall_curve,
        average_precision_score,
    )
except ImportError as e:
    raise ImportError(
        "scikit-learn is required for evaluation metrics, ROC, and PR curve plotting. "
        "Please install it via 'pip install scikit-learn'."
    ) from e

# SciPy for binomial test (p-value vs random guessing)
try:
    from scipy.stats import binomtest
except ImportError as e:
    raise ImportError(
        "SciPy is required for the binomial significance test (p-value vs random guessing). "
        "Please install it via 'pip install scipy'."
    ) from e

from stroketimer.data import dataloader as dl_mod
from stroketimer.utils import get_value, resolve_repo_path  # imported for completeness, not strictly required


# -------------------------------------------------------------------------
# Generic import helper
# -------------------------------------------------------------------------
def source_import(file_path: str):
    """Import a python file by path and return the loaded module."""
    from stroketimer.utils import source_import as _source_import
    return _source_import(file_path)


def _safe_path_token(value: Any) -> str:
    """Return a filesystem-friendly token for case-level output folders."""
    token = str(value).strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in token)


def normalize_config(config: Dict) -> Dict:
    for criterion in config.get("criterions", {}).values():
        params = criterion.get("loss_params", {})
        if "class_freq_path" in params:
            params["class_freq_json"] = params.pop("class_freq_path")
        if "gamma_freq" in params:
            params["freq_gamma"] = params.pop("gamma_freq")
    return config


def apply_env_overrides(config: Dict) -> Dict:
    training_opt = config.setdefault("training_opt", {})
    if os.environ.get("DATA_ROOT"):
        training_opt["data_root"] = os.environ["DATA_ROOT"]
    if os.environ.get("CSV_PATH"):
        config.setdefault("ncct", {})["csv_path"] = os.environ["CSV_PATH"]
    if os.environ.get("OUTPUT_DIR"):
        training_opt["log_dir"] = os.environ["OUTPUT_DIR"]
    if os.environ.get("CLASS_FREQ_JSON"):
        freq_path = os.environ["CLASS_FREQ_JSON"]
        for criterion in config.get("criterions", {}).values():
            params = criterion.get("loss_params", {})
            if "class_freq_json" in params:
                params["class_freq_json"] = freq_path
            if "class_counts" in params:
                params["class_counts"] = freq_path
    return config


# -------------------------------------------------------------------------
# Build networks and load checkpoint (single model)
# -------------------------------------------------------------------------
def build_networks_from_cfg(
    config: Dict,
    device: torch.device,
    ckpt_path: str,
) -> Dict[str, nn.Module]:
    """
    Build feat_model and classifier (if defined in config["networks"]),
    without DataParallel, to make Grad-CAM hooks easier.

    Automatically strips off "module." prefix in checkpoint keys.
    """
    networks_defs = config["networks"]
    networks: Dict[str, nn.Module] = {}

    # 1) build networks
    for key, val in networks_defs.items():
        create_fn = source_import(val["def_file"]).create_model
        model_args = dict(val["params"])

        # keep only arguments that create_model actually accepts
        try:
            sig = inspect.signature(create_fn)
            allowed = set(sig.parameters.keys())
            clean_args = {k: v for k, v in model_args.items() if k in allowed}
        except (TypeError, ValueError):
            clean_args = model_args

        print(f"[build_networks] Creating '{key}' from {val['def_file']} with args={clean_args}")
        net = create_fn(**clean_args)
        net.to(device)
        networks[key] = net

    # 2) load checkpoint
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    print(f"[build_networks] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict_best" in ckpt:
        state = ckpt["state_dict_best"]
    else:
        state = ckpt  # fallback: raw state dict

    for key, net in networks.items():
        if key not in state:
            print(f"[build_networks] Warning: no weights for '{key}' in checkpoint, skip.")
            continue
        sd_ckpt = state[key]

        # strip "module." prefix (used in DataParallel)
        new_sd = {}
        for k, v in sd_ckpt.items():
            k2 = k[7:] if k.startswith("module.") else k
            new_sd[k2] = v

        sd_model = net.state_dict()
        loadable = {}
        for k, v in new_sd.items():
            if k in sd_model and sd_model[k].shape == v.shape:
                loadable[k] = v
        sd_model.update(loadable)
        net.load_state_dict(sd_model)
        print(f"[build_networks] Loaded {len(loadable)} params into '{key}'")

    return networks


# -------------------------------------------------------------------------
# Build test dataloader
# -------------------------------------------------------------------------
def build_test_loader(config: Dict, args_seed: int):
    """
    Build only the test dataloader, matching the training main script.
    Uses whatever 'ncct.csv_path' is currently set in the config if dataset == 'ncct_dataset'.
    """
    training_opt = config["training_opt"]
    dataset = training_opt["dataset"]
    root_from_cfg = training_opt.get("data_root", None)
    fallback_key = dataset.rstrip("_LT")

    LEGACY_ROOTS = {
        "ImageNet": "/media/Cygnus/haoc/longtail/ImageNet_LT",
        "Places": "/media/Cygnus/haoc/longtail/Places_LT",
        "iNaturalist18": "/media/Cygnus/haoc/longtail/iNaturalist18",
        "CIFAR10": "./dataset/CIFAR10",
        "CIFAR100": "./dataset/CIFAR100",
    }
    pretty_root = root_from_cfg if root_from_cfg is not None else LEGACY_ROOTS.get(fallback_key, "./")
    print("[build_test_loader] data_root =", pretty_root)
    print("[build_test_loader] dataset   =", dataset)

    if dataset == "ncct_dataset":
        ncct_cfg = config.get("ncct", {})
        loader = dl_mod.load_data(
            data_root=pretty_root,
            dataset=dataset,
            phase="test",
            batch_size=training_opt["batch_size"],
            sampler_dic=None,
            num_workers=training_opt.get("num_workers", 4),
            shuffle=False,
            ncct_csv_path=ncct_cfg.get("csv_path"),
            ncct_transform=None,
            ncct_fix_depth=ncct_cfg.get("fix_depth", True),
            ncct_zscore=ncct_cfg.get("zscore", True),
            ncct_split_ratios=tuple(ncct_cfg.get("split_ratios", (0.7, 0.15, 0.15))),
            ncct_seed=ncct_cfg.get("seed", args_seed),
            ncct_balance_eval=ncct_cfg.get("balance_eval", True),
        )
        num_classes = int(training_opt["num_classes"])
    else:
        # fallback for 2D datasets
        loader = dl_mod.load_data(
            data_root=pretty_root,
            dataset=dataset,
            phase="test",
            batch_size=training_opt["batch_size"],
            sampler_dic=None,
            num_workers=training_opt.get("num_workers", 4),
            shuffle=False,
        )
        num_classes = loader.dataset.get_num_classes()

    return loader, num_classes


# -------------------------------------------------------------------------
# Classification metrics + ROC + PR + sensitivity/specificity + p-value
# -------------------------------------------------------------------------
def _softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    x = logits - np.max(logits, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def _sens_spec_multiclass_ovr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Tuple[List[Tuple[int, float, float, int, int, int, int]], float, float]:
    """
    Compute one-vs-rest sensitivity and specificity for each class in multiclass classification.

    For class c:
      TP = (y_true == c) & (y_pred == c)
      FN = (y_true == c) & (y_pred != c)
      FP = (y_true != c) & (y_pred == c)
      TN = (y_true != c) & (y_pred != c)

    Returns:
        per_class_detail: list of (c, sen, spe, tp, fn, tn, fp)
        macro_sen: mean SEN over classes (ignoring NaN)
        macro_spe: mean SPE over classes (ignoring NaN)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    per_class_detail: List[Tuple[int, float, float, int, int, int, int]] = []
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        tn = int(((y_true != c) & (y_pred != c)).sum())

        sen = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spe = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        per_class_detail.append((c, sen, spe, tp, fn, tn, fp))

    sens = [x[1] for x in per_class_detail if not np.isnan(x[1])]
    spes = [x[2] for x in per_class_detail if not np.isnan(x[2])]
    macro_sen = float(np.mean(sens)) if len(sens) > 0 else float("nan")
    macro_spe = float(np.mean(spes)) if len(spes) > 0 else float("nan")
    return per_class_detail, macro_sen, macro_spe


def compute_and_save_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    model_name: str,
    out_dir: str,
):
    """
    Compute accuracy, precision, recall, F1 (micro & macro), ROC/AUC,
    and Precision-Recall (PR) curves + AP (micro/macro & per-class)
    for a multi-class classifier (one-vs-rest), and save results.

    Additionally:
      - Compute one-vs-rest sensitivity and specificity per class + macro averages.
      - Compute a binomial p-value vs uniform random guessing (chance = 1/num_classes),
        testing H0: accuracy == chance, H1: accuracy > chance.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Softmax probabilities and predictions
    probs = _softmax_np(logits)  # (N, C)
    preds = np.argmax(probs, axis=1)  # (N,)

    # Basic metrics
    acc = accuracy_score(labels, preds)
    N = labels.shape[0]

    # ------- Sensitivity / Specificity -------
    per_detail, macro_sen, macro_spe = _sens_spec_multiclass_ovr(labels, preds, num_classes)
    per_class_sens_spe = [(c, sen, spe) for (c, sen, spe, tp, fn, tn, fp) in per_detail]

    # ------- p-value vs random guessing (binomial test) -------
    num_correct = int((preds == labels).sum())
    p_chance = 1.0 / float(num_classes)
    p_value_vs_random = binomtest(
        num_correct,
        N,
        p=p_chance,
        alternative="greater",
    ).pvalue

    # Precision/Recall/F1 (macro/micro)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )

    report = classification_report(labels, preds, digits=4)

    # One-hot for ROC / PR
    y_onehot = np.zeros((N, num_classes), dtype=np.float32)
    y_onehot[np.arange(N), labels] = 1.0

    # ---- ROC AUC (macro/micro, OVR) ----
    try:
        auc_macro_ovr = roc_auc_score(y_onehot, probs, multi_class="ovr", average="macro")
        auc_micro_ovr = roc_auc_score(y_onehot, probs, multi_class="ovr", average="micro")
    except ValueError:
        auc_macro_ovr = float("nan")
        auc_micro_ovr = float("nan")

    # Per-class ROC curves
    plt.figure(figsize=(6, 6))
    per_class_auc = []

    for c in range(num_classes):
        if y_onehot[:, c].sum() == 0 or np.all(y_onehot[:, c] == 1.0):
            print(f"[metrics-{model_name}] Skip ROC for class {c} (no pos/neg samples).")
            continue

        fpr, tpr, _ = roc_curve(y_onehot[:, c], probs[:, c])
        roc_auc_c = auc(fpr, tpr)
        per_class_auc.append((c, roc_auc_c))

        plt.plot(fpr, tpr, lw=1.5, label=f"class {c} (AUC={roc_auc_c:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.0)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (one-vs-rest) - {model_name}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    roc_png = os.path.join(out_dir, f"roc_ovr_{model_name}.png")
    plt.savefig(roc_png, dpi=200)
    plt.close()
    print(f"[metrics-{model_name}] Saved ROC plot to {roc_png}")

    # ---- Precision-Recall curves + Average Precision (AP) ----
    try:
        ap_macro_ovr = average_precision_score(y_onehot, probs, average="macro")
    except ValueError:
        ap_macro_ovr = float("nan")

    try:
        ap_micro_ovr = average_precision_score(y_onehot, probs, average="micro")
    except ValueError:
        ap_micro_ovr = float("nan")

    plt.figure(figsize=(6, 6))
    per_class_ap = []

    for c in range(num_classes):
        if y_onehot[:, c].sum() == 0 or np.all(y_onehot[:, c] == 1.0):
            print(f"[metrics-{model_name}] Skip PR for class {c} (no pos/neg samples).")
            continue

        precision_c, recall_c, _ = precision_recall_curve(y_onehot[:, c], probs[:, c])
        try:
            ap_c = average_precision_score(y_onehot[:, c], probs[:, c])
        except ValueError:
            ap_c = float("nan")

        per_class_ap.append((c, ap_c))
        plt.plot(recall_c, precision_c, lw=1.5, label=f"class {c} (AP={ap_c:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall (one-vs-rest) - {model_name}")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()

    pr_png = os.path.join(out_dir, f"pr_ovr_{model_name}.png")
    plt.savefig(pr_png, dpi=200)
    plt.close()
    print(f"[metrics-{model_name}] Saved PR plot to {pr_png}")

    # ---- Save metrics to txt ----
    metrics_txt = os.path.join(out_dir, f"classification_metrics_{model_name}.txt")
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Num samples: {N}\n")
        f.write(f"Num classes: {num_classes}\n\n")

        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(
            f"P-value vs random guessing (chance={p_chance:.4f}, "
            f"binomial test, H1: accuracy>chance): {p_value_vs_random:.4e}\n\n"
        )

        # Sensitivity/Specificity summary
        f.write("Multi-class SEN/SPE (one-vs-rest):\n")
        f.write(f"  Macro Sensitivity (OVR): {macro_sen:.4f}\n")
        f.write(f"  Macro Specificity (OVR): {macro_spe:.4f}\n")
        f.write("  Per-class:\n")
        for c, sen, spe in per_class_sens_spe:
            f.write(f"    class {c}: SEN={sen:.4f}, SPE={spe:.4f}\n")
        f.write("\n")

        # Standard macro/micro metrics
        f.write(f"Macro Precision: {prec_macro:.4f}\n")
        f.write(f"Macro Recall:    {rec_macro:.4f}\n")
        f.write(f"Macro F1:        {f1_macro:.4f}\n")
        f.write(f"Micro Precision: {prec_micro:.4f}\n")
        f.write(f"Micro Recall:    {rec_micro:.4f}\n")
        f.write(f"Micro F1:        {f1_micro:.4f}\n\n")

        f.write(f"Macro AUC (OVR-ROC): {auc_macro_ovr:.4f}\n")
        f.write(f"Micro AUC (OVR-ROC): {auc_micro_ovr:.4f}\n\n")
        f.write(f"Macro AP  (OVR-PR):  {ap_macro_ovr:.4f}\n")
        f.write(f"Micro AP  (OVR-PR):  {ap_micro_ovr:.4f}\n\n")

        if per_class_auc:
            f.write("Per-class AUC (ROC, one-vs-rest):\n")
            for c, auc_c in per_class_auc:
                f.write(f"  class {c}: {auc_c:.4f}\n")
            f.write("\n")

        if per_class_ap:
            f.write("Per-class AP (PR, one-vs-rest):\n")
            for c, ap_c in per_class_ap:
                f.write(f"  class {c}: {ap_c:.4f}\n")
            f.write("\n")

        f.write("Classification report (per-class precision/recall/F1):\n")
        f.write(report)

    print(f"[metrics-{model_name}] Saved metrics to {metrics_txt}")
    print(
        f"[metrics-{model_name}] Accuracy={acc:.4f}, "
        f"P-value vs random={p_value_vs_random:.4e}, "
        f"Macro-F1={f1_macro:.4f}, Micro-F1={f1_micro:.4f}"
    )
    print(f"[metrics-{model_name}] Macro-SEN(OVR)={macro_sen:.4f}, Macro-SPE(OVR)={macro_spe:.4f}")

    print(
        f"[metrics-{model_name}] Macro AUC (OVR-ROC)={auc_macro_ovr:.4f}, "
        f"Micro AUC (OVR-ROC)={auc_micro_ovr:.4f}"
    )
    print(
        f"[metrics-{model_name}] Macro AP  (OVR-PR)={ap_macro_ovr:.4f}, "
        f"Micro AP  (OVR-PR)={ap_micro_ovr:.4f}"
    )


# -------------------------------------------------------------------------
# Grad-CAM++ helpers
# -------------------------------------------------------------------------
def pick_target_layer(net: nn.Module) -> nn.Module:
    """Pick a reasonable target layer for Grad-CAM++."""
    if isinstance(net, nn.DataParallel):
        net = net.module

    # Direct ResNet/ResNeXt-style
    if hasattr(net, "layer4"):
        return net.layer4

    # Some models might have a backbone.layer4
    if hasattr(net, "backbone") and hasattr(net.backbone, "layer4"):
        return net.backbone.layer4

    raise RuntimeError(
        f"Cannot find a suitable target layer on {type(net)}. "
        f"Please adapt pick_target_layer() to your backbone."
    )


def _call_classifier_generic(
    classifier: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Call classifier in a robust way:
      - Try classifier(features, labels, embed)
      - Else fallback to classifier(features)

    Return logits tensor of shape (B, C).
    """
    # Default dummy labels if classifier signature requires it
    if labels is None:
        labels = torch.zeros(features.size(0), dtype=torch.long, device=features.device)

    try:
        out = classifier(features, labels, None)
    except TypeError:
        out = classifier(features)

    if isinstance(out, (list, tuple)):
        logits = out[0]
    else:
        logits = out
    return logits


def compute_gradcam_pp_3d(
    feat_model: nn.Module,
    classifier: nn.Module,
    x: torch.Tensor,
    class_idx: int,
    target_layer: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """
    Compute 3D Grad-CAM++ for a single sample x with shape (1, 1, D, H, W).

    Assumes:
      - feat_model(x) -> (B, feat_dim) or (B, feat_dim, ...)
      - classifier(features) or classifier(features, labels, embed)
    """
    feat_model.eval()
    classifier.eval()

    activations = None
    gradients = None

    def fwd_hook(module, inp, out):
        nonlocal activations
        activations = out

    def bwd_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

    x = x.to(device, non_blocking=True)
    feat_model.zero_grad(set_to_none=True)
    classifier.zero_grad(set_to_none=True)

    # Forward with gradients enabled
    with torch.set_grad_enabled(True):
        out_feat = feat_model(x)
        if isinstance(out_feat, (list, tuple)):
            features = out_feat[0]
        else:
            features = out_feat

        logits = _call_classifier_generic(classifier, features, labels=None)
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

    A = activations.detach()[0]  # (C, D', H', W')
    G = gradients.detach()[0]    # (C, D', H', W')

    # Grad-CAM++ weights in 3D
    G2 = G * G
    G3 = G2 * G

    sum_A = A.sum(dim=(1, 2, 3), keepdim=True)
    sum_G2 = G2.sum(dim=(1, 2, 3), keepdim=True)
    sum_G3 = G3.sum(dim=(1, 2, 3), keepdim=True)

    eps = 1e-8
    alpha_num = sum_G2
    alpha_den = 2.0 * sum_G2 + sum_A * sum_G3
    alpha_den = torch.where(alpha_den != 0.0, alpha_den, torch.ones_like(alpha_den))
    alpha = alpha_num / (alpha_den + eps)  # (C,1,1,1)

    weights = (alpha * G).sum(dim=(1, 2, 3))  # (C,)
    weights = F.relu(weights)

    cam = (weights.view(-1, 1, 1, 1) * A).sum(dim=0)  # (D', H', W')
    cam = F.relu(cam)

    cam = cam.unsqueeze(0).unsqueeze(0)  # (1,1,D',H',W')
    D, H, W = x.shape[2:]
    cam = F.interpolate(
        cam,
        size=(D, H, W),
        mode="trilinear",
        align_corners=False,
    )[0, 0]

    cam_np = cam.cpu().numpy()
    cam_np -= cam_np.min()
    if cam_np.max() > eps:
        cam_np /= cam_np.max()
    else:
        cam_np[:] = 0.0

    handle_fwd.remove()
    handle_bwd.remove()

    return cam_np  # (D, H, W)


# -------------------------------------------------------------------------
# Visualization helpers: slice grid overlays
# -------------------------------------------------------------------------
def save_slice_grid_heatmap(
    volume: np.ndarray,
    heatmap: np.ndarray,
    out_png: str,
    title: str = "",
    brain_threshold: float = 0.0,
):
    """
    Draw a 6x8 grid of slices (48 slices total). Each subplot shows
    the CT slice (grayscale) with the Grad-CAM heatmap overlay.

    Only the brain region is colored, using a simple threshold on the CT volume.

    Args:
        volume: (D, H, W), z-scored CT volume.
        heatmap: (D, H, W), values in [0, 1].
        brain_threshold: threshold for defining brain mask. Default 0.0
    """
    assert volume.shape == heatmap.shape
    D, H, W = volume.shape
    assert D == 48, "Current grid assumes D=48 slices."

    nrows, ncols = 6, 8
    assert nrows * ncols == D

    brain_mask = volume > brain_threshold  # (D, H, W)

    plt.figure(figsize=(ncols * 2, nrows * 2))
    for d in range(D):
        ax = plt.subplot(nrows, ncols, d + 1)
        ax.imshow(volume[d], cmap="gray", aspect="auto")

        hm_slice = heatmap[d].copy()
        mask_slice = brain_mask[d]
        hm_slice[~mask_slice] = 0.0

        # Normalize heatmap inside brain mask for better visibility
        if mask_slice.any():
            vals = hm_slice[mask_slice]
            vmax = vals.max()
            if vmax > 0:
                hm_slice[mask_slice] = vals / vmax

        ax.imshow(hm_slice, cmap="jet", alpha=0.4, aspect="auto")
        ax.axis("off")
        ax.set_title(f"{d}", fontsize=6)

    if title:
        plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Training YAML config")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save visualizations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradcam-target",
        type=str,
        choices=("pred", "label"),
        default="pred",
        help="Use the predicted class or ground-truth label as the Grad-CAM target for each case.",
    )
    args = parser.parse_args()

    assert os.path.isfile(args.cfg), f"Config not found: {args.cfg}"

    with open(args.cfg, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=Loader)
    config = apply_env_overrides(normalize_config(config))

    training_opt = config.setdefault("training_opt", {})
    training_opt.setdefault("lambda_proto", 0.0)
    training_opt.setdefault("lambda_ortho", 0.0)

    if args.out_dir is None:
        args.out_dir = os.path.join(training_opt["log_dir"], "viz_single_model")
    os.makedirs(args.out_dir, exist_ok=True)
    args.ckpt = str(resolve_repo_path(args.ckpt))

    # Reproducibility seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[main] Using device:", device)

    # ------------------------------------------------------------------
    # 1) Test loader
    # ------------------------------------------------------------------
    test_loader, num_classes = build_test_loader(config, args_seed=args.seed)
    print(f"[main] Test set size (base CSV) = {len(test_loader.dataset)}, num_classes = {num_classes}")

    # ------------------------------------------------------------------
    # 2) Networks (feat_model + classifier)
    # ------------------------------------------------------------------
    networks = build_networks_from_cfg(config, device, ckpt_path=args.ckpt)
    assert "feat_model" in networks, "Config must contain 'feat_model'."
    assert "classifier" in networks, "Config must contain 'classifier'."

    feat_model = networks["feat_model"].to(device)
    classifier = networks["classifier"].to(device)
    feat_model.eval()
    classifier.eval()

    # Grad-CAM target layer
    target_layer = pick_target_layer(feat_model)
    print("[main] Grad-CAM target layer =", target_layer)

    # ------------------------------------------------------------------
    # 3) Collect logits and per-sample metadata
    # ------------------------------------------------------------------

    y_list: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    sample_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, (inputs, labels, indices) in enumerate(test_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Feature forward
            out_feat = feat_model(inputs)
            if isinstance(out_feat, (list, tuple)):
                features = out_feat[0]
            else:
                features = out_feat  # (B, feat_dim)

            logits = _call_classifier_generic(classifier, features, labels=None)
            preds = logits.argmax(dim=1)

            all_logits.append(logits.detach().cpu().numpy())
            y_list.append(labels.detach().cpu().numpy())

            if torch.is_tensor(indices):
                batch_indices = indices.detach().cpu().numpy().tolist()
            else:
                batch_indices = list(indices)

            for b in range(inputs.size(0)):
                y = int(labels[b].item())
                pred = int(preds[b].item())
                dataset_idx = int(batch_indices[b])
                sample = {}
                if hasattr(test_loader.dataset, "samples") and dataset_idx < len(test_loader.dataset.samples):
                    sample = test_loader.dataset.samples[dataset_idx]
                sample_rows.append({
                    "dataset_index": dataset_idx,
                    "pid": sample.get("pid", ""),
                    "path": sample.get("path", ""),
                    "label": y,
                    "pred": pred,
                })

            if (batch_idx + 1) % 10 == 0:
                print(f"[main] Processed {batch_idx + 1} / {len(test_loader)} batches")

    # Stack
    y_all = np.concatenate(y_list, axis=0)
    logits_all = np.concatenate(all_logits, axis=0)
    print(f"[main] Collected logits: N = {y_all.shape[0]}")

    # ------------------------------------------------------------------
    # 4) Three-class metrics + ROC + PR + SEN/SPE + p-value
    # ------------------------------------------------------------------
    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    compute_and_save_classification_metrics(
        logits=logits_all,
        labels=y_all,
        num_classes=num_classes,
        model_name="single_model",
        out_dir=metrics_dir,
    )

    probs_all = _softmax_np(logits_all)
    pred_csv = os.path.join(metrics_dir, "predictions_single_model.csv")
    fieldnames = ["dataset_index", "pid", "path", "label", "pred"] + [
        f"prob_class_{c}" for c in range(num_classes)
    ]
    with open(pred_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(sample_rows):
            out_row = dict(row)
            out_row["label"] = int(y_all[i])
            out_row["pred"] = int(np.argmax(probs_all[i]))
            for c in range(num_classes):
                out_row[f"prob_class_{c}"] = float(probs_all[i, c])
            writer.writerow(out_row)
    print(f"[main] Saved per-sample predictions to {pred_csv}")

    sample_rows_by_index = {int(row["dataset_index"]): row for row in sample_rows}

    # ------------------------------------------------------------------
    # 5) 3D Grad-CAM++ per case
    # ------------------------------------------------------------------
    gradcam_dir = os.path.join(args.out_dir, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    gradcam_count = 0
    for batch_idx, (inputs, labels, indices) in enumerate(test_loader):
        if torch.is_tensor(indices):
            batch_indices = indices.detach().cpu().numpy().tolist()
        else:
            batch_indices = list(indices)

        for b in range(inputs.size(0)):
            dataset_idx = int(batch_indices[b])
            row = sample_rows_by_index.get(dataset_idx, {})
            label = int(row.get("label", int(labels[b].item())))
            pred = int(row.get("pred", label))
            target_class = pred if args.gradcam_target == "pred" else label

            pid = _safe_path_token(row.get("pid", f"case_{dataset_idx}"))
            out_case_dir = os.path.join(
                gradcam_dir,
                f"case_{dataset_idx:05d}_{pid}_true{label}_pred{pred}",
            )
            os.makedirs(out_case_dir, exist_ok=True)

            x = inputs[b:b + 1].detach().cpu()
            vol = inputs[b, 0].detach().cpu().numpy()

            print(
                f"[main] Computing Grad-CAM++ for case {dataset_idx} "
                f"(pid={pid}, true={label}, pred={pred}, target={target_class}) ..."
            )

            cam = compute_gradcam_pp_3d(
                feat_model=feat_model,
                classifier=classifier,
                x=x.clone(),
                class_idx=target_class,
                target_layer=target_layer,
                device=device,
            )

            np.save(os.path.join(out_case_dir, f"cam3d_target_class{target_class}.npy"), cam)
            np.save(os.path.join(out_case_dir, "volume_zscore.npy"), vol)
            np.save(os.path.join(out_case_dir, "input_tensor.npy"), x.numpy())

            png_cam = os.path.join(out_case_dir, f"slice_grid_target_class{target_class}.png")
            save_slice_grid_heatmap(
                volume=vol,
                heatmap=cam,
                out_png=png_cam,
                title=f"Grad-CAM++ pid={pid}, true={label}, pred={pred}, target={target_class}",
                brain_threshold=0.0,
            )
            gradcam_count += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"[main] Grad-CAM processed {batch_idx + 1} / {len(test_loader)} batches")

    print(f"[main] Generated Grad-CAM++ outputs for {gradcam_count} cases.")

    print("[main] All visualizations and metrics have been generated.")


if __name__ == "__main__":
    main()
