#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT_BET batch launcher:
- predict / predict3d (+ optional recursive vendor/model/kernel traversal)
- post-prediction resampling BACK to FIXED voxel size (default 48x256x256, i.e. slices x H x W)
- overwrite outputs in-place (no *_fixed artifacts)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

print(sys.argv[0])

# ==== timestamp ====
t = time.localtime()
timeStamp = f"{t.tm_year}{t.tm_mon}{t.tm_mday}_{t.tm_hour}{t.tm_min}{t.tm_sec}"
code_dir = os.getcwd()
sys.path.append(code_dir)

from auggen import AugmentationGenerator as dataGenerator
from model_CT_SS import Unet_CT_SS as genUnet

# -------------------- NIfTI helpers --------------------
_VALID_EXTS = (".nii", ".nii.gz")

def _is_nii(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")

def _has_nii(folder: Path) -> bool:
    try:
        for p in folder.iterdir():
            if p.is_file() and _is_nii(p):
                return True
    except Exception:
        pass
    return False

def _list_level_dirs(root: Path, depth: int) -> List[Path]:
    if depth < 1:
        return [root]
    pattern = "/".join(["*"] * depth)   # e.g. "*/*/*"
    return [p for p in root.glob(pattern) if p.is_dir()]

# -------------------- resample-to-FIXED size (overwrite) --------------------
try:
    import SimpleITK as sitk
except Exception:
    sitk = None  # We'll enforce presence if resample_back is requested

def _strip_mask_suffix(stem: str) -> str:
    s = stem
    for suf in ("_mask", "-mask", ".mask"):
        if s.lower().endswith(suf):
            return s[: -len(suf)]
    return s

def _find_ref_for_output(out_p: Path, img_root: Path) -> Optional[Path]:
    """
    Find input NIfTI corresponding to a predicted output.
    Strategy: exact same name -> strip mask suffix then same name -> fuzzy stem match.
    """
    exact = img_root / out_p.name
    if exact.exists() and _is_nii(exact):
        return exact
    base = _strip_mask_suffix(out_p.stem)
    for ext in (".nii.gz", ".nii"):
        cand = img_root / f"{base}{ext}"
        if cand.exists():
            return cand
    key = base.lower()
    for p in img_root.rglob("*"):
        if p.is_file() and _is_nii(p) and key in p.stem.lower():
            return p
    return None

def _atomic_write(img: "sitk.Image", target: Path):
    tmp = target.with_name(target.name + ".tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(tmp))
    os.replace(str(tmp), str(target))  # atomic replace

def _parse_target_size(s: str) -> Tuple[int, int, int]:
    """
    Parse CLI string like '48,256,256' into (slices, H, W).
    """
    parts = [int(x) for x in s.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise ValueError("--target_size must be like '48,256,256' (slices,H,W)")
    slices, H, W = parts
    if slices <= 0 or H <= 0 or W <= 0:
        raise ValueError("All dims in --target_size must be > 0")
    return slices, H, W

def _resample_overwrite_to_fixed(out_p: Path, ref_p: Path, target_slicesHW: Tuple[int, int, int]) -> bool:
    """
    Resample 'out_p' to FIXED voxel size (slices,H,W) using ref geometry for direction/origin/FOV.
    Overwrite 'out_p' in-place. Nearest neighbor (label-safe).
    """
    ref = sitk.ReadImage(str(ref_p))
    out_img = sitk.ReadImage(str(out_p))

    # target in (slices,H,W) -> SimpleITK expects (X,Y,Z) = (W,H,Slices)
    slices, H, W = target_slicesHW
    size_xyz = (int(W), int(H), int(slices))

    # Compute ref FOV (physical size) and choose output spacing so that FOV is preserved
    ref_size = ref.GetSize()            # (X,Y,Z)
    ref_spacing = ref.GetSpacing()      # (sx,sy,sz)
    fov_x = ref_size[0] * ref_spacing[0]
    fov_y = ref_size[1] * ref_spacing[1]
    fov_z = ref_size[2] * ref_spacing[2]

    out_spacing = (fov_x / size_xyz[0], fov_y / size_xyz[1], fov_z / size_xyz[2])

    res = sitk.Resample(
        out_img,
        size_xyz,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,   # label-safe
        ref.GetOrigin(),
        out_spacing,
        ref.GetDirection(),
        0,
        out_img.GetPixelID(),
    )
    _atomic_write(res, out_p)
    print(f"[RESAMPLE] {out_p.name} -> fixed size (s,H,W)={target_slicesHW}  (xyz)={size_xyz}")
    return True

def _resample_all_in_dir_overwrite_fixed(pred_root: Path, img_root: Path, target_slicesHW: Tuple[int, int, int]) -> tuple[int, int]:
    """
    For every NIfTI under pred_root: find its input ref under img_root,
    resample to FIXED (slices,H,W), overwrite in-place.
    """
    tot = ok = 0
    for m in pred_root.rglob("*"):
        if not (m.is_file() and _is_nii(m)):
            continue
        tot += 1
        ref = _find_ref_for_output(m, img_root)
        if ref is None:
            print(f"[WARN] No reference found for {m}")
            continue
        try:
            _resample_overwrite_to_fixed(m, ref, target_slicesHW)
            ok += 1
        except Exception as e:
            print(f"[ERR] Resample failed for {m}: {e}")
    return ok, tot

# -------------------- argparse --------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="CT_BET train/predict launcher with recursive traversal + resample back to FIXED size (overwrite)."
    )
    p.add_argument("--mode", choices=["train", "predict", "predict3d"], default="predict")
    p.add_argument("--image_folder", type=str, default=None,
                   help="Input folder (.nii/.nii.gz) or ROOT when --recursive is set.")
    p.add_argument("--mask_folder", type=str, default=None,
                   help="(Optional) Mask folder for training/eval.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output folder root. Defaults to results_folder/<label>/predictions")
    p.add_argument("--weights", type=str, default=None,
                   help="Path to weights .h5. Default: weights_folder/... if omitted.")
    p.add_argument("--gpu", type=str, default="0",
                   help="CUDA_VISIBLE_DEVICES. Use '' for CPU.")
    p.add_argument("--save_pred_mask", action="store_true",
                   help="Also save 3D NIfTI mask volume per case.")
    p.add_argument("--slice_fmt", choices=["npy", "png"], default="npy")
    p.add_argument("--img_row", type=int, default=512)
    p.add_argument("--img_col", type=int, default=512)
    p.add_argument("--channel", type=int, default=1)
    p.add_argument("--nb_classes", type=int, default=2)
    p.add_argument("--saved_class", type=int, default=2)

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--decay", type=float, default=1e-6)
    p.add_argument("--optimizer", type=str, default="adam")

    # recursive options
    p.add_argument("--recursive", action="store_true",
                   help="Treat --image_folder as ROOT and run per subfolder at given depth.")
    p.add_argument("--depth", type=int, default=3,
                   help="Folder depth when --recursive (default 3 => vendor/model/kernel).")

    # resample options
    p.add_argument("--resample_back", action="store_true", default=True,
                   help="After prediction, resample NIfTI outputs back to FIXED size and OVERWRITE them.")
    p.add_argument("--no-resample_back", dest="resample_back", action="store_false",
                   help="Disable post-prediction resampling.")
    p.add_argument("--target_size", type=str, default="48,256,256",
                   help="Target voxel size as 'slices,Height,Width'. Default '48,256,256'.")

    return p.parse_args()

# -------------------- main --------------------
def main():
    args = parse_args()

    # GPU selection
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Enforce SimpleITK if resample is requested
    if args.resample_back and sitk is None:
        print("[FATAL] --resample_back requires SimpleITK. Install via "
              "conda install -c conda-forge simpleitk  (or)  pip install SimpleITK")
        sys.exit(2)

    # Parse target size (slices,H,W)
    target_slicesHW = _parse_target_size(args.target_size)

    # output scaffold
    oLabel = Path(sys.argv[0]).stem + "_" + timeStamp
    resultsFolder = "results_folder"
    pred_folder = "predictions"

    if args.out_dir is None:
        base = Path(code_dir) / resultsFolder / oLabel
        (base / pred_folder).mkdir(parents=True, exist_ok=True)
    log_root = (Path(code_dir) / resultsFolder / oLabel
                if args.out_dir is None else Path(args.out_dir))
    log_root.mkdir(parents=True, exist_ok=True)

    # data aug (off)
    dataAugmentation = False
    datagen = ""; datagenPrams = ""; afold = ""

    # model wrapper
    unetSS = genUnet(
        root_folder=code_dir,
        image_folder=args.image_folder if args.image_folder else "image_data",
        mask_folder=args.mask_folder if args.mask_folder else "mask_data",
        save_folder=(str(Path(code_dir) / resultsFolder / oLabel)
                     if args.out_dir is None else str(Path(args.out_dir))),
        pred_folder=pred_folder,
        savePredMask=args.save_pred_mask,
        testLabelFlag=False, testMetricFlag=False,
        dataAugmentation=dataAugmentation,
        logFileName=f"log_{oLabel}.txt",
        datagen=datagen, oLabel=oLabel, checkWeightFileName=oLabel + ".h5",
        afold=afold, numEpochs=100, bs=1, nb_classes=args.nb_classes, sC=args.saved_class,
        img_row=args.img_row, img_col=args.img_col, channel=args.channel,
        classifier="softmax", optimizer=args.optimizer, lr=args.lr, decay=args.decay,
        dtype="float32", dtypeL="uint8", wType="slice",
        loss="categorical_crossentropy", metric="accuracy", model="unet",
    )

    # weights
    weightFile = (str(Path(code_dir) / "weights_folder" / "unet_CT_SS_20171114_170726.h5")
                  if args.weights is None else args.weights)

    def _post_resample(out_root: str, in_dir: str):
        if not args.resample_back:
            return
        ok, tot = _resample_all_in_dir_overwrite_fixed(Path(out_root), Path(in_dir), target_slicesHW)
        print(f"[post] resampled to {args.target_size} (s,H,W) & overwritten: {ok}/{tot}")

    def _run_once(in_dir: str, out_root: str, use_3d: bool):
        Path(out_root).mkdir(parents=True, exist_ok=True)
        print("Running {} prediction".format("3D" if use_3d else "2D"))
        print("Weights:", weightFile)
        print("Input folder:", in_dir)
        print("Output folder:", out_root)

        unetSS.pred_folder = pred_folder
        unetSS.save_folder = out_root
        if not use_3d:
            if hasattr(unetSS, "_slice_fmt"):
                unetSS._slice_fmt = args.slice_fmt
            unetSS.Predict(weightFile, in_dir=in_dir, out_dir=out_root)
        else:
            unetSS.Predict3D(weightFile, in_dir=in_dir, out_dir=out_root)

        # resample (overwrite) right after each folder
        _post_resample(out_root, in_dir)

    if args.mode == "train":
        print("Training...")
        unetSS.train()
        return

    use_3d = (args.mode == "predict3d")

    if not args.recursive:
        out_root = (str(Path(code_dir) / resultsFolder / oLabel / pred_folder)
                    if args.out_dir is None else args.out_dir)
        in_dir = args.image_folder if args.image_folder is not None else "image_data"
        _run_once(in_dir, out_root, use_3d)
    else:
        if args.image_folder is None:
            raise ValueError("--recursive requires --image_folder as the root directory.")
        root = Path(args.image_folder).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"image_folder (root) not found: {root}")

        base_out = (Path(code_dir) / resultsFolder / oLabel / pred_folder
                    if args.out_dir is None else Path(args.out_dir))
        base_out.mkdir(parents=True, exist_ok=True)

        level_dirs = _list_level_dirs(root, args.depth)
        level_dirs.sort()
        print(f"[INFO] Found {len(level_dirs)} candidate folders at depth={args.depth} under {root}")

        ran = 0
        for kdir in level_dirs:
            if not _has_nii(kdir):
                print(f"[SKIP] No NIfTI in: {kdir}")
                continue
            rel = kdir.relative_to(root)
            out_root = base_out / rel
            out_root.mkdir(parents=True, exist_ok=True)
            _run_once(str(kdir), str(out_root), use_3d)
            ran += 1

        print(f"[DONE] Launched predictions for {ran} folders.")

if __name__ == "__main__":
    main()
