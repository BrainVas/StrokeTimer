#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launcher for CT_BET: train/predict with optional CLI-specified input/output.

Original header:
Created on Thu Nov  9 13:18:46 2017
@author: m131199
"""

import os
import sys
import time
import argparse

print(sys.argv[0])  # Print script name (for debugging/logging)

# ==== Prepare timestamp for logging/output naming ====
code_dir = os.getcwd()
hour = str(time.localtime()[3])
mins = str(time.localtime()[4])
sec = str(time.localtime()[5])
timeStamp = (
    str(time.localtime()[0])
    + str(time.localtime()[1])
    + str(time.localtime()[2])
    + "_"
    + hour
    + mins
    + sec
)

# Add current dir to sys.path so we can import local modules
sys.path.append(code_dir)

from auggen import AugmentationGenerator as dataGenerator
from model_CT_SS import Unet_CT_SS as genUnet


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="CT_BET train/predict launcher with CLI I/O paths"
    )
    # Run mode (train, 2D predict, or 3D predict)
    p.add_argument(
        "--mode",
        choices=["train", "predict", "predict3d"],
        default="predict",
        help="Run mode",
    )
    # Input folder of NIfTI images
    p.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="Input folder of NIfTI images (.nii/.nii.gz). If omitted, use class default.",
    )
    # Mask folder (used only for training or evaluation)
    p.add_argument(
        "--mask_folder",
        type=str,
        default=None,
        help="(Optional) Mask folder for training/eval. Usually empty in predict.",
    )
    # Output folder
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output folder. If omitted, defaults to results_folder/<label>/predictions",
    )
    # Weights file (.h5)
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to weights .h5. If omitted, will look under weights_folder/",
    )
    # GPU device selection
    p.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES setting (string). Use '' for CPU.",
    )
    # Save 3D NIfTI mask volume in addition to per-slice
    p.add_argument(
        "--save_pred_mask",
        action="store_true",
        help="Also save 3D NIfTI mask volume per case.",
    )
    # Format for per-slice output
    p.add_argument(
        "--slice_fmt",
        choices=["npy", "png"],
        default="npy",
        help="Format for per-slice export (if your model code enables it).",
    )
    # Basic model/input shape options
    p.add_argument("--img_row", type=int, default=512)
    p.add_argument("--img_col", type=int, default=512)
    p.add_argument("--channel", type=int, default=1)
    p.add_argument("--nb_classes", type=int, default=2)
    p.add_argument(
        "--saved_class", type=int, default=2, help="Which class channel to save (1-based)."
    )

    # Optimizer hyperparameters
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--decay", type=float, default=1e-6)
    p.add_argument("--optimizer", type=str, default="adam")

    return p.parse_args()


def main():
    args = parse_args()

    # === Select GPU/CPU ===
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # === Default output structure ===
    arg0 = sys.argv[0]
    oLabel = os.path.basename(arg0)[:-3] + "_" + timeStamp
    resultsFolder = "results_folder"
    pred_folder = "predictions"

    # Create default structure if out_dir is not provided
    if args.out_dir is None:
        os.makedirs(os.path.join(code_dir, resultsFolder, oLabel), exist_ok=True)
        os.makedirs(
            os.path.join(code_dir, resultsFolder, oLabel, pred_folder), exist_ok=True
        )

    # === Log file ===
    log_root = (
        os.path.join(code_dir, resultsFolder, oLabel)
        if args.out_dir is None
        else args.out_dir
    )
    os.makedirs(log_root, exist_ok=True)
    log_path = os.path.join(log_root, f"log_{oLabel}.txt")
    logFile = open(log_path, "w")

    # === Data augmentation (disabled by default) ===
    dataAugmentation = False
    if dataAugmentation:
        datagen = dataGenerator(
            rotation_z=30,
            rotation_x=0,
            rotation_y=0,
            translation_xy=5,
            translation_z=0,
            scale_xy=0.1,
            scale_z=0,
            flip_h=True,
            flip_v=False,
        )
        datagenPrams = datagen.__str__()
        afold = 3
    else:
        datagen = ""
        datagenPrams = ""
        afold = ""

    # === Instantiate model wrapper ===
    unetSS = genUnet(
        root_folder=code_dir,
        image_folder=args.image_folder if args.image_folder else "image_data",
        mask_folder=args.mask_folder if args.mask_folder else "mask_data",
        save_folder=(
            os.path.join(code_dir, resultsFolder, oLabel)
            if args.out_dir is None
            else args.out_dir
        ),
        pred_folder=pred_folder,
        savePredMask=args.save_pred_mask,
        testLabelFlag=False,
        testMetricFlag=False,
        dataAugmentation=dataAugmentation,
        logFileName=f"log_{oLabel}.txt",
        datagen=datagen,
        oLabel=oLabel,
        checkWeightFileName=oLabel + ".h5",
        afold=afold,
        numEpochs=100,
        bs=1,
        nb_classes=args.nb_classes,
        sC=args.saved_class,  # 1-based index
        img_row=args.img_row,
        img_col=args.img_col,
        channel=args.channel,
        classifier="softmax",
        optimizer=args.optimizer,
        lr=args.lr,
        decay=args.decay,
        dtype="float32",
        dtypeL="uint8",
        wType="slice",
        loss="categorical_crossentropy",
        metric="accuracy",
        model="unet",
    )

    # Write configuration to log
    logFile.write("\n" + "-" * 30 + "\n")
    logFile.write(unetSS.__str__())
    logFile.write("\n" + "-" * 30 + "\n")
    logFile.write(datagenPrams)
    logFile.write("\n" + "-" * 30 + "\n")
    logFile.close()

    # === Resolve weights path ===
    if args.weights is None:
        weight_folder = os.path.join(code_dir, "weights_folder")
        weightFile = os.path.join(weight_folder, "unet_CT_SS_20171114_170726.h5")
    else:
        weightFile = args.weights

    # === Run modes ===
    if args.mode == "predict":
        # 2D slice-by-slice prediction
        out_root = (
            os.path.join(code_dir, resultsFolder, oLabel, pred_folder)
            if args.out_dir is None
            else args.out_dir
        )
        os.makedirs(out_root, exist_ok=True)

        print("Running 2D prediction")
        print("Weights:", weightFile)
        print("Input folder:", args.image_folder if args.image_folder else "(class default)")
        print("Output folder:", out_root)

        unetSS.pred_folder = pred_folder
        unetSS.save_folder = out_root

        # Optional: pass slice format if supported by model class
        if hasattr(unetSS, "_slice_fmt"):
            unetSS._slice_fmt = args.slice_fmt

        unetSS.Predict(weightFile, in_dir=args.image_folder, out_dir=out_root)

    elif args.mode == "predict3d":
        # 3D volume prediction
        out_root = (
            os.path.join(code_dir, resultsFolder, oLabel, pred_folder)
            if args.out_dir is None
            else args.out_dir
        )
        os.makedirs(out_root, exist_ok=True)

        print("Running 3D prediction")
        print("Weights:", weightFile)
        print("Input folder:", args.image_folder if args.image_folder else "(class default)")
        print("Output folder:", out_root)

        unetSS.pred_folder = pred_folder
        unetSS.save_folder = out_root
        unetSS.Predict3D(weightFile, in_dir=args.image_folder, out_dir=out_root)

    elif args.mode == "train":
        # Training mode
        print("Training...")
        unetSS.train()
        # If you want 3D training:
        # unetSS.train3D()
    else:
        print("please set a valid --mode: train | predict | predict3d")


if __name__ == "__main__":
    main()
