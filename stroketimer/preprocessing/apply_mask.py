#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Brain extraction pipeline:

- Find image & mask NIfTI files in two folders.
- Pair them by an ID parsed from filename (configurable via delimiter + token_index).
- For each pair:
    * Read image & mask.
    * Resample both to fixed voxel size (X, Y, Z) while preserving FOV.
    * Binarize mask.
    * Multiply image * mask to get brain-only volume.
    * Apply fixed CT brain window (center=40, width=80 -> [0, 80]).
    * Save full 3D volume as .npy.
    * Save 2D slices along a chosen axis as .npy.

Notes
-----
- SimpleITK uses image size as (X, Y, Z), but numpy arrays are (Z, Y, X).
- target_size is always given in (X, Y, Z).
"""

import os
import argparse
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------- Fixed brain window ----------------
WINDOW_CENTER = 40.0
WINDOW_WIDTH = 80.0
WINDOW_LOWER = WINDOW_CENTER - WINDOW_WIDTH / 2.0  # 0
WINDOW_UPPER = WINDOW_CENTER + WINDOW_WIDTH / 2.0  # 80


class BrainExtractionProcessor:
    def __init__(self, target_size: Tuple[int, int, int] = (256, 256, 48)):
        """
        Parameters
        ----------
        target_size : (int, int, int)
            Desired output voxel counts in (X, Y, Z) for SimpleITK.
            Remember: SimpleITK size is (X, Y, Z), but numpy arrays from
            SimpleITK are shaped (Z, Y, X).
        """
        self.target_size_xyz = target_size

    # ---------- File helpers ----------
    @staticmethod
    def find_nifti_files(directory: str, recursive: bool = True) -> List[str]:
        """
        Find all NIfTI files in a directory.

        Parameters
        ----------
        directory : str
            Root directory.
        recursive : bool
            If True, search with **/*.nii / **/*.nii.gz, else only in top-level.

        Returns
        -------
        List[str]
            Sorted list of absolute paths.
        """
        directory = Path(directory).resolve()
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        patterns = ["**/*.nii", "**/*.nii.gz"] if recursive else ["*.nii", "*.nii.gz"]
        files: List[Path] = []
        for pattern in patterns:
            files.extend(directory.glob(pattern))
        return sorted(str(f) for f in files)

    @staticmethod
    def get_filename_stem(filepath: str) -> str:
        """
        Get filename without NIfTI extension.

        Examples
        --------
        /path/mrclean_late_30012.nii.gz -> mrclean_late_30012
        /path/mrclean_late_30012.nii    -> mrclean_late_30012
        """
        base = os.path.basename(filepath)
        if base.endswith(".nii.gz"):
            return base[:-7]
        if base.endswith(".nii"):
            return base[:-4]
        return base

    # ---------- Imaging ops ----------
    @staticmethod
    def ensure_binary_mask(mask: sitk.Image, threshold: float = 0.5) -> sitk.Image:
        """
        Ensure mask is binary after resampling.

        Parameters
        ----------
        mask : sitk.Image
            Mask image (possibly interpolated).
        threshold : float
            Threshold value; > threshold will be 1, else 0.

        Returns
        -------
        sitk.Image
            Binary mask with values {0, 1}.
        """
        arr = sitk.GetArrayFromImage(mask)
        binary = (arr > threshold).astype(np.uint8)
        out = sitk.GetImageFromArray(binary)
        out.CopyInformation(mask)
        return out

    @staticmethod
    def _compute_output_spacing(img: sitk.Image, out_size_xyz: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """
        Compute spacing to preserve physical FOV:

            out_spacing[i] = (in_size[i] * in_spacing[i]) / out_size[i]

        Parameters
        ----------
        img : sitk.Image
            Input image.
        out_size_xyz : (int, int, int)
            Desired size in (X, Y, Z).

        Returns
        -------
        (float, float, float)
            Output spacing in (sx, sy, sz).
        """
        in_size = img.GetSize()        # (X, Y, Z)
        in_spacing = img.GetSpacing()  # (sx, sy, sz)
        out_spacing = tuple(
            (in_size[i] * in_spacing[i]) / max(1, out_size_xyz[i]) for i in range(3)
        )
        return out_spacing

    def resample_to_size_preserve_fov(self, img: sitk.Image, out_size_xyz: Tuple[int, int, int], is_mask: bool) -> sitk.Image:
        """
        Resample to a fixed voxel size, adjusting spacing to preserve FOV.

        Parameters
        ----------
        img : sitk.Image
            Input image.
        out_size_xyz : (int, int, int)
            Desired output size in (X, Y, Z).
        is_mask : bool
            If True, use nearest-neighbor interpolation; else linear.

        Returns
        -------
        sitk.Image
            Resampled image.
        """
        out_spacing = self._compute_output_spacing(img, out_size_xyz)
        interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
        default_value = 0

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(out_size_xyz)            # (X, Y, Z)
        resampler.SetOutputSpacing(out_spacing)    # (sx, sy, sz)
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(default_value)
        return resampler.Execute(img)

    @staticmethod
    def apply_fixed_brain_window(array: np.ndarray) -> np.ndarray:
        """
        Clip intensities to CT brain window [0, 80] (center=40, width=80).

        Parameters
        ----------
        array : np.ndarray
            Input volume, shape (Z, Y, X).

        Returns
        -------
        np.ndarray
            Windowed volume.
        """
        return np.clip(array, WINDOW_LOWER, WINDOW_UPPER)

    # ---------- Main per-pair ----------
    def process_pair(
        self,
        image_path: str,
        mask_path: str,
        outdir_slices: str,
        outdir_volumes: str,
        suffix: str = "_brain_extracted",
        save_axis: str = "z",
    ) -> bool:
        """
        Process one image+mask pair.

        Steps:
            - Read NIfTI.
            - Resample to fixed size (X, Y, Z) with preserved FOV.
            - Binarize mask.
            - Multiply image and mask.
            - Apply fixed brain window.
            - Save full volume and per-slice arrays as .npy.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        try:
            logger.info(
                f"Processing pair:\n"
                f"  image: {os.path.basename(image_path)}\n"
                f"  mask : {os.path.basename(mask_path)}"
            )

            # 1) Read
            img = sitk.ReadImage(image_path)
            msk = sitk.ReadImage(mask_path)

            # 2) Resample both to fixed voxel size (preserve FOV)
            img_res = self.resample_to_size_preserve_fov(img, self.target_size_xyz, is_mask=False)
            msk_res = self.resample_to_size_preserve_fov(msk, self.target_size_xyz, is_mask=True)

            # 3) Binary mask (post-resample)
            msk_bin = self.ensure_binary_mask(msk_res)

            # 4) To numpy (SimpleITK -> (Z, Y, X))
            img_arr = sitk.GetArrayFromImage(img_res).astype(np.float32)
            msk_arr = sitk.GetArrayFromImage(msk_bin).astype(np.uint8)

            if img_arr.shape != msk_arr.shape:
                logger.error(f"Shape mismatch after resample: image {img_arr.shape} vs mask {msk_arr.shape}")
                return False

            # 5) Multiply, then window
            brain = img_arr * msk_arr
            brain = self.apply_fixed_brain_window(brain)

            # 6) Save full volume
            os.makedirs(outdir_volumes, exist_ok=True)
            stem = self.get_filename_stem(image_path)
            vol_path = os.path.join(outdir_volumes, f"{stem}{suffix}.npy")
            np.save(vol_path, brain)
            logger.info(f"Saved volume: {vol_path} | shape={brain.shape}")

            # 7) Save per-slice
            os.makedirs(outdir_slices, exist_ok=True)
            axis_map = {"z": 0, "y": 1, "x": 2}
            axis = axis_map.get(save_axis.lower(), 0)
            num_slices = brain.shape[axis]

            for idx in range(num_slices):
                if axis == 0:
                    sl = brain[idx, :, :]
                elif axis == 1:
                    sl = brain[:, idx, :]
                else:
                    sl = brain[:, :, idx]
                out_path = os.path.join(outdir_slices, f"{stem}{suffix}_{save_axis}{idx:03d}.npy")
                np.save(out_path, sl)

            logger.info(f"Saved {num_slices} slices along axis '{save_axis}' into {outdir_slices}")
            return True

        except Exception as e:
            logger.error(f"Failed processing {image_path} with {mask_path}: {str(e)}")
            return False


# ---------- Pairing helpers ----------

def normalize_stem_for_id(stem: str) -> str:
    """
    Normalize filename stem for ID extraction.

    - Removes common prediction/mask suffixes like:
        _pred, _mask, _brain, _brain_extracted
    - Example:
        mrclean_late_30012_pred  -> mrclean_late_30012
        mrclean_late_30012_mask  -> mrclean_late_30012

    This helps image and mask map to the same ID key.
    """
    # Order matters: longer suffix first to avoid partial matches
    suffixes = [
        "_brain_extracted",
        "_brain",
        "_pred",
        "_mask",
    ]
    for suf in suffixes:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def build_file_mapping(files: List[str], delimiter: str, token_index: int) -> Dict[str, List[str]]:
    """
    Build mapping: ID -> list of file paths.

    ID is parsed from filename stem using delimiter and token_index:

    - First, we normalize the stem (remove _pred/_mask/_brain/_brain_extracted).
    - Then:
        * If token_index >= 0:
              key = parts[token_index]  (if exists, otherwise fall back to full stem)
        * If token_index < 0:
              key = parts[-1]           (i.e., last token, often the numeric ID, e.g. 30012)

    Examples
    --------
    stem = "mrclean_late_30012"
    delimiter = "_"

    - token_index = 2 -> parts = ["mrclean", "late", "30012"] -> key="30012"
    - token_index = -1 -> key="30012" (last token)
    """
    mapping: Dict[str, List[str]] = {}
    proc = BrainExtractionProcessor()

    for fp in files:
        stem = proc.get_filename_stem(fp)
        stem_norm = normalize_stem_for_id(stem)

        if delimiter:
            parts = stem_norm.split(delimiter)
        else:
            parts = [stem_norm]

        if token_index < 0:
            key = parts[-1]  # use last token as ID
        elif token_index < len(parts):
            key = parts[token_index]
        else:
            # Fallback: use full normalized stem
            key = stem_norm

        mapping.setdefault(key, []).append(fp)

    return mapping


def select_best_file(files: List[str]) -> Optional[str]:
    """
    Select the best candidate among multiple files with the same ID.

    Strategy: choose the largest file by size (heuristic for "highest quality").

    Parameters
    ----------
    files : List[str]
        Candidate files for a single ID.

    Returns
    -------
    Optional[str]
        Selected file path, or None if list is empty.
    """
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    return sorted(files, key=os.path.getsize, reverse=True)[0]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Resample image & mask to fixed size (X,Y,Z), apply brain mask, "
            "CT brain window 40/80, and save full volumes + slices as .npy."
        )
    )
    parser.add_argument("--image_dir", required=True, help="Directory containing source NIfTI images.")
    parser.add_argument("--mask_dir", required=True, help="Directory containing mask NIfTI images.")
    parser.add_argument("--output_dir_slices", required=True, help="Output directory for 2D slices (.npy).")
    parser.add_argument("--output_dir_volumes", required=True, help="Output directory for 3D volumes (.npy).")
    parser.add_argument("--delimiter", default="_", help="Delimiter to split filename stem (default: '_').")
    parser.add_argument(
        "--token_index",
        type=int,
        default=-1,
        help=(
            "Which token to use as ID after splitting by delimiter.\n"
            "  >=0: use parts[token_index]\n"
            "  <0 : use last token (e.g., mrclean_late_30012 -> 30012). "
            "Default: -1"
        ),
    )
    parser.add_argument("--suffix", default="_brain_extracted", help="Suffix appended to output filenames.")
    parser.add_argument("--save_axis", choices=["z", "y", "x"], default="z", help="Slice axis to save (default: z).")
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=3,
        default=[256, 256, 48],
        help="Output voxel counts as X Y Z (default: 256 256 48).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If set, search for NIfTI files recursively under image_dir/mask_dir.",
    )

    args = parser.parse_args()

    proc = BrainExtractionProcessor(target_size=tuple(args.target_size))

    # 1) Collect files
    logger.info(f"Scanning image_dir={args.image_dir}")
    img_files = proc.find_nifti_files(args.image_dir, recursive=args.recursive)
    logger.info(f"Found {len(img_files)} image NIfTI files.")

    logger.info(f"Scanning mask_dir={args.mask_dir}")
    msk_files = proc.find_nifti_files(args.mask_dir, recursive=args.recursive)
    logger.info(f"Found {len(msk_files)} mask NIfTI files.")

    if not img_files:
        logger.error("No image files found. Exiting.")
        return 1
    if not msk_files:
        logger.error("No mask files found. Exiting.")
        return 1

    # 2) Build mappings
    img_map = build_file_mapping(img_files, args.delimiter, args.token_index)
    msk_map = build_file_mapping(msk_files, args.delimiter, args.token_index)

    logger.info(f"#image ID keys: {len(img_map)}")
    logger.info(f"#mask  ID keys: {len(msk_map)}")

    keys = sorted(set(img_map) & set(msk_map))
    logger.info(f"Found {len(keys)} matching IDs (present in both image & mask).")

    # Debug: print some sample keys
    for k in keys[:10]:
        logger.info(
            f"ID={k}: #images={len(img_map[k])}, #masks={len(msk_map[k])}"
        )

    if not keys:
        logger.error("No matching IDs between image_dir and mask_dir. Nothing to process.")
        return 1

    os.makedirs(args.output_dir_slices, exist_ok=True)
    os.makedirs(args.output_dir_volumes, exist_ok=True)

    # 3) Process each key
    num_ok = 0
    num_fail = 0

    for key in keys:
        img = select_best_file(img_map[key])
        msk = select_best_file(msk_map[key])

        if not img or not msk:
            logger.warning(f"Skipping ID={key} due to missing image or mask.")
            num_fail += 1
            continue

        ok = proc.process_pair(
            img,
            msk,
            args.output_dir_slices,
            args.output_dir_volumes,
            suffix=args.suffix,
            save_axis=args.save_axis,
        )

        if ok:
            num_ok += 1
        else:
            num_fail += 1

    logger.info(
        f"Done. Successfully processed {num_ok} ID(s); failed {num_fail} ID(s)."
    )
    return 0 if num_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
