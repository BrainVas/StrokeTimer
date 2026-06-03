"""Prepare one 4D CTP/NIfTI file for CT_BET skull stripping.

The input should be a NIfTI image. If it is 4D, the first timepoint is
extracted. The output is reoriented to RAS+ and resized in-plane to 512 x 512.
Local input/output files are intentionally provided through CLI arguments so
private paths are not stored in the repository.
"""

from __future__ import annotations

import argparse

import nibabel as nib
from scipy.ndimage import zoom


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input NIfTI path.")
    parser.add_argument("--output", required=True, help="Output NIfTI path.")
    parser.add_argument("--size", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    return parser.parse_args()


def main():
    args = parse_args()

    img = nib.load(args.input)
    data = img.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]

    first_timepoint_img = nib.Nifti1Image(data, img.affine, img.header)
    reoriented_img = nib.as_closest_canonical(first_timepoint_img)

    data = reoriented_img.get_fdata()
    current_shape = data.shape[:2]
    target_shape = tuple(args.size)
    zoom_factors = [target_shape[i] / current_shape[i] for i in range(2)]
    zoom_factors.append(1.0)

    resized_data = zoom(data, zoom_factors, order=1)
    out_img = nib.Nifti1Image(resized_data, reoriented_img.affine, reoriented_img.header)
    nib.save(out_img, args.output)


if __name__ == "__main__":
    main()
