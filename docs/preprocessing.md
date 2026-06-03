# Preprocessing

StrokeTimer uses a two-step preprocessing pipeline before training:

```text
Raw or resampled NCCT
  -> CT_BET brain-mask prediction
  -> apply_mask.py brain extraction
  -> brain-only volume used by StrokeTimer
```

## 1. Brain Mask Prediction With CT_BET

The CT_BET source code is included under:

```text
stroketimer/preprocessing/CT_BET/
```

Place local CT_BET inputs and weights in the CT_BET runtime folders:

```text
stroketimer/preprocessing/CT_BET/
├── image_data/       # local input CT files
├── mask_data/        # local predicted or reference masks
├── results_folder/   # local CT_BET outputs
└── weights_folder/   # local CT_BET .h5 weights
```

The `.h5` weights and NIfTI examples are not tracked. See `weights_folder/weight_file` for CT_BET's original weight note.

Typical CT_BET run:

```bash
cd stroketimer/preprocessing/CT_BET
bash main.sh
```

## 2. Apply Brain Mask

After CT_BET generates masks, apply them to the source volumes:

```bash
python -m stroketimer.preprocessing.apply_mask \
  --raw_dir /path/to/raw_or_resampled_ct \
  --out_dir /path/to/brain_extracted_output \
  --size 48 256 256 \
  --workers 8
```

The downstream training dataloader expects brain-only volumes under the `DATA_ROOT` hierarchy described in `docs/data_format.md`.
