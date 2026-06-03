# Data Format

StrokeTimer expects preprocessed NCCT volumes stored locally. The repository does not track CSV, NIfTI, NPY, DICOM, or raw archive files.

## Volume Layout

Set `DATA_ROOT` to a directory with this hierarchy:

```text
center_data/
└── center_name/
    └── manufacturer/
        └── model/
            └── kernel/
                └── *.npy
```

Each volume should be a brain-only NCCT array with shape `(48, 256, 256)` and dtype `float32`. The dataloader can center-crop or pad depth to 48 when `fix_depth: true`.

## Metadata CSV

Set `CSV_PATH` to a CSV containing at least:

```text
patient_id,bucket_age,phase
```

Required semantics:

- `patient_id`: case identifier matching the filename prefix, for example `CASE_0001`.
- `bucket_age`: one of `<4.5`, `4.5<`, `4.5-6`, `>6`.
- `phase`: one of `train`, `val`, `test`.

Optional but supported:

- `pid_canon`: canonical patient id. If absent, it is derived from `patient_id`.

## Runtime Overrides

The default config points to local placeholder paths. Use environment variables to run on your machine:

```bash
DATA_ROOT=/path/to/center_data \
CSV_PATH=/path/to/meta.csv \
CLASS_FREQ_JSON=configs/ncct_class_frequency.json \
bash scripts/train_stroketimer.sh
```
