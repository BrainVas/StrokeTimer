# StrokeTimer: Robust Representation Learning for Ischemic Stroke Onset-Time Estimation from Non-contrast CT

This repository contains the code release for **StrokeTimer**, a framework for ischemic stroke onset-time classification from non-contrast CT (NCCT). StrokeTimer combines self-supervised disentanglement learning with energy-guided contrastive learning to address subtle early ischemic changes, severe class imbalance, and center-scanner heterogeneity.

Onset time is modeled with three clinically relevant windows: `<4.5 h`, `4.5-6 h`, and `>6 h`.

## Quick Start

### 1. Setup

```bash
git clone https://github.com/BrainVas/StrokeTimer.git
cd StrokeTimer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Preprocess NCCT

StrokeTimer uses CT_BET first to predict the brain mask, then `apply_mask.py` to extract the brain-only volume:

```text
Raw or resampled NCCT
  -> CT_BET brain-mask prediction
  -> apply_mask.py brain extraction
  -> brain-only volume for training/evaluation
```

CT_BET is included under:

```text
stroketimer/preprocessing/CT_BET/
```

See [docs/preprocessing.md](docs/preprocessing.md) for the full preprocessing workflow.

### 3. Train

```bash
bash scripts/train_stroketimer.sh
```

Override local data and output paths without editing YAML:

```bash
DATA_ROOT=/path/to/center_data \
CSV_PATH=/path/to/meta_with_phase_center_stratified_balanced.csv \
OUTPUT_DIR=outputs/my_run \
bash scripts/train_stroketimer.sh
```

### 4. Evaluate

```bash
CHECKPOINT=checkpoints/best/final_model_checkpoint.pth \
bash scripts/evaluate_stroketimer.sh
```

Evaluation is implemented by `stroketimer.visualize`. It loads the checkpoint, computes metrics, saves ROC/PR outputs, writes per-case predictions, and generates Grad-CAM visualizations for every case under `OUTPUT_DIR`. Grad-CAM targets the predicted class by default; pass `--gradcam-target label` to visualize the ground-truth class instead.

## Project Structure

```text
StrokeTimer/
|-- configs/                    # Reproducible YAML and class-frequency JSON
|-- scripts/                    # Local and Slurm launchers
|-- stroketimer/                # Python package
|   |-- data/                   # NCCT dataset and dataloaders
|   |-- losses/                 # Energy-guided CMS and balanced softmax
|   |-- models/                 # ResNeXt3D backbone and classifier
|   |-- preprocessing/          # Preprocessing helpers
|   |   |-- CT_BET/             # Brain-mask prediction before apply_mask
|   |   `-- apply_mask.py       # Brain extraction using predicted masks
|   |-- train.py                # Main training/evaluation entrypoint
|   `-- runner.py               # Stage scheduler, optimization, metrics
|-- docs/                       # Paper PDF and usage notes
|-- data/                       # Local data placeholder, ignored by git
`-- checkpoints/                # Local checkpoints, ignored by git
```

## Data Path Structure

Raw medical data and metadata are not included in this repository. Locally, set `DATA_ROOT` to a directory with this hierarchy:

```text
center_data/
`-- <center_name>/
    `-- <manufacturer>/
        `-- <scanner_model>/
            `-- <reconstruction_kernel>/
                `-- <case_id>_brain_extracted.npy
```

A convenient local workspace can look like this:

```text
data/
|-- meta_with_phase_center_stratified_balanced.csv
`-- center_data/
    `-- <center_name>/<manufacturer>/<scanner_model>/<kernel>/*.npy
```

The metadata CSV should contain at least:

```text
patient_id,bucket_age,phase
```

- `patient_id`: case identifier matching the volume filename prefix.
- `bucket_age`: one of `<4.5`, `4.5-6`, `>6`.
- `phase`: one of `train`, `val`, `test`.

See [docs/data_format.md](docs/data_format.md) for details.

## Checkpoints

Checkpoint files are kept locally under `checkpoints/` and ignored by git. The trained weight is available at [checkpoints](https://drive.google.com/drive/folders/1F9rZokUZS34ctuWBuaUmGgnL9_6KSPKN?usp=drive_link). See [docs/checkpoints.md](docs/checkpoints.md).

## Citation
Please cite our paper if you find it useful. Feel free to contact r.su@tue.nl with questions or for collaboration.

```bibtex
@article{stroketimer2026,
  title  = {StrokeTimer: Robust Representation Learning for Ischemic Stroke Onset-Time Estimation from Non-contrast CT},
  author = {Wang, Weiru and Olthuis, Susanne and Lavrova, Lisa and van Oostenbrugge, Robert and Majoie, Charles and Van Zwam, Wim and Su, Ruisheng},
  year   = {2026},
  note   = {Manuscript}
}
```

## Notes

This repository is intended for academic research. It does not include raw or processed patient imaging data.
