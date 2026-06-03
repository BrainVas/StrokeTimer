# Checkpoints

The trained checkpoint is stored locally but ignored by git:

```text
checkpoints/best/final_model_checkpoint.pth
```

This file is about 151 MB, so it should not be committed to a normal GitHub repository. Keep it locally, publish it through a release asset, institutional storage, or Git LFS if public distribution is needed later.

The trained weight is available at [checkpoints](https://drive.google.com/drive/folders/1F9rZokUZS34ctuWBuaUmGgnL9_6KSPKN?usp=drive_link).

Run evaluation with:

```bash
CHECKPOINT=checkpoints/best/final_model_checkpoint.pth \
bash scripts/evaluate_stroketimer.sh
```

The evaluation script calls:

```bash
python -m stroketimer.visualize --cfg configs/stroketimer_best.yaml --ckpt checkpoints/best/final_model_checkpoint.pth
```

The `.gitignore` file excludes `.pth`, `.pt`, `.ckpt`, and `.h5` artifacts to prevent accidental uploads.
