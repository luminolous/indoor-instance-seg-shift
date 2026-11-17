# indoor-instance-seg-shift

Benchmark of three instance segmentation models on indoor scenes, an input-size ablation, and an
analysis of how performance drops under domain shift (original test split and a self-collected
phone-photo test set).

Course assignment: Computer Vision, Evaluasi CPMK 3 (ITS).

> **Status:** work in progress. Code for data preparation, training and evaluation is being
> built; no model has been trained yet. All result tables below are placeholders.

## Contents

- [Goal](#goal)
- [Dataset](#dataset)
- [Models](#models)
- [Experiment: input size](#experiment-input-size)
- [Evaluation protocol](#evaluation-protocol)
- [Setup (Windows + CUDA)](#setup-windows--cuda)
- [How to run](#how-to-run)
- [Repository layout](#repository-layout)
- [Results](#results)
- [Failure analysis](#failure-analysis)
- [Process evidence](#process-evidence)
- [License](#license)

## Goal

1. Train and compare **3 instance segmentation models from 2+ architecture families** on
   Panoptic Indoor Segmentation v5.
2. Study one assigned variable: **input size 320 / 640 / 960** (longest image side).
3. Measure performance on three evaluation sets with increasing distribution shift and analyse
   the gap.
4. Explain 5 concrete failures on phone photos by the model's mechanism, and test each
   explanation with a small experiment.

## Dataset

**Panoptic Indoor Segmentation v5** (Roboflow, COCO Segmentation export, CC BY 4.0).

| Split | Images | Annotations |
|---|---|---|
| train | 397 | 4479 |
| valid | 166 | 2022 |
| test | 359 | 5344 |

- 75 category ids (0-74). Id 0 (`indoor-objects`) is the Roboflow supercategory and has no
  annotations.
- The dataset is used **exactly as exported**: no class merging, dropping, renaming or
  re-splitting. The only transformation is a format conversion (COCO to YOLO-seg) that keeps
  the class ids.
- 4 train/valid images are stored rotated by 90 degrees relative to their annotations (no EXIF
  tag). They are rotated back **in memory when loaded** (`data.rotation_fixes` in
  `configs/base.yaml`, applied by `iiss.data.images.load_image`); the files are not modified.
  See `results/data_validation.md`.
- The original test split is already shifted from train (mixed, often higher resolutions and a
  different class distribution; 9 test classes have no train instances). See
  `results/eda/summary.md`.

**Phone test set** (self-collected): at least 40 phone photos of real rooms, at least 10 in hard
conditions (low light, backlight, occlusion, extreme angle), annotated in Roboflow with the same
class names and remapped to the v5 ids by name.

The dataset files are not part of this repository. Place the Roboflow export at
`dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/` (`train/`, `valid/`, `test/`,
each with `_annotations.coco.json`).

## Models

| Key | Model | Family | Library | Pretrained |
|---|---|---|---|---|
| `yolo` | YOLO11-seg (`yolo11s-seg`) | one-stage CNN, prototype masks | `ultralytics` | COCO |
| `maskrcnn` | Mask R-CNN R50-FPN v2 | two-stage CNN, 28x28 RoI mask head | `torchvision` | COCO |
| `m2f` | Mask2Former Swin-T | transformer, mask classification | `transformers` | COCO instance |

## Experiment: input size

`S` in {320, 640, 960} is the **longest image side after resize** (aspect ratio preserved), used
at both training and evaluation time. 640 is the baseline.

| | 320 | 640 | 960 |
|---|---|---|---|
| yolo | ablation | baseline | ablation |
| maskrcnn | ablation | baseline | ablation |
| m2f | ablation | baseline | ablation |

Run id format: `{model}_s{S}_seed{seed}`, e.g. `yolo_s640_seed42`.

Training protocol, identical for every run:

- train on `train`, select checkpoints on `valid` only
- seed 42, mixed precision (AMP)
- **2-hour wall-clock cap per run**, cumulative across resumes, enforced in code
- same epoch budget and LR schedule across input sizes within one model
- framework-default augmentation

## Evaluation protocol

- One evaluator for all models: predictions are exported to COCO results JSON (RLE masks at the
  original image resolution) and scored with `pycocotools` `COCOeval(iouType="segm")`.
  Framework-internal metrics are not reported.
- Metrics: mask mAP@[.50:.95], mAP@.50, APs/APm/APl, per-class AP; bbox mAP as secondary.
- Evaluation sets: `valid` (in-distribution), `test` (original, shifted), `phone` (self-made,
  shifted further).
- Breakdowns: phone set by capture condition, classes present vs absent in train, things vs
  stuff.

## Setup (Windows + CUDA)

Tested target: Windows 11, NVIDIA RTX 4050 Laptop GPU (6 GB), Python 3.12, [uv](https://docs.astral.sh/uv/).

### 1. Check the NVIDIA driver

```powershell
nvidia-smi
```

The header must show a driver that supports CUDA 12.6 or newer (driver >= 560). The PyTorch
wheels bundle their own CUDA runtime, so a separate CUDA Toolkit install is **not** needed.

### 2. Create the virtual environment

```powershell
uv venv .venv --python 3.12
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 3. Install PyTorch (CUDA build) first

PyPI only ships CPU wheels of PyTorch for Windows, so install torch from the PyTorch index
**before** the other dependencies:

```powershell
uv pip install "torch==2.14.1" "torchvision==0.29.1" --index-url https://download.pytorch.org/whl/cu126
```

`cu126` works with any driver >= 560. With a newer driver you may use `cu130` instead.

### 4. Install the project and the remaining dependencies

```powershell
uv pip install -e ".[dev]"
```

Optional Weights & Biases logging: `uv pip install -e ".[dev,wandb]"`.

Without `uv`, replace `uv pip` with `python -m pip` in the commands above.

### 5. Environment variables

```powershell
Copy-Item .env.example .env
```

Fill in only the keys you need. `.env` is gitignored.

### 6. Verify

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -q
```

Expected: the version ends with `+cu126`, followed by `True` and the GPU name, and all tests
pass. If it prints `False` or a `+cpu` version, a later install replaced torch with the CPU
wheel. Re-run step 3.

### Notes

- `pycocotools` ships prebuilt Windows wheels; no Visual C++ build tools are required.
- detectron2 is intentionally not used (difficult to install on native Windows). Mask R-CNN
  comes from `torchvision`, Mask2Former from Hugging Face `transformers`.
- Runs that do not fit in 6 GB VRAM fall back to Google Colab.

## How to run

> Entry points marked `planned` are not implemented yet. This section is updated as each one
> lands.

```powershell
# 1. Validate the dataset, run the EDA, convert COCO -> YOLO-seg
python -m iiss.data.validate            # -> results/data_validation.md
python -m iiss.data.eda                 # -> results/eda/
python -m iiss.data.coco_to_yolo        # -> dataset/derived/yolo/ (--overwrite to rebuild)

# 2. Train one run (2 h cap, resumable)
python -m iiss.train --run configs/runs/yolo_s640_seed42.yaml          # planned

# 3. Export predictions and evaluate
python -m iiss.predict --run configs/runs/yolo_s640_seed42.yaml --eval-set test   # planned
python -m iiss.evaluate --run yolo_s640_seed42 --eval-set test                    # planned
```

Every CLI will support `--smoke` (16 train / 8 eval images, `S=320`, at most 2 epochs, output
in `runs/smoke/`) for a quick end-to-end check.

Configs are merged in the order `configs/base.yaml` -> `configs/models/<model>.yaml` ->
`configs/runs/<run_id>.yaml`. Each run writes its resolved config and `env.json` into its run
directory.

## Repository layout

```
configs/            base.yaml, models/<model>.yaml, runs/<run_id>.yaml
src/iiss/
  data/             COCO IO, validation, COCO -> YOLO conversion, phone-set importer, EXIF
  models/           one wrapper per model: build(), train(), predict()
  analysis/         overlays, instance matching, failure tagging, small experiments
  utils/            config, paths, seeding, env snapshot, logging, run timer
scripts/            thin one-off scripts (EDA, figure export)
notebooks/          exploration and report figures only
tests/              pytest, CPU-only, tiny fixtures
dataset/            data (not committed); dataset/derived/ is generated
runs/               checkpoints and raw logs (not committed)
results/            metric tables and small figures (committed)
```

## Results

> **No results yet.** Every cell is filled only from real evaluation output in `results/`.

### Mask mAP@[.50:.95] by model, input size and evaluation set

| Model | S | valid | test | phone |
|---|---|---|---|---|
| yolo | 320 | _pending_ | _pending_ | _pending_ |
| yolo | 640 | _pending_ | _pending_ | _pending_ |
| yolo | 960 | _pending_ | _pending_ | _pending_ |
| maskrcnn | 320 | _pending_ | _pending_ | _pending_ |
| maskrcnn | 640 | _pending_ | _pending_ | _pending_ |
| maskrcnn | 960 | _pending_ | _pending_ | _pending_ |
| m2f | 320 | _pending_ | _pending_ | _pending_ |
| m2f | 640 | _pending_ | _pending_ | _pending_ |
| m2f | 960 | _pending_ | _pending_ | _pending_ |

### Domain shift (drop from valid)

| Model | S | valid -> test (abs / rel) | valid -> phone (abs / rel) |
|---|---|---|---|
| _pending_ | | | |

### Phone set by capture condition

| Model | S | normal | low_light | backlight | occlusion | extreme_angle |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | |

### Cost

| Run | GPU time (h) | Peak VRAM (MB) | Cap hit | Inference (ms/img) |
|---|---|---|---|---|
| _pending_ | | | | |

## Failure analysis

_Pending._ Five phone-set images with clear failures (missed object, mask leakage, wrong class,
split object), each with a mechanism-based explanation and one small experiment that tests it.

## Process evidence

- Per run: `results.csv` or framework equivalent, TensorBoard logs, resolved `config.yaml`,
  `env.json` (package versions, GPU, git commit).
- `runs/gpu_time_log.csv`: one row per training session (run id, start, end, wall-clock
  seconds, GPU, peak VRAM, cap hit, status).
- `logbook.md`: problems and decisions, written by the author.

## License

Code: MIT (see `LICENSE`). Dataset: CC BY 4.0, Panoptic Indoor Segmentation v5 on Roboflow.
