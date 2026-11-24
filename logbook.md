# Logbook

Record of problems and decisions during the project, written by the author. Newest entry at the
bottom. One entry per problem or decision; keep it short and link to the evidence.

## How to write an entry

Copy the template below, fill in every field, and keep the heading format
`### YYYY-MM-DD — <short title>` so entries stay sortable.

- **Context:** which task, run id or step this concerns.
- **Problem / observation:** what happened or what had to be decided. Quote error messages exactly.
- **Options considered:** the alternatives, if any.
- **Decision:** what was chosen and why.
- **Evidence:** links to files, logs, commits or screenshots (e.g. `runs/<run_id>/results.csv`,
  `runs/gpu_time_log.csv`, a commit hash, `results/...`).
- **Follow-up:** anything left to do, or "none".

## Template

```markdown
### YYYY-MM-DD — <short title>

- **Context:**
- **Problem / observation:**
- **Options considered:**
- **Decision:**
- **Evidence:**
- **Follow-up:**
```

## GPU time summary

Filled in from `runs/gpu_time_log.csv` (do not type numbers by hand).

| Run id | Sessions | Total GPU time (h) | Cap hit | Notes |
|---|---|---|---|---|
| `yolo_s640_seed42` | 1 | 0.35 | no |  |
| `maskrcnn_s640_seed42` | 1 | 0.42 | no |  |
| `m2f_s640_seed42` | 1 | 1.32 | no |  |
| `yolo_s320_seed42` | 1 | 0.20 | no |  |
| `yolo_s960_seed42` | 1 | 0.72 | no |  |
| `maskrcnn_s320_seed42` | 1 | 0.37 | no |  |
| `maskrcnn_s960_seed42` | 1 | 0.65 | no |  |
| `m2f_s320_seed42` | 1 | 0.75 | no |  |
| `m2f_s960_seed42` | 1 | 2.05 | yes | stopped at ~10/50 epochs, VRAM spill |
| **Total** | 9 | 6.80 | | |

## Entries

<!-- Add entries below this line. -->

### 2026-11-17 — Dataset version and export format

- **Context:** Data download from Roboflow Universe (`panopticindoor/panoptic-indoor-segmentation`).
- **Problem / observation:** The slides ask for "Panoptic Indoor Segmentation v5". Roboflow offers many export formats. The YOLO TXT formats sit under "Convert To Object Detection" and the OBB formats keep only boxes, so both drop the masks.
- **Options considered:** COCO Segmentation; YOLOv8/v11 TXT; YOLO OBB.
- **Decision:** Export v5 (2024-04-01, 922 images) as **COCO Segmentation**. It keeps the polygons, detectron-style libraries read it directly, and YOLO-seg labels can be derived from it.
- **Evidence:** `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/README.roboflow.txt`, `results/data_validation.md`.
- **Follow-up:** none.

### 2026-11-17 — Minimal dataset modification policy

- **Context:** Planning the class set for training.
- **Problem / observation:** v5 has 75 categories (id 0 `indoor-objects` is an empty supercategory), near-duplicate classes (`scissor`/`scissors`, `paper`/`paper-merged`, `dining table`/`table-merged`, `wall`/`wall-other-merged`), a 1-instance noise class `st-`, and many "stuff" classes.
- **Options considered:** (A) merge duplicates and drop noise; (B) keep only "thing" classes with enough train instances; (C) keep the dataset exactly as exported.
- **Decision:** (C). The slides require the self-made test set to use "the same classes as the dataset", and the digit 8–9 variable ("3 hardest classes vs all classes") implies all classes is the default setting. Class index = COCO `category_id` everywhere (`num_classes = 75`). The only transformation is COCO → YOLO-seg format conversion. Subset analyses (things vs stuff, classes absent from train) are done at evaluation time only.
- **Evidence:** `configs/base.yaml`, commit 622bab9 (converter).
- **Follow-up:** none.

### 2026-11-17 — Models, hardware and assigned variable

- **Context:** Experiment design.
- **Problem / observation:** Need ≥ 3 models from ≥ 2 families. My student ID ends in **0**, so the variable to investigate is **input size 320 / 640 / 960**. Compute: local laptop GPU (RTX 4050 Laptop, 6 GB) instead of Colab.
- **Options considered:** detectron2 Mask R-CNN vs torchvision Mask R-CNN; Colab vs local GPU.
- **Decision:** YOLO11s-seg (one-stage CNN, Ultralytics), Mask R-CNN R50-FPN v2 (two-stage CNN, torchvision), Mask2Former Swin-T (transformer, HF transformers), all COCO-pretrained. detectron2 avoided because it is hard to install on native Windows. Input size S = longest image side after resize, aspect preserved, same S at train and eval. 2-hour training cap per run (from the slides' Colab note), applied to every run for fairness.
- **Evidence:** `configs/models/`, `configs/runs/`.
- **Follow-up:** none.

### 2026-11-17 — The original test split is already shifted from train

- **Context:** First look at the dataset statistics.
- **Problem / observation:** Train/valid are almost all 640×480, while test has 216 distinct sizes (longest side 225–3398 px). Class distribution differs strongly (train vs test instances): `dining table` 4 vs 112, `handbag` 5 vs 125, `curtain` 3 vs 107, `wallet` 197 vs 5, `stairs` 175 vs 6. Some test classes have zero train instances (`keyboard`, `mouse`, `toothbrush`, `teddy bear`, `hair drier`, `flower`, `scissors`). Train is only 397 images (43%), test 359.
- **Options considered:** —
- **Decision:** Treat evaluation as three levels: **valid** (in-distribution), **original test** (shifted), **phone test** (shifted further).
- **Evidence:** `results/data_validation.md`, `results/eda/class_shift.csv`, `results/eda/summary.md`.
- **Follow-up:** Discuss in the domain-shift section of the report.

### 2026-11-18 — Four images stored rotated relative to their annotations

- **Context:** Dataset validation (`python -m iiss.data.validate`).
- **Problem / observation:** 4 image files (train 182169107, 182169121; valid 182169108, 182169109) are 480×640 on disk while the JSON and polygons describe 640×480. No EXIF orientation to explain it.
- **Options considered:** edit the image files; drop the 4 images; fix the rotation at load time.
- **Decision:** Fix at load time via `data.rotation_fixes` in `configs/base.yaml` (90° or 270° CCW per file, each verified with a mask overlay). The dataset files stay untouched, consistent with the minimal-modification policy. Every pixel consumer uses `iiss.data.images.load_image`.
- **Evidence:** commit c7b2e75, `results/data_validation.md` (now PASS).
- **Follow-up:** none.

### 2026-11-18 — Things/stuff list for evaluation breakdowns

- **Context:** Evaluator breakdowns.
- **Problem / observation:** Need a things vs stuff split for analysis.
- **Options considered:** my own judgement per class; the official COCO panoptic definition.
- **Decision:** Use the 28 COCO panoptic "stuff" categories that exist in v5, plus `wall` (not a COCO name). This means `pillow`, `blanket`, `towel`, `curtain`, `light`, `shelf`, `counter` and `stairs` count as stuff, following COCO. Used for evaluation breakdowns only, never in training.
- **Evidence:** `configs/base.yaml` (`stuff_classes`).
- **Follow-up:** none.

### 2026-11-18 — Evaluator sanity check (GT as predictions)

- **Context:** Before any training, checked that the pycocotools evaluator scores ground truth as perfect.
- **Problem / observation:** (1) A match to GT annotation id 0 was counted as a false positive, because pycocotools stores the matched GT id and uses it as a boolean. (2) `maxDets=100` applies per image and category; one test image has 111 `book` instances, so even perfect predictions cannot reach AP 1.0 on test.
- **Options considered:** —
- **Decision:** Shift annotation ids to start at 1 in memory when loading GT. Keep the standard COCO protocol (`maxDets=100`) and report the ceiling: mask AP of perfect predictions = **1.0 on valid, 0.9971 on test**.
- **Evidence:** commit 1f03000, `results/eval/sanity_gt_as_pred/`.
- **Follow-up:** Show the 0.9971 ceiling next to test results.

### 2026-11-18 — Training/evaluation protocol for a fair comparison

- **Context:** After the first two model wrappers.
- **Problem / observation:** (1) YOLO selects `best` with the Ultralytics fitness (box + mask mix) while Mask R-CNN would select by pycocotools mask AP: different criteria. (2) Pasting Mask R-CNN masks directly at 12 MP resolution needs ~5 GB of GPU memory.
- **Options considered:** select `best` per model; use `last` for all; full-resolution mask paste vs predict at S and upsample.
- **Decision:** The evaluated checkpoint is **`last`** for every model (fixed schedules); `best` is saved and logged only. All models predict masks at input resolution S and are bilinearly upsampled to the original size (no `retina_masks`, no full-resolution paste). When gradient accumulation is used, the learning rate is scaled to the effective batch.
- **Evidence:** `src/iiss/predict.py` (default `weights/last.pt`), model wrappers.
- **Follow-up:** Effective mask resolution = min(model mask resolution, S raster); use this in the failure analysis.

### 2026-11-23 — VRAM probe crashed with CUDA out of memory

- **Context:** `notebooks/vram_probe.ipynb`, first full run.
- **Problem / observation:** The run crashed at YOLO S=960, batch 16: `AcceleratorError: CUDA error: out of memory` (inside Ultralytics' own OOM retry).
- **Options considered:** —
- **Decision:** Keep the crashed notebook as evidence. Rewrite the probe so each configuration runs in its own subprocess, any "out of memory" error counts as OOM, and peaks above the physical 6141 MB (Windows shared-memory spill) never count as fitting. Batch overrides only where VRAM forces them: `yolo_s960` batch 4 (nominal batch 64 kept), `m2f_s960` batch 1 × accumulate 2.
- **Evidence:** `results/vram_probe/attempt1_crashed/`, `results/vram_probe/recommendation.csv`, commit a9b8c91.
- **Follow-up:** none.

### 2026-11-23 — m2f_s960 trained locally despite the expected cap hit

- **Context:** Mask2Former at S=960.
- **Problem / observation:** The probe estimated 2.58 h for 50 epochs, above the 2 h cap.
- **Options considered:** run on Colab; reduce epochs for this run only; run locally and accept the cap.
- **Decision:** Run locally with the 2 h cap and report the cap hit. Reducing epochs for one run would break the "same epochs across S" rule.
- **Evidence:** `configs/runs/m2f_s960_seed42.yaml`, commit 84d06a3.
- **Follow-up:** See 2026-11-24 entry on the actual result.

### 2026-11-23 — Hard-killed sessions logged in the GPU time log

- **Context:** GPU time accounting (`runs/gpu_time_log.csv`).
- **Problem / observation:** <!-- AUTHOR: describe what happened (e.g. a session killed before it could write its log row) -->
- **Options considered:** —
- **Decision:** Log hard-killed sessions so the 2 h cap stays cumulative across resumes.
- **Evidence:** commit bce32e4.
- **Follow-up:** none.

### 2026-11-23 — Baselines at S=640 completed

- **Context:** Runs `yolo_s640_seed42`, `maskrcnn_s640_seed42`, `m2f_s640_seed42` (launched by me from the CLI).
- **Problem / observation:** All three finished their full schedules within the cap (YOLO 100 epochs 1257 s, Mask R-CNN 24 epochs 1494 s, Mask2Former 50 epochs 4751 s). YOLO `optimizer: auto` chose AdamW with lr ≈ 1.27e-4. In-training validation (mask mAP50-95; Ultralytics metric for YOLO, pycocotools for the others): YOLO 0.258, Mask R-CNN last 0.168, Mask2Former last 0.304 (best 0.329 at epoch 15, then declining).
- **Options considered:** —
- **Decision:** Keep these as the baselines. Final numbers come from `iiss.evaluate`, not from the in-training validators.
- **Evidence:** `runs/<run_id>/`, `runs/gpu_time_log.csv`.
- **Follow-up:** Mask2Former peaking at epoch 15 and then declining is an overfitting candidate for the report.

### 2026-11-24 — Epoch budgets locked

- **Context:** After the baselines and the VRAM probe.
- **Problem / observation:** —
- **Options considered:** —
- **Decision:** YOLO 100 epochs (batch 8; 960: batch 4), Mask R-CNN 24 epochs (milestones 16/22, batch 2, lr 0.0025), Mask2Former 50 epochs (batch 2; 960: 1 × accumulate 2, base lr 1e-4). Identical across S within each model.
- **Evidence:** commit ef1f003.
- **Follow-up:** none.

### 2026-11-24 — Ablation runs (S = 320 and 960)

- **Context:** The six ablation runs, 00:39–05:24.
- **Problem / observation:** I asked Claude Code to run the six ablations sequentially through the training notebooks (an exception to my rule that heavy runs are launched by me).
- **Options considered:** —
- **Decision:** Accepted; each run has its own executed notebook copy as evidence.
- **Evidence:** `notebooks/runs/train_<model>_s<S>.ipynb`, `runs/ablation_driver.log`, `runs/gpu_time_log.csv`, commit 2e3e93b.
- **Follow-up:** none.

### 2026-11-24 — m2f_s960 hit the cap after spilling into shared GPU memory

- **Context:** `m2f_s960_seed42`.
- **Problem / observation:** Peak memory reached 9918 MB, above the 6 GB physical VRAM, so Windows (WDDM) spilled into shared system memory from epoch 5 on and epochs slowed from ~207–515 s to ~950–990 s. The cap stopped training after 9 full epochs + ~74% of epoch 10 (of 50), so the poly LR schedule never decayed. Validation segm AP at the cap 0.306 (still rising). The VRAM probe (a few iterations only) underestimated the real peak, which varies with instances per image.
- **Options considered:** accept as a limitation; re-run on Colab T4 (16 GB, no spill) with the same 2 h cap; resume past the cap.
- **Decision:** <!-- AUTHOR: accept as limitation, or re-run on Colab -->. Not resumed: the cap is final per protocol.
- **Evidence:** `runs/m2f_s960_seed42/`, `runs/gpu_time_log.csv` (cap_hit = True).
- **Follow-up:** Its 960 result is not directly comparable to 320/640 (undertrained); state this next to the number.

### 2026-11-24 — Preliminary observations from in-training validation

- **Context:** Last-checkpoint valid scores per model × S (YOLO: Ultralytics metric, others: pycocotools), before running the unified evaluator.
- **Problem / observation:** YOLO 0.192 / 0.258 / 0.240 (960 below 640); Mask R-CNN 0.134 / 0.168 / 0.178 (monotonic); Mask2Former 0.257 / 0.304 / 0.306 (960 capped). Train images are 640×480, so S=960 upsamples every training image (EDA: 100% upscaled at 960).
- **Options considered:** —
- **Decision:** Hypothesis to test later: YOLO gains nothing from upsampled low-res input, while Mask R-CNN benefits from larger RoI features. Confirm with `iiss.evaluate` before drawing conclusions.
- **Evidence:** `runs/ablation_driver.log`, `results/eda/summary.md`.
- **Follow-up:** P7 evaluation.

### 2026-11-24 — Self-made phone test set

- **Context:** Assignment part 1 (≥ 40 photos, ≥ 10 hard).
- **Problem / observation:** (1) I chose to photograph only my bedroom, so domain variety is narrow. (2) The first upload went through WhatsApp, which stripped all EXIF (0/40) and recompressed to 1600×1600; replaced with the phone originals (Xiaomi M2101K7BNY, 3472×3472, EXIF present). (3) Many photos are close-ups of the same objects: 40 photos cover 9 scenes. (4) All 12 hard photos are low light (several nearly black, mean brightness ≈ 12/255); no backlight/occlusion/extreme-angle photos.
- **Options considered:** retake wide shots to replace near-duplicates; keep the set.
- **Decision:** Keep the 40 originals (28 `normal`, 12 `low_light`) and record the scene of each photo in `metadata.csv` so correlated photos can be acknowledged in the analysis. Annotate in Roboflow with Smart Polygon (no auto-labeling), all visible v5 classes including wall/floor/ceiling; objects with no v5 class (laundry basket, rice cooker) are not annotated. Export without preprocessing (no resize, no auto-orient).
- **Evidence:** `dataset/manual-benchmark/originals/`, `dataset/manual-benchmark/metadata.csv`, Roboflow screenshots <!-- AUTHOR: add paths -->.
- **Follow-up:** Annotate, export COCO Segmentation to `dataset/manual-benchmark/roboflow-export/`, run the importer.
