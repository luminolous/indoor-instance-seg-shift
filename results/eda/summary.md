# EDA summary

- Generated: 2026-11-18T00:31:13+07:00
- Dataset root: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation`
- Command: `python -m iiss.data.eda`
- Source: annotation JSON only. Generated file; do not edit by hand.

## Longest image side (px) and resize direction per input size S

| split | images | min | p25 | median | p75 | max | upscaled_pct_at_320 | downscaled_pct_at_320 | upscaled_pct_at_640 | downscaled_pct_at_640 | upscaled_pct_at_960 | downscaled_pct_at_960 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 397 | 561 | 640.0 | 640.0 | 640.0 | 857 | 0.0 | 100.0 | 0.25 | 4.03 | 100.0 | 0.0 |
| valid | 166 | 640 | 640.0 | 640.0 | 640.0 | 857 | 0.0 | 100.0 | 0.0 | 3.61 | 100.0 | 0.0 |
| test | 359 | 225 | 600.0 | 800.0 | 1280.0 | 3398 | 14.76 | 85.24 | 28.13 | 67.97 | 58.22 | 40.39 |

`upscaled_pct_at_S` = share of images whose longest side is smaller than S.

## Instances per image

| split | images | mean | median | p90 | max | zero |
|---|---|---|---|---|---|---|
| train | 397 | 11.28 | 8.0 | 22.0 | 61 | 0 |
| valid | 166 | 12.18 | 9.0 | 21.5 | 79 | 1 |
| test | 359 | 14.89 | 12.0 | 26.0 | 129 | 0 |

## Object size bins (COCO: small < 32^2, medium < 96^2, large >= 96^2 px^2)

`native` = original resolution (what COCOeval APs/APm/APl use).
`S=...` = after resizing the longest side to S (what the model sees).

| split | resolution | instances | small_pct | medium_pct | large_pct | median_sqrt_area_px |
|---|---|---|---|---|---|---|
| train | native | 4479 | 11.03 | 31.41 | 57.56 | 119.0 |
| train | S=320 | 4479 | 28.64 | 39.18 | 32.17 | 59.0 |
| train | S=640 | 4479 | 11.07 | 31.44 | 57.49 | 118.0 |
| train | S=960 | 4479 | 4.89 | 23.76 | 71.36 | 177.0 |
| valid | native | 2022 | 9.64 | 33.58 | 56.78 | 115.1 |
| valid | S=320 | 2022 | 28.09 | 41.35 | 30.56 | 57.4 |
| valid | S=640 | 2022 | 9.79 | 33.53 | 56.68 | 114.8 |
| valid | S=960 | 2022 | 3.76 | 24.33 | 71.91 | 172.1 |
| test | native | 5344 | 10.74 | 31.92 | 57.34 | 117.7 |
| test | S=320 | 5344 | 36.88 | 42.53 | 20.58 | 45.3 |
| test | S=640 | 5344 | 16.9 | 35.78 | 47.32 | 90.6 |
| test | S=960 | 5344 | 9.0 | 27.88 | 63.12 | 135.8 |

## Train vs test class shift

Classes present in test but absent in train (9): `toothbrush` (14), `mouse` (8), `keyboard` (7), `flower` (6), `scissors` (6), `teddy bear` (6), `toaster` (4), `hair drier` (2), `scissor` (2)

Classes present in train but absent in test (5): `remote` (9), `wall` (2), `shelf` (2), `st-` (1), `tree-merged` (1)

Largest share increases from train to test (percentage points):

| name | train_instances | test_instances | train_share_pct | test_share_pct | share_diff_pp |
|---|---|---|---|---|---|
| light | 41 | 295 | 0.915 | 5.52 | 4.605 |
| book | 105 | 349 | 2.344 | 6.531 | 4.187 |
| person | 192 | 413 | 4.287 | 7.728 | 3.441 |
| handbag | 5 | 125 | 0.112 | 2.339 | 2.227 |
| dining table | 4 | 112 | 0.089 | 2.096 | 2.007 |
| curtain | 3 | 107 | 0.067 | 2.002 | 1.935 |
| cup | 25 | 131 | 0.558 | 2.451 | 1.893 |
| chair | 217 | 353 | 4.845 | 6.606 | 1.761 |
| rug-merged | 35 | 107 | 0.781 | 2.002 | 1.221 |
| ceiling-merged | 77 | 149 | 1.719 | 2.788 | 1.069 |

Largest share decreases from train to test:

| name | train_instances | test_instances | train_share_pct | test_share_pct | share_diff_pp |
|---|---|---|---|---|---|
| wall-other-merged | 756 | 650 | 16.879 | 12.163 | -4.716 |
| pan | 225 | 34 | 5.023 | 0.636 | -4.387 |
| wallet | 197 | 5 | 4.398 | 0.094 | -4.304 |
| toy | 217 | 41 | 4.845 | 0.767 | -4.078 |
| stairs | 175 | 6 | 3.907 | 0.112 | -3.795 |
| trash-bin | 153 | 4 | 3.416 | 0.075 | -3.341 |
| table-merged | 319 | 258 | 7.122 | 4.828 | -2.294 |
| box | 95 | 1 | 2.121 | 0.019 | -2.102 |
| floor-other-merged | 261 | 239 | 5.827 | 4.472 | -1.355 |
| counter | 102 | 67 | 2.277 | 1.254 | -1.023 |

Full tables: `class_counts.csv`, `class_shift.csv`.

## Figures

- `class_counts.png`
- `long_side_hist.png`
- `instances_per_image_hist.png`
- `small_objects_vs_S.png`

Total instances: train 4479, valid 2022, test 5344
