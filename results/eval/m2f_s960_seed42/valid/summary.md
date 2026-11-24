# Evaluation: m2f_s960_seed42 on valid

- Generated: 2026-11-24T23:00:29+07:00
- Predictions: `runs/m2f_s960_seed42/predictions/valid.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/valid/_annotations.coco.json`
- Images: 166, GT instances: 2022, predictions: 16370
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 30.6 | 40.8 | 32.9 | 7.4 | 26.8 | 34.9 | 29.9 | 48.9 | 50.8 | 16.2 | 42.0 | 60.7 |
| bbox | 30.8 | 38.7 | 33.0 | 7.3 | 25.4 | 36.2 | 30.1 | 49.8 | 51.9 | 15.4 | 40.0 | 63.7 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 33.6 | 45.9 | 10.6 | 35.0 | 36.6 |
| stuff | 29 | 26.0 | 33.2 | 2.3 | 13.2 | 32.8 |
| in_train | 65 | 31.1 | 41.5 | 7.4 | 27.4 | 34.9 |
| absent_in_train | 10 | 0.0 | 0.0 | n/a | 0.0 | n/a |
