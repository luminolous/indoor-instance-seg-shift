# Evaluation: m2f_s640_seed42 on valid

- Generated: 2026-11-24T22:32:28+07:00
- Predictions: `runs/m2f_s640_seed42/predictions/valid.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/valid/_annotations.coco.json`
- Images: 166, GT instances: 2022, predictions: 16374
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 30.4 | 40.3 | 30.7 | 7.6 | 21.2 | 40.8 | 29.7 | 46.3 | 47.4 | 12.2 | 35.5 | 58.8 |
| bbox | 30.4 | 39.4 | 31.2 | 7.9 | 20.2 | 41.0 | 30.6 | 47.4 | 48.4 | 13.2 | 34.4 | 61.3 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 32.3 | 42.8 | 10.0 | 25.4 | 43.8 |
| stuff | 29 | 27.6 | 36.6 | 3.6 | 14.2 | 36.8 |
| in_train | 65 | 30.9 | 41.0 | 7.6 | 21.7 | 40.8 |
| absent_in_train | 10 | 0.0 | 0.0 | n/a | 0.0 | n/a |
