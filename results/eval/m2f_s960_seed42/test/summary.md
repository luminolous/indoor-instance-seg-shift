# Evaluation: m2f_s960_seed42 on test

- Generated: 2026-11-24T23:04:28+07:00
- Predictions: `runs/m2f_s960_seed42/predictions/test.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/test/_annotations.coco.json`
- Images: 359, GT instances: 5344, predictions: 35294
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 15.8 | 22.5 | 16.9 | 6.8 | 13.5 | 23.0 | 20.5 | 34.8 | 35.9 | 16.5 | 31.8 | 42.7 |
| bbox | 15.8 | 21.8 | 16.3 | 7.4 | 12.6 | 23.7 | 20.7 | 35.0 | 36.6 | 17.8 | 31.4 | 44.1 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 17.2 | 24.4 | 9.7 | 16.7 | 26.8 |
| stuff | 29 | 13.5 | 19.3 | 2.1 | 8.5 | 17.0 |
| in_train | 65 | 18.2 | 25.9 | 7.4 | 14.9 | 26.5 |
| absent_in_train | 10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
