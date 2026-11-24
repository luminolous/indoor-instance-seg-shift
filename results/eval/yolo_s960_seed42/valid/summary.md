# Evaluation: yolo_s960_seed42 on valid

- Generated: 2026-11-24T21:33:01+07:00
- Predictions: `runs/yolo_s960_seed42/predictions/valid.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/valid/_annotations.coco.json`
- Images: 166, GT instances: 2022, predictions: 14322
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 24.1 | 32.7 | 25.4 | 12.0 | 19.2 | 28.2 | 19.1 | 34.1 | 35.6 | 20.5 | 33.7 | 39.1 |
| bbox | 26.3 | 33.8 | 27.7 | 14.2 | 20.5 | 31.1 | 19.9 | 37.2 | 39.0 | 23.1 | 35.6 | 43.4 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 27.0 | 37.0 | 11.3 | 22.9 | 30.8 |
| stuff | 29 | 19.7 | 26.3 | 13.1 | 13.1 | 24.8 |
| in_train | 65 | 24.5 | 33.3 | 12.0 | 19.6 | 28.2 |
| absent_in_train | 10 | 0.0 | 0.0 | n/a | 0.0 | n/a |
