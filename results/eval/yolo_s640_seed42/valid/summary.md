# Evaluation: yolo_s640_seed42 on valid

- Generated: 2026-11-24T21:11:14+07:00
- Predictions: `runs/yolo_s640_seed42/predictions/valid.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/valid/_annotations.coco.json`
- Images: 166, GT instances: 2022, predictions: 14892
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 25.6 | 35.6 | 26.3 | 9.4 | 20.8 | 28.1 | 21.8 | 35.4 | 37.0 | 18.9 | 31.8 | 39.8 |
| bbox | 28.4 | 36.5 | 28.9 | 11.0 | 22.9 | 31.3 | 23.2 | 39.1 | 41.1 | 21.4 | 34.1 | 44.3 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 28.4 | 40.1 | 11.1 | 25.3 | 29.9 |
| stuff | 29 | 21.3 | 28.8 | 6.7 | 13.3 | 25.7 |
| in_train | 65 | 26.1 | 36.3 | 9.4 | 21.2 | 28.1 |
| absent_in_train | 10 | 0.0 | 0.0 | n/a | 0.0 | n/a |
