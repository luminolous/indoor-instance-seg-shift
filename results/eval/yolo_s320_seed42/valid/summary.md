# Evaluation: yolo_s320_seed42 on valid

- Generated: 2026-11-24T20:56:40+07:00
- Predictions: `runs/yolo_s320_seed42/predictions/valid.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/valid/_annotations.coco.json`
- Images: 166, GT instances: 2022, predictions: 14177
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 18.9 | 28.3 | 20.7 | 3.3 | 12.9 | 24.7 | 16.1 | 27.7 | 28.8 | 9.3 | 22.4 | 35.2 |
| bbox | 21.8 | 28.9 | 23.0 | 5.2 | 15.1 | 27.7 | 18.0 | 31.8 | 33.3 | 12.0 | 25.8 | 40.2 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 19.7 | 30.0 | 2.7 | 15.2 | 25.8 |
| stuff | 29 | 17.8 | 25.7 | 4.3 | 9.2 | 23.2 |
| in_train | 65 | 19.3 | 28.8 | 3.3 | 13.2 | 24.7 |
| absent_in_train | 10 | 0.0 | 0.0 | n/a | 0.0 | n/a |
