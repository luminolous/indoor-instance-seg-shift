# Evaluation: yolo_s640_seed42 on test

- Generated: 2026-11-24T21:16:16+07:00
- Predictions: `runs/yolo_s640_seed42/predictions/test.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/test/_annotations.coco.json`
- Images: 359, GT instances: 5344, predictions: 33846
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 10.8 | 16.7 | 11.4 | 5.5 | 12.6 | 13.9 | 12.4 | 21.0 | 21.9 | 11.9 | 24.2 | 24.6 |
| bbox | 12.4 | 17.0 | 12.9 | 6.4 | 13.5 | 16.0 | 13.8 | 24.0 | 25.1 | 13.4 | 26.8 | 28.4 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 11.3 | 17.3 | 5.0 | 14.5 | 15.1 |
| stuff | 29 | 9.9 | 15.7 | 6.4 | 9.6 | 11.9 |
| in_train | 65 | 12.4 | 19.2 | 6.0 | 13.9 | 16.0 |
| absent_in_train | 10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
