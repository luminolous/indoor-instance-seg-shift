# Evaluation: yolo_s960_seed42 on test

- Generated: 2026-11-24T21:41:39+07:00
- Predictions: `runs/yolo_s960_seed42/predictions/test.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/test/_annotations.coco.json`
- Images: 359, GT instances: 5344, predictions: 33413
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 10.4 | 16.1 | 11.0 | 6.6 | 11.4 | 12.4 | 12.2 | 21.1 | 22.9 | 15.6 | 24.6 | 23.2 |
| bbox | 12.1 | 16.8 | 12.7 | 7.6 | 12.4 | 14.7 | 13.4 | 23.9 | 25.9 | 17.3 | 26.6 | 26.7 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 11.5 | 17.3 | 7.0 | 14.5 | 13.5 |
| stuff | 29 | 8.7 | 14.1 | 6.0 | 6.7 | 10.6 |
| in_train | 65 | 12.0 | 18.5 | 7.2 | 12.6 | 14.3 |
| absent_in_train | 10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
