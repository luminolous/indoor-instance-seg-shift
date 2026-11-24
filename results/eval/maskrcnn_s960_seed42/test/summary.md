# Evaluation: maskrcnn_s960_seed42 on test

- Generated: 2026-11-24T22:14:56+07:00
- Predictions: `runs/maskrcnn_s960_seed42/predictions/test.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/test/_annotations.coco.json`
- Images: 359, GT instances: 5344, predictions: 35748
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 9.5 | 15.4 | 9.8 | 5.1 | 8.9 | 13.0 | 13.8 | 24.1 | 24.7 | 14.2 | 21.9 | 28.7 |
| bbox | 10.8 | 16.7 | 11.6 | 5.3 | 8.9 | 15.2 | 15.5 | 27.0 | 27.7 | 14.8 | 22.0 | 33.4 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 10.5 | 16.8 | 6.1 | 11.0 | 15.1 |
| stuff | 29 | 7.8 | 13.2 | 3.4 | 5.7 | 9.6 |
| in_train | 65 | 10.9 | 17.8 | 5.5 | 9.8 | 15.0 |
| absent_in_train | 10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
