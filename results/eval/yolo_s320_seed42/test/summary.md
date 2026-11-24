# Evaluation: yolo_s320_seed42 on test

- Generated: 2026-11-24T21:01:47+07:00
- Predictions: `runs/yolo_s320_seed42/predictions/test.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/test/_annotations.coco.json`
- Images: 359, GT instances: 5344, predictions: 33379
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 7.1 | 12.4 | 7.2 | 4.4 | 7.2 | 9.9 | 10.7 | 17.1 | 17.8 | 7.8 | 16.5 | 20.8 |
| bbox | 8.8 | 13.2 | 9.1 | 5.1 | 8.0 | 12.3 | 12.3 | 20.4 | 21.4 | 9.2 | 18.6 | 25.1 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 7.0 | 12.3 | 5.1 | 9.5 | 10.0 |
| stuff | 29 | 7.4 | 12.6 | 3.2 | 3.7 | 9.6 |
| in_train | 65 | 8.2 | 14.3 | 4.8 | 7.9 | 11.4 |
| absent_in_train | 10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
