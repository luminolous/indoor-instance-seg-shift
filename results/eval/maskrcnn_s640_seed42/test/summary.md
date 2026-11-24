# Evaluation: maskrcnn_s640_seed42 on test

- Generated: 2026-11-24T22:05:26+07:00
- Predictions: `runs/maskrcnn_s640_seed42/predictions/test.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/test/_annotations.coco.json`
- Images: 359, GT instances: 5344, predictions: 35676
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 9.0 | 14.9 | 9.1 | 4.5 | 9.6 | 12.2 | 13.0 | 22.6 | 23.1 | 9.3 | 21.8 | 27.1 |
| bbox | 10.2 | 16.0 | 10.6 | 4.3 | 10.0 | 14.1 | 14.2 | 25.0 | 25.7 | 9.7 | 22.4 | 30.9 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 10.0 | 16.4 | 5.2 | 12.1 | 14.2 |
| stuff | 29 | 7.4 | 12.4 | 3.3 | 5.8 | 9.1 |
| in_train | 65 | 10.4 | 17.1 | 4.9 | 10.7 | 14.1 |
| absent_in_train | 10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
