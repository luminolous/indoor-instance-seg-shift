# Evaluation: m2f_s640_seed42 on test

- Generated: 2026-11-24T22:38:42+07:00
- Predictions: `runs/m2f_s640_seed42/predictions/test.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/test/_annotations.coco.json`
- Images: 359, GT instances: 5344, predictions: 35312
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 17.8 | 24.9 | 18.3 | 7.6 | 15.6 | 24.8 | 24.4 | 37.5 | 38.1 | 17.8 | 31.9 | 46.4 |
| bbox | 17.8 | 24.2 | 18.1 | 6.5 | 14.9 | 25.8 | 24.6 | 38.4 | 39.1 | 17.3 | 31.2 | 48.4 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 19.8 | 27.5 | 8.8 | 19.4 | 29.2 |
| stuff | 29 | 14.5 | 20.5 | 5.7 | 9.8 | 17.9 |
| in_train | 65 | 20.4 | 28.5 | 8.2 | 16.8 | 28.5 |
| absent_in_train | 10 | 0.3 | 0.4 | 0.0 | 4.2 | 0.6 |
