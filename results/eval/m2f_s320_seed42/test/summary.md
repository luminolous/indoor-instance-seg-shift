# Evaluation: m2f_s320_seed42 on test

- Generated: 2026-11-24T22:24:01+07:00
- Predictions: `runs/m2f_s320_seed42/predictions/test.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/test/_annotations.coco.json`
- Images: 359, GT instances: 5344, predictions: 34868
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 12.8 | 19.8 | 12.5 | 3.2 | 12.1 | 17.9 | 19.0 | 28.9 | 29.2 | 9.6 | 23.5 | 36.1 |
| bbox | 13.6 | 20.2 | 13.5 | 3.8 | 11.3 | 19.6 | 19.9 | 30.5 | 30.9 | 10.7 | 22.1 | 38.8 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 14.0 | 21.6 | 3.8 | 15.6 | 20.8 |
| stuff | 29 | 10.7 | 16.8 | 2.4 | 6.8 | 13.2 |
| in_train | 65 | 14.6 | 22.7 | 3.5 | 12.5 | 20.6 |
| absent_in_train | 10 | 0.3 | 0.3 | 0.0 | 8.4 | 0.0 |
