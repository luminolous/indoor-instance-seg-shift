# Evaluation: m2f_s320_seed42 on valid

- Generated: 2026-11-24T22:21:39+07:00
- Predictions: `runs/m2f_s320_seed42/predictions/valid.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/valid/_annotations.coco.json`
- Images: 166, GT instances: 2022, predictions: 16321
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 25.7 | 35.9 | 25.4 | 2.4 | 16.6 | 36.3 | 25.4 | 39.6 | 40.4 | 4.0 | 27.0 | 52.2 |
| bbox | 25.5 | 33.8 | 25.4 | 3.1 | 15.5 | 36.5 | 26.1 | 40.7 | 41.5 | 4.9 | 25.6 | 54.5 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 26.0 | 36.2 | 2.9 | 20.4 | 37.9 |
| stuff | 29 | 25.3 | 35.3 | 1.5 | 10.3 | 34.3 |
| in_train | 65 | 26.2 | 36.5 | 2.4 | 17.0 | 36.3 |
| absent_in_train | 10 | 0.0 | 0.0 | n/a | 0.0 | n/a |
