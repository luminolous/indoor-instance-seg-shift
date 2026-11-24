# Evaluation: maskrcnn_s640_seed42 on valid

- Generated: 2026-11-24T22:02:29+07:00
- Predictions: `runs/maskrcnn_s640_seed42/predictions/valid.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/valid/_annotations.coco.json`
- Images: 166, GT instances: 2022, predictions: 16497
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 16.7 | 25.9 | 17.7 | 5.0 | 12.6 | 20.3 | 16.6 | 29.2 | 30.0 | 10.8 | 27.1 | 36.2 |
| bbox | 18.6 | 27.7 | 20.6 | 5.9 | 13.0 | 22.9 | 18.1 | 33.1 | 33.8 | 13.6 | 26.8 | 42.1 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 17.9 | 27.5 | 5.6 | 15.6 | 21.3 |
| stuff | 29 | 14.9 | 23.4 | 4.0 | 7.7 | 18.8 |
| in_train | 65 | 17.0 | 26.3 | 5.0 | 12.9 | 20.3 |
| absent_in_train | 10 | 0.0 | 0.0 | n/a | 0.0 | n/a |
