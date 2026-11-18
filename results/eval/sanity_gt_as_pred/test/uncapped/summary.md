# Evaluation: sanity_gt_as_pred on test

- Generated: 2026-11-18T08:55:36+07:00
- Predictions: `results/eval/sanity_gt_as_pred/test/uncapped/gt_as_predictions.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/test/_annotations.coco.json`
- Images: 359, GT instances: 5344, predictions: 5344
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 63.5 | 96.5 | 100.0 | 100.0 | 100.0 | 100.0 |
| bbox | 98.9 | 100.0 | 99.7 | 93.6 | 99.6 | 100.0 | 63.1 | 95.9 | 99.3 | 94.8 | 99.7 | 100.0 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stuff | 29 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| in_train | 65 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| absent_in_train | 10 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
