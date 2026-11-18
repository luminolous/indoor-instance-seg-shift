# Evaluation: sanity_gt_as_pred on test

- Generated: 2026-11-18T08:56:09+07:00
- Predictions: `results/eval/sanity_gt_as_pred/test/standard/gt_as_predictions.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/test/_annotations.coco.json`
- Images: 359, GT instances: 5344, predictions: 5295
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 99.7 | 99.7 | 99.7 | 98.9 | 99.8 | 99.9 | 63.4 | 96.4 | 99.8 | 98.9 | 99.8 | 100.0 |
| bbox | 98.7 | 99.7 | 99.4 | 92.6 | 99.3 | 99.9 | 63.1 | 95.8 | 99.1 | 93.8 | 99.5 | 100.0 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 99.6 | 99.6 | 98.2 | 99.6 | 100.0 |
| stuff | 29 | 99.9 | 99.9 | 100.0 | 100.0 | 99.9 |
| in_train | 65 | 99.7 | 99.7 | 98.8 | 99.7 | 99.9 |
| absent_in_train | 10 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
