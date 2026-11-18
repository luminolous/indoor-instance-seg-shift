# Evaluation: sanity_gt_as_pred on valid

- Generated: 2026-11-18T08:54:55+07:00
- Predictions: `results/eval/sanity_gt_as_pred/valid/standard/gt_as_predictions.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/valid/_annotations.coco.json`
- Images: 166, GT instances: 2022, predictions: 2022
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 61.6 | 97.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| bbox | 99.0 | 100.0 | 100.0 | 95.1 | 99.1 | 100.0 | 61.3 | 96.4 | 99.3 | 96.2 | 99.3 | 100.0 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| stuff | 29 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| in_train | 65 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| absent_in_train | 10 | 100.0 | 100.0 | n/a | 100.0 | n/a |
