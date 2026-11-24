# Evaluation: maskrcnn_s320_seed42 on valid

- Generated: 2026-11-24T21:54:28+07:00
- Predictions: `runs/maskrcnn_s320_seed42/predictions/valid.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/valid/_annotations.coco.json`
- Images: 166, GT instances: 2022, predictions: 16309
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 13.4 | 21.1 | 13.9 | 1.2 | 9.4 | 18.0 | 13.6 | 25.4 | 25.9 | 3.2 | 21.6 | 33.0 |
| bbox | 14.8 | 22.6 | 15.4 | 1.7 | 10.1 | 20.3 | 15.2 | 28.8 | 29.4 | 4.3 | 22.1 | 37.9 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 13.9 | 21.9 | 0.8 | 9.8 | 19.6 |
| stuff | 29 | 12.6 | 20.0 | 1.8 | 8.7 | 16.0 |
| in_train | 65 | 13.6 | 21.5 | 1.2 | 9.6 | 18.0 |
| absent_in_train | 10 | 0.0 | 0.0 | n/a | 0.0 | n/a |
