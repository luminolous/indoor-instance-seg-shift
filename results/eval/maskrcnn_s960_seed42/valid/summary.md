# Evaluation: maskrcnn_s960_seed42 on valid

- Generated: 2026-11-24T22:11:23+07:00
- Predictions: `runs/maskrcnn_s960_seed42/predictions/valid.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/valid/_annotations.coco.json`
- Images: 166, GT instances: 2022, predictions: 16535
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 17.8 | 25.8 | 19.0 | 6.2 | 15.7 | 21.0 | 17.1 | 32.2 | 32.9 | 13.2 | 27.4 | 39.3 |
| bbox | 19.0 | 27.9 | 21.1 | 7.4 | 15.1 | 23.0 | 18.2 | 35.0 | 35.8 | 14.3 | 27.2 | 44.1 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 20.1 | 28.6 | 6.7 | 18.7 | 23.2 |
| stuff | 29 | 14.4 | 21.6 | 5.3 | 10.6 | 18.1 |
| in_train | 65 | 18.2 | 26.3 | 6.2 | 16.0 | 21.0 |
| absent_in_train | 10 | 0.0 | 0.0 | n/a | 0.0 | n/a |
