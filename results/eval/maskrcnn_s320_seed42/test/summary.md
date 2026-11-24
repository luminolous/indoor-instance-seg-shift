# Evaluation: maskrcnn_s320_seed42 on test

- Generated: 2026-11-24T21:57:07+07:00
- Predictions: `runs/maskrcnn_s320_seed42/predictions/test.json`
- Ground truth: `dataset/panoptic-indoor-segmentation.v5i.coco-segmentation/test/_annotations.coco.json`
- Images: 359, GT instances: 5344, predictions: 35515
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 6.5 | 11.3 | 6.7 | 1.7 | 5.8 | 9.4 | 10.7 | 18.0 | 18.3 | 6.3 | 13.6 | 22.9 |
| bbox | 7.7 | 12.9 | 8.2 | 1.9 | 6.0 | 11.4 | 12.4 | 20.6 | 21.0 | 6.4 | 14.3 | 26.7 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 6.7 | 11.3 | 1.5 | 7.1 | 10.1 |
| stuff | 29 | 6.3 | 11.4 | 2.0 | 3.7 | 8.1 |
| in_train | 65 | 7.5 | 13.0 | 1.8 | 6.4 | 10.8 |
| absent_in_train | 10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
