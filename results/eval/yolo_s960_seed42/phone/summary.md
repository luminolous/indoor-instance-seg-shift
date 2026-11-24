# Evaluation: yolo_s960_seed42 on phone

- Generated: 2026-11-24T21:53:45+07:00
- Predictions: `runs/yolo_s960_seed42/predictions/phone.json`
- Ground truth: `dataset/derived/phone/annotations.coco.json`
- Images: 40, GT instances: 166, predictions: 3810
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 5.3 | 7.6 | 5.8 | n/a | n/a | 5.3 | 6.6 | 11.4 | 12.0 | n/a | n/a | 12.0 |
| bbox | 7.0 | 10.7 | 7.5 | n/a | n/a | 7.0 | 8.2 | 14.2 | 14.6 | n/a | n/a | 14.6 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 4.2 | 5.3 | n/a | n/a | 4.2 |
| stuff | 29 | 6.2 | 9.7 | n/a | n/a | 6.2 |
| in_train | 65 | 5.3 | 7.6 | n/a | n/a | 5.3 |
| absent_in_train | 10 | n/a | n/a | n/a | n/a | n/a |

## Image groups (segm)

| Group | n_images | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| normal | 28 | 7.5 | 10.8 | n/a | n/a | 7.5 |
| low_light | 12 | 0.0 | 0.1 | n/a | n/a | 0.0 |
