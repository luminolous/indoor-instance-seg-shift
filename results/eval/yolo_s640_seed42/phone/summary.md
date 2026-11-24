# Evaluation: yolo_s640_seed42 on phone

- Generated: 2026-11-24T21:30:16+07:00
- Predictions: `runs/yolo_s640_seed42/predictions/phone.json`
- Ground truth: `dataset/derived/phone/annotations.coco.json`
- Images: 40, GT instances: 166, predictions: 3878
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 7.2 | 9.4 | 7.9 | n/a | n/a | 7.3 | 7.1 | 14.5 | 14.7 | n/a | n/a | 14.7 |
| bbox | 7.1 | 9.5 | 6.7 | n/a | n/a | 7.2 | 6.5 | 16.7 | 17.1 | n/a | n/a | 17.1 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 6.4 | 7.3 | n/a | n/a | 6.6 |
| stuff | 29 | 7.9 | 11.2 | n/a | n/a | 7.9 |
| in_train | 65 | 7.2 | 9.4 | n/a | n/a | 7.3 |
| absent_in_train | 10 | n/a | n/a | n/a | n/a | n/a |

## Image groups (segm)

| Group | n_images | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| normal | 28 | 10.1 | 13.1 | n/a | n/a | 10.3 |
| low_light | 12 | 0.0 | 0.1 | n/a | n/a | 0.0 |
