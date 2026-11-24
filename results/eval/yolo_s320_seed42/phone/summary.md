# Evaluation: yolo_s320_seed42 on phone

- Generated: 2026-11-24T21:09:42+07:00
- Predictions: `runs/yolo_s320_seed42/predictions/phone.json`
- Ground truth: `dataset/derived/phone/annotations.coco.json`
- Images: 40, GT instances: 166, predictions: 3164
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 5.5 | 9.0 | 6.2 | n/a | n/a | 5.5 | 7.0 | 12.2 | 12.9 | n/a | n/a | 12.9 |
| bbox | 7.1 | 10.1 | 6.7 | n/a | n/a | 7.1 | 9.3 | 16.5 | 17.5 | n/a | n/a | 17.5 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 3.3 | 4.5 | n/a | n/a | 3.3 |
| stuff | 29 | 7.4 | 13.0 | n/a | n/a | 7.4 |
| in_train | 65 | 5.5 | 9.0 | n/a | n/a | 5.5 |
| absent_in_train | 10 | n/a | n/a | n/a | n/a | n/a |

## Image groups (segm)

| Group | n_images | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| normal | 28 | 6.7 | 11.1 | n/a | n/a | 6.7 |
| low_light | 12 | 4.4 | 9.0 | n/a | n/a | 4.4 |
