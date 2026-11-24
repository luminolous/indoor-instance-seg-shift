# Evaluation: m2f_s960_seed42 on phone

- Generated: 2026-11-24T23:10:52+07:00
- Predictions: `runs/m2f_s960_seed42/predictions/phone.json`
- Ground truth: `dataset/derived/phone/annotations.coco.json`
- Images: 40, GT instances: 166, predictions: 3993
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 22.6 | 27.7 | 22.7 | n/a | n/a | 22.6 | 29.6 | 43.3 | 43.9 | n/a | n/a | 43.9 |
| bbox | 22.9 | 27.2 | 23.1 | n/a | n/a | 22.9 | 30.9 | 45.1 | 46.1 | n/a | n/a | 46.1 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 27.5 | 30.5 | n/a | n/a | 27.5 |
| stuff | 29 | 18.3 | 25.3 | n/a | n/a | 18.3 |
| in_train | 65 | 22.6 | 27.7 | n/a | n/a | 22.6 |
| absent_in_train | 10 | n/a | n/a | n/a | n/a | n/a |

## Image groups (segm)

| Group | n_images | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| normal | 28 | 29.3 | 34.8 | n/a | n/a | 29.4 |
| low_light | 12 | 9.5 | 12.3 | n/a | n/a | 9.5 |
