# Evaluation: m2f_s640_seed42 on phone

- Generated: 2026-11-24T22:57:56+07:00
- Predictions: `runs/m2f_s640_seed42/predictions/phone.json`
- Ground truth: `dataset/derived/phone/annotations.coco.json`
- Images: 40, GT instances: 166, predictions: 4000
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 19.8 | 27.8 | 19.3 | n/a | n/a | 19.8 | 33.0 | 41.2 | 41.6 | n/a | n/a | 41.6 |
| bbox | 20.4 | 25.5 | 20.6 | n/a | n/a | 20.4 | 35.0 | 44.3 | 45.0 | n/a | n/a | 45.0 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 24.7 | 31.5 | n/a | n/a | 24.7 |
| stuff | 29 | 15.4 | 24.6 | n/a | n/a | 15.4 |
| in_train | 65 | 19.8 | 27.8 | n/a | n/a | 19.8 |
| absent_in_train | 10 | n/a | n/a | n/a | n/a | n/a |

## Image groups (segm)

| Group | n_images | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| normal | 28 | 25.1 | 33.6 | n/a | n/a | 25.1 |
| low_light | 12 | 7.8 | 10.6 | n/a | n/a | 7.8 |
