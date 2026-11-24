# Evaluation: m2f_s320_seed42 on phone

- Generated: 2026-11-24T22:31:51+07:00
- Predictions: `runs/m2f_s320_seed42/predictions/phone.json`
- Ground truth: `dataset/derived/phone/annotations.coco.json`
- Images: 40, GT instances: 166, predictions: 3998
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 25.5 | 32.2 | 25.6 | n/a | n/a | 25.5 | 29.3 | 45.2 | 45.2 | n/a | n/a | 45.2 |
| bbox | 25.7 | 31.2 | 26.6 | n/a | n/a | 25.7 | 30.1 | 47.5 | 47.5 | n/a | n/a | 47.5 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 32.5 | 38.2 | n/a | n/a | 32.5 |
| stuff | 29 | 19.3 | 26.9 | n/a | n/a | 19.3 |
| in_train | 65 | 25.5 | 32.2 | n/a | n/a | 25.5 |
| absent_in_train | 10 | n/a | n/a | n/a | n/a | n/a |

## Image groups (segm)

| Group | n_images | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| normal | 28 | 30.1 | 37.5 | n/a | n/a | 30.1 |
| low_light | 12 | 13.6 | 17.8 | n/a | n/a | 13.6 |
