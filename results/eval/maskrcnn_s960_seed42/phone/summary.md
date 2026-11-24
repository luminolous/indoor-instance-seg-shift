# Evaluation: maskrcnn_s960_seed42 on phone

- Generated: 2026-11-24T22:20:35+07:00
- Predictions: `runs/maskrcnn_s960_seed42/predictions/phone.json`
- Ground truth: `dataset/derived/phone/annotations.coco.json`
- Images: 40, GT instances: 166, predictions: 3590
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 8.7 | 13.9 | 10.0 | n/a | n/a | 8.9 | 19.6 | 26.8 | 26.8 | n/a | n/a | 26.8 |
| bbox | 11.8 | 18.9 | 12.5 | n/a | n/a | 11.9 | 24.7 | 34.4 | 34.4 | n/a | n/a | 34.4 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 7.1 | 12.0 | n/a | n/a | 7.1 |
| stuff | 29 | 10.1 | 15.6 | n/a | n/a | 10.4 |
| in_train | 65 | 8.7 | 13.9 | n/a | n/a | 8.9 |
| absent_in_train | 10 | n/a | n/a | n/a | n/a | n/a |

## Image groups (segm)

| Group | n_images | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| normal | 28 | 9.7 | 15.3 | n/a | n/a | 9.8 |
| low_light | 12 | 9.8 | 14.4 | n/a | n/a | 9.8 |
