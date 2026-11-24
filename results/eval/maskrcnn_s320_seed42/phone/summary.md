# Evaluation: maskrcnn_s320_seed42 on phone

- Generated: 2026-11-24T22:01:50+07:00
- Predictions: `runs/maskrcnn_s320_seed42/predictions/phone.json`
- Ground truth: `dataset/derived/phone/annotations.coco.json`
- Images: 40, GT instances: 166, predictions: 3010
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 6.8 | 11.4 | 7.0 | n/a | n/a | 6.8 | 15.4 | 21.3 | 21.3 | n/a | n/a | 21.3 |
| bbox | 8.3 | 14.0 | 8.1 | n/a | n/a | 8.3 | 19.0 | 28.3 | 28.3 | n/a | n/a | 28.3 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 7.6 | 10.6 | n/a | n/a | 7.6 |
| stuff | 29 | 6.0 | 12.2 | n/a | n/a | 6.0 |
| in_train | 65 | 6.8 | 11.4 | n/a | n/a | 6.8 |
| absent_in_train | 10 | n/a | n/a | n/a | n/a | n/a |

## Image groups (segm)

| Group | n_images | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| normal | 28 | 9.8 | 15.2 | n/a | n/a | 9.8 |
| low_light | 12 | 3.7 | 8.1 | n/a | n/a | 3.7 |
