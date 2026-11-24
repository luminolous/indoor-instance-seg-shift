# Evaluation: maskrcnn_s640_seed42 on phone

- Generated: 2026-11-24T22:10:15+07:00
- Predictions: `runs/maskrcnn_s640_seed42/predictions/phone.json`
- Ground truth: `dataset/derived/phone/annotations.coco.json`
- Images: 40, GT instances: 166, predictions: 3130
- Scorer: pycocotools COCOeval. Values are AP/AR x 100.

## Summary

| IoU type | AP | AP50 | AP75 | APs | APm | APl | AR1 | AR10 | AR100 | ARs | ARm | ARl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| segm | 8.4 | 14.5 | 7.5 | n/a | n/a | 8.4 | 14.8 | 23.4 | 23.4 | n/a | n/a | 23.4 |
| bbox | 11.6 | 19.0 | 13.8 | n/a | n/a | 11.6 | 19.4 | 31.0 | 31.0 | n/a | n/a | 31.0 |

## Category groups (segm)

| Group | n_categories | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| things | 46 | 9.3 | 15.4 | n/a | n/a | 9.3 |
| stuff | 29 | 7.6 | 13.7 | n/a | n/a | 7.6 |
| in_train | 65 | 8.4 | 14.5 | n/a | n/a | 8.4 |
| absent_in_train | 10 | n/a | n/a | n/a | n/a | n/a |

## Image groups (segm)

| Group | n_images | AP | AP50 | APs | APm | APl |
|---|---|---|---|---|---|---|
| normal | 28 | 10.7 | 17.5 | n/a | n/a | 10.7 |
| low_light | 12 | 5.5 | 11.7 | n/a | n/a | 5.5 |
