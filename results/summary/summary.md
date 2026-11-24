# Results summary

Generated 2026-11-24T23:17:21+07:00 by `python -m iiss.analysis.domain_shift` from `results/eval/`. All values are mask AP x 100 (pycocotools, `last` checkpoint), unless noted.

- Ground truth: valid: 58 classes / 2022 instances / 166 images, test: 69 classes / 5344 instances / 359 images, phone: 17 classes / 166 instances / 40 images.
- *Common* columns average over the 17 classes with GT in all three sets.
- A run marked `*` hit the 2-hour training cap and is undertrained.
- APs/APm/APl use the `area` stored in the GT (bounding-box area, as Roboflow exports it). On the phone set every object is large, so APs and APm are undefined.
- The phone set has few instances per class, so small differences are not meaningful.
- Inference time is not tabulated: the recorded times were affected by other load.

## Mask AP by model, input size and eval set

| Run | AP valid | AP test | AP phone | AP50 valid | AP50 test | AP50 phone | AP valid (common) | AP test (common) | AP phone (common) |
|---|---|---|---|---|---|---|---|---|---|
| YOLO11s-seg 320 | 18.9 | 7.1 | 5.5 | 28.3 | 12.4 | 9.0 | 15.5 | 10.9 | 5.5 |
| YOLO11s-seg 640 | 25.6 | 10.8 | 7.2 | 35.6 | 16.7 | 9.4 | 20.0 | 16.0 | 7.2 |
| YOLO11s-seg 960 | 24.1 | 10.4 | 5.3 | 32.7 | 16.1 | 7.6 | 19.7 | 15.1 | 5.3 |
| Mask R-CNN R50-FPN v2 320 | 13.4 | 6.5 | 6.8 | 21.1 | 11.3 | 11.4 | 12.0 | 10.5 | 6.8 |
| Mask R-CNN R50-FPN v2 640 | 16.7 | 9.0 | 8.4 | 25.9 | 14.9 | 14.5 | 15.6 | 13.8 | 8.4 |
| Mask R-CNN R50-FPN v2 960 | 17.8 | 9.5 | 8.7 | 25.8 | 15.4 | 13.9 | 17.3 | 14.8 | 8.7 |
| Mask2Former 320 | 25.7 | 12.8 | 25.5 | 35.9 | 19.8 | 32.2 | 22.5 | 16.3 | 25.5 |
| Mask2Former 640 | 30.4 | 17.8 | 19.8 | 40.3 | 24.9 | 27.8 | 26.7 | 21.3 | 19.8 |
| Mask2Former 960* | 30.6 | 15.8 | 22.6 | 40.8 | 22.5 | 27.7 | 25.5 | 20.6 | 22.6 |

## Domain shift (common classes)

Change from the earlier to the later set: AP points (relative).

| Run | valid -> test | test -> phone | valid -> phone |
|---|---|---|---|
| YOLO11s-seg 320 | -4.6 (-30%) | -5.4 (-49%) | -10.0 (-65%) |
| YOLO11s-seg 640 | -3.9 (-20%) | -8.8 (-55%) | -12.8 (-64%) |
| YOLO11s-seg 960 | -4.6 (-23%) | -9.8 (-65%) | -14.5 (-73%) |
| Mask R-CNN R50-FPN v2 320 | -1.4 (-12%) | -3.8 (-36%) | -5.2 (-43%) |
| Mask R-CNN R50-FPN v2 640 | -1.9 (-12%) | -5.4 (-39%) | -7.3 (-46%) |
| Mask R-CNN R50-FPN v2 960 | -2.5 (-14%) | -6.1 (-41%) | -8.6 (-50%) |
| Mask2Former 320 | -6.3 (-28%) | +9.3 (+57%) | +3.0 (+13%) |
| Mask2Former 640 | -5.4 (-20%) | -1.5 (-7%) | -6.9 (-26%) |
| Mask2Former 960* | -4.8 (-19%) | +2.0 (+10%) | -2.8 (-11%) |

## Domain shift (all classes with GT)

Change from the earlier to the later set: AP points (relative).

| Run | valid -> test | test -> phone | valid -> phone |
|---|---|---|---|
| YOLO11s-seg 320 | -11.8 (-62%) | -1.7 (-23%) | -13.4 (-71%) |
| YOLO11s-seg 640 | -14.9 (-58%) | -3.6 (-33%) | -18.4 (-72%) |
| YOLO11s-seg 960 | -13.7 (-57%) | -5.2 (-50%) | -18.9 (-78%) |
| Mask R-CNN R50-FPN v2 320 | -6.9 (-51%) | +0.3 (+4%) | -6.6 (-49%) |
| Mask R-CNN R50-FPN v2 640 | -7.7 (-46%) | -0.7 (-7%) | -8.4 (-50%) |
| Mask R-CNN R50-FPN v2 960 | -8.4 (-47%) | -0.8 (-8%) | -9.1 (-51%) |
| Mask2Former 320 | -13.0 (-50%) | +12.8 (+100%) | -0.2 (-1%) |
| Mask2Former 640 | -12.6 (-42%) | +2.0 (+11%) | -10.6 (-35%) |
| Mask2Former 960* | -14.8 (-48%) | +6.8 (+43%) | -8.0 (-26%) |

## Phone set by condition

normal: 28 images, 123 instances, 17 classes; low_light: 12 images, 43 instances, 12 classes. *Common* = the 12 classes with GT in every condition.

| Run | AP normal | AP low_light | AP normal (common) | AP low_light (common) |
|---|---|---|---|---|
| YOLO11s-seg 320 | 6.7 | 4.4 | 9.5 | 4.4 |
| YOLO11s-seg 640 | 10.1 | 0.0 | 14.3 | 0.0 |
| YOLO11s-seg 960 | 7.5 | 0.0 | 10.6 | 0.0 |
| Mask R-CNN R50-FPN v2 320 | 9.8 | 3.7 | 13.4 | 3.7 |
| Mask R-CNN R50-FPN v2 640 | 10.7 | 5.5 | 12.8 | 5.5 |
| Mask R-CNN R50-FPN v2 960 | 9.7 | 9.8 | 12.9 | 9.8 |
| Mask2Former 320 | 30.1 | 13.6 | 32.1 | 13.6 |
| Mask2Former 640 | 25.1 | 7.8 | 29.0 | 7.8 |
| Mask2Former 960* | 29.3 | 9.5 | 30.9 | 9.5 |

## Category groups (mask AP)

| Run | valid things | valid stuff | test things | test stuff | phone things | phone stuff | test in train | test absent in train |
|---|---|---|---|---|---|---|---|---|
| YOLO11s-seg 320 | 19.7 | 17.8 | 7.0 | 7.4 | 3.3 | 7.4 | 8.2 | 0.0 |
| YOLO11s-seg 640 | 28.4 | 21.3 | 11.3 | 9.9 | 6.4 | 7.9 | 12.4 | 0.0 |
| YOLO11s-seg 960 | 27.0 | 19.7 | 11.5 | 8.7 | 4.2 | 6.2 | 12.0 | 0.0 |
| Mask R-CNN R50-FPN v2 320 | 13.9 | 12.6 | 6.7 | 6.3 | 7.6 | 6.0 | 7.5 | 0.0 |
| Mask R-CNN R50-FPN v2 640 | 17.9 | 14.9 | 10.0 | 7.4 | 9.3 | 7.6 | 10.4 | 0.0 |
| Mask R-CNN R50-FPN v2 960 | 20.1 | 14.4 | 10.5 | 7.8 | 7.1 | 10.1 | 10.9 | 0.0 |
| Mask2Former 320 | 26.0 | 25.3 | 14.0 | 10.7 | 32.5 | 19.3 | 14.6 | 0.3 |
| Mask2Former 640 | 32.3 | 27.6 | 19.8 | 14.5 | 24.7 | 15.4 | 20.4 | 0.3 |
| Mask2Former 960* | 33.6 | 26.0 | 17.2 | 13.5 | 27.5 | 18.3 | 18.2 | 0.0 |
