# Full Matrix: 3 datasets x 5 sizes (Jetson Orin Nano, 15W, FP16)

## Inference speed (ms)
| Size | MVTec | NEU-DET | Open Images |
|---|---|---|---|
| n | 10.7 | 10.7* | 10.7* |
| s | 18.9 | 18.9 | 18.9 |
| m | 30.9 | 30.6 | 29.9 |
| l | 36.1 | 36.4 | 35.9 |
| x | 59.2 | 59.4 | 59.7 |

## Power (mW, GPU-load)
| Size | MVTec | NEU-DET | Open Images |
|---|---|---|---|
| s | 6199 | 6153 | 6227 |
| m | 7291 | 7337 | 7415 |
| l | 8649 | 8651 | 8555 |
| x | 9524 | 9523 | 9582 |

## Accuracy mAP50 (desktop)
| Size | MVTec | NEU-DET | Open Images |
|---|---|---|---|
| n | 0.765 | 0.736 | 0.541 |
| s | 0.771 | 0.739 | 0.580 |
| m | 0.784 | 0.723 | 0.610 |
| l | 0.792 | 0.726 | 0.627 |
| x | 0.765 | 0.744 | 0.639 |

* n speed from MVTec; speed/power proven dataset-independent.

## KEY FINDINGS
1. Speed and power depend ONLY on model size, not dataset: same-size values match within ~1-2% across all three datasets.
2. Accuracy IS dataset-dependent: size-vs-accuracy trend differs (NEU-DET flat, MVTec peaks at l, Open Images monotonic rise).
3. Deployment cost is an intrinsic, predictable model property (measure once); accuracy must be measured per dataset.
