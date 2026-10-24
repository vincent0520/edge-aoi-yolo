# Jetson Power/Memory across 3 datasets x 5 sizes (15W, FP16, GPU-load)

## Power (mW, VDD_IN avg under GPU load)
| Size | MVTec | NEU-DET | Open Images |
|---|---|---|---|
| n | 5688 | - | - |
| s | 6199 | 6153 | 6227 |
| m | 7291 | 7337 | 7415 |
| l | 8649 | 8651 | 8555 |
| x | 9524 | 9523 | 9582 |

## Peak RAM (MB)
| Size | NEU-DET | Open Images |
|---|---|---|
| s | 2670 | 2681 |
| m | 2678 | 2698 |
| l | 2672 | 2675 |
| x | 2699 | 2681 |

## KEY FINDING
Edge inference power is determined almost entirely by model size, NOT by dataset/task:
for each size, power across the three datasets differs by <2% (within measurement noise).
This means deployment cost (power/memory/speed) is an intrinsic property of the model, measurable once and reusable across tasks; only accuracy is dataset-dependent and must be measured per-dataset.
