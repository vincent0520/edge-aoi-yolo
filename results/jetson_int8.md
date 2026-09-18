# Jetson Orin Nano - INT8 TensorRT Engine (NEU-DET val)

## Platform
Jetson Orin Nano 8GB, JetPack 6.2, TensorRT 10.3, INT8 engine (4.7 MB), calibrated on NEU-DET

## Full quantization ladder
| Stage | Platform | mAP50 | mAP50-95 | P | R | crazing R | inference | size |
|---|---|---|---|---|---|---|---|---|
| FP32 | Desktop | 0.736 | 0.409 | 0.645 | 0.667 | 0.278 | - | - |
| ONNX | Desktop | 0.710 | 0.375 | 0.653 | 0.645 | 0.218 | 3.5ms | 12MB |
| FP16 | Jetson | 0.710 | 0.374 | 0.657 | 0.640 | 0.215 | 10.8ms | 8.5MB |
| INT8 | Jetson | 0.635 | 0.312 | 0.536 | 0.659 | 0.316 | 9.4ms | 4.7MB |

## INT8 per-class mAP50 / P / R
crazing 0.345 / 0.438 / 0.316
inclusion 0.641 / 0.506 / 0.764
patches 0.917 / 0.751 / 0.879
pitted_surface 0.566 / 0.601 / 0.543
rolled-in_scale 0.574 / 0.533 / 0.595
scratches 0.763 / 0.384 / 0.859

## KEY FINDING (revises the original hypothesis)
The simple weak-class-first hypothesis is REVISED. INT8 does not just hurt the hardest class.
Instead INT8 shifts the precision/recall balance: overall precision drops (0.657 -> 0.536)
while many classes RECALL RISES (crazing 0.215->0.316, inclusion 0.697->0.764, rolled-in 0.494->0.595).
INT8 makes the model predict more boxes: catches more real defects (recall up) but more false alarms (precision down).

## AOI implication
INT8 trades OVERKILL (false positives) rather than ESCAPE (false negatives).
Since missed defects are usually costlier than over-rejection in AOI, INT8 may be an ACCEPTABLE compression for AOI.
This is invisible to generic mAP-only benchmarks - only visible by decomposing precision/recall from an AOI viewpoint.

## Size and speed
INT8 smallest (4.7MB, ~1/2 of FP16) and fastest (9.4ms inference, ~106 FPS).
