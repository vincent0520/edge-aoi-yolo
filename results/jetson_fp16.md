# Jetson Orin Nano - FP16 TensorRT Engine (NEU-DET val)

## Platform
Jetson Orin Nano 8GB, JetPack 6.2, TensorRT 10.3, FP16 engine (8.5 MB)

## Three-way comparison
| Stage | Platform | mAP50 | mAP50-95 | crazing recall | inference |
|---|---|---|---|---|---|
| FP32 | Desktop 4070Ti | 0.736 | 0.409 | 0.278 | - |
| ONNX | Desktop 4070Ti | 0.710 | 0.375 | 0.218 | 3.5ms |
| FP16 | Jetson Orin Nano | 0.710 | 0.374 | 0.215 | 10.8ms |

## Per-class mAP50 / recall (FP16 on Jetson)
patches 0.923 / 0.887
scratches 0.825 / 0.844
inclusion 0.771 / 0.697
pitted_surface 0.784 / 0.704
rolled-in_scale 0.610 / 0.494
crazing 0.346 / 0.215

## Speed (per image)
preprocess 2.7ms, inference 10.8ms (~92 FPS), postprocess 4.6ms

## Observations
1. FP16 is near-lossless: mAP50 matches desktop ONNX (0.710), only slightly below FP32 (0.736).
2. Weak-class-first effect holds: crazing stays worst (recall 0.215), patches unaffected (0.887).
3. Real edge speed (10.8ms inference) is the key edge-only datapoint desktop cannot provide.
