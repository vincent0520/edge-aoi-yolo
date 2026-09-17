# FP32 PyTorch vs ONNX (NEU-DET val, imgsz 640)

## Overall
| Metric | FP32 (PyTorch) | ONNX | Delta |
|---|---|---|---|
| mAP50 | 0.736 | 0.710 | -0.026 |
| mAP50-95 | 0.409 | 0.375 | -0.034 |

## Per-class recall (key observation)
| Class | FP32 | ONNX | Delta |
|---|---|---|---|
| patches (easiest) | 0.860 | 0.889 | +0.029 |
| crazing (hardest) | 0.278 | 0.218 | -0.060 |

## Observation
Even lossless-ish ONNX export degrades the hardest class (crazing) most, while the easiest class (patches) is unaffected. Precision loss hits weak classes disproportionately - the core hypothesis for the compression chapter. Expect this effect to amplify with FP16 and especially INT8 on Jetson.
