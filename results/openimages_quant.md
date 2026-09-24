# Open Images v8n Quantization (Jetson)

## FP16 vs INT8 (overall)
| Metric | FP16 | INT8 | delta |
|---|---|---|---|
| mAP50 | 0.540 | 0.513 | -0.027 |
| Precision | 0.594 | 0.557 | -0.037 |
| Recall | 0.511 | 0.522 | +0.011 |

## Three-dataset comparison (INT8 vs FP16 direction)
| Dataset | Task | Precision | Recall |
|---|---|---|---|
| NEU-DET | industrial defect | down (-0.117) | up (+0.014) |
| MVTec | industrial defect | up (+0.006) | down (-0.045) |
| Open Images | generic objects | down (-0.037) | up (+0.011) |

## KEY FINDING
INT8 precision/recall shift direction is INCONSISTENT across three datasets and
is NOT explained by task type: the two industrial datasets (NEU-DET, MVTec) move in
OPPOSITE directions, while Open Images (generic) matches NEU-DET, not the other industrial one.
This indicates the effect is dataset-specific and must be measured per-dataset, not predicted from task type.
