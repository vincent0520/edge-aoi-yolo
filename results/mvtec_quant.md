# MVTec v8n Quantization (Jetson Orin Nano)

## FP16 vs INT8 (overall)
| Metric | FP16 | INT8 | delta |
|---|---|---|---|
| mAP50 | 0.771 | 0.753 | -0.018 |
| mAP50-95 | 0.464 | 0.452 | -0.012 |
| Precision | 0.784 | 0.790 | +0.006 |
| Recall | 0.728 | 0.683 | -0.045 |

## cable (weakest class)
| Metric | FP16 | INT8 | delta |
|---|---|---|---|
| Precision | 0.774 | 0.895 | +0.121 |
| Recall | 0.474 | 0.447 | -0.027 |

## KEY FINDING (revises the cross-dataset hypothesis)
The precision/recall shift from INT8 is DATASET-DEPENDENT, not a universal direction.
- NEU-DET INT8: recall UP, precision DOWN (model predicts more, more aggressive).
- MVTec INT8: recall DOWN, precision UP (model predicts fewer, more conservative).
The two datasets move in OPPOSITE directions.

## AOI implication
INT8 cannot be assumed beneficial for AOI. On NEU-DET it reduces misses (recall up, good);
on MVTec it increases misses (recall down, bad for AOI which fears false negatives).
Whether INT8 is acceptable for AOI must be validated per-dataset, not assumed.
