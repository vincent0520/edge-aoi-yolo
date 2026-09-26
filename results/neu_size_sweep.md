# NEU-DET Model Size Sweep (v8, 100 epochs, seed=0)

## mAP50 by model size
| Model | Params(M) | NEU-DET mAP50 | MVTec mAP50 (ref) |
|---|---|---|---|
| n | 3.2 | 0.736 | 0.765 |
| s | 11.2 | 0.739 | 0.771 |
| m | 25.9 | 0.723 | 0.784 |
| l | 43.7 | 0.726 | 0.792 |
| x | 68.2 | 0.744 | 0.765 |

## Observation (data only)
NEU-DET: mAP50 stays within 0.72-0.744 across all five sizes, no systematic increase with size; largest (x) is highest but by a small margin.
MVTec: mAP50 rises n->l (0.765->0.792) then x drops to 0.765.
The size-vs-accuracy trend differs between the two datasets.

## Cross-experiment note
This mirrors the quantization finding: both model-size effect and INT8 precision/recall shift are inconsistent across datasets, i.e. dataset-specific rather than predictable from task type.
