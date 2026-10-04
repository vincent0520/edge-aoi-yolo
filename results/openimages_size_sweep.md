# Open Images Model Size Sweep (v8, 50k imgs)

## mAP50 by size (Open Images, 50000 imgs train)
| Model | Params(M) | mAP50 |
|---|---|---|
| n | 3.2 | 0.541 |
| s | 11.2 | 0.580 |
| m | 25.9 | 0.610 |
| l | 43.7 | 0.627 |
| x | 68.2 | 0.639 |

## Three-dataset size sweep comparison
| Model | NEU-DET | MVTec | Open Images |
|---|---|---|---|
| n | 0.736 | 0.765 | 0.541 |
| s | 0.739 | 0.771 | 0.580 |
| m | 0.723 | 0.784 | 0.610 |
| l | 0.726 | 0.792 | 0.627 |
| x | 0.744 | 0.765 | 0.639 |

## Observation (data only)
Three datasets show three different size-vs-accuracy behaviours:
- NEU-DET (1800 imgs): flat, 0.72-0.744, no systematic gain with size.
- MVTec (1258 imgs): rises to l (0.792) then x drops (0.765).
- Open Images (50000 imgs): monotonic rise n->x (0.541->0.639).
Larger models benefit accuracy only when training data is large; on small industrial-defect sets, scaling up gives no gain.
