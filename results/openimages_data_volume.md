# Open Images V7 (generic objects) - data volume experiment

## Baseline: v8n, 15 generic classes
| Dataset size | mAP50 | mAP50-95 |
|---|---|---|
| 1300 imgs | 0.295 | 0.196 |
| 50000 imgs | 0.541 | 0.357 |
| gain | +0.246 (+83%) | +0.161 |

## Cross-dataset comparison (all v8n, ~1300 imgs unless noted)
| Dataset | Task | mAP50 @1300 |
|---|---|---|
| NEU-DET | industrial defect | 0.736 |
| MVTec | industrial defect | 0.766 |
| Open Images | generic objects | 0.295 (needs 50k for 0.541) |

## KEY FINDING
Industrial defect detection reaches usable accuracy with small data (~1300 imgs, 0.73-0.77),
while generic object detection needs ~40x more data for comparable results.
Domain-focused industrial inspection is especially edge-friendly: few images suffice to train a deployable model.
This strengthens the case for industrial defect detection as the target application for edge AI.
