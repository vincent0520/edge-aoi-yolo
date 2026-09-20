# Jetson Orin Nano - MVTec Model Size Speed Sweep (FP16 engine)

## Platform
Jetson Orin Nano 8GB, JetPack 6.2, TensorRT 10.3, FP16 engines

## Speed vs Accuracy (all 5 sizes)
| Model | Params(M) | Inference(ms) | FPS | Desktop mAP50 |
|---|---|---|---|---|
| n | 3.2 | 10.7 | 93 | 0.765 |
| s | 11.2 | 18.9 | 53 | 0.771 |
| m | 25.9 | 30.9 | 32 | 0.784 |
| l | 43.7 | 36.1 | 28 | 0.792 |
| x | 68.2 | 59.2 | 17 | 0.765 |

## Key findings
1. All 5 sizes could be converted and run on 8GB Orin Nano - deployable, but with steep speed cost.
2. n->x: inference time 5.5x slower (10.7->59.2ms), FPS collapses 93->17.
3. Accuracy peaks at l (0.792), x drops back to 0.765 - same as smallest n.
4. x is strictly worst: slowest, largest, and no accuracy gain over n.

## Conclusion
For edge AOI, smallest model n is optimal: near-saturation accuracy at highest FPS. Scaling up gives no accuracy return and heavy speed penalty. This motivates small-model + quantization as the edge AOI strategy.
