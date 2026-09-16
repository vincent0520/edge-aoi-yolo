# Edge AOI - YOLO Defect Detection

Steel surface defect detection (NEU-DET) with YOLOv8, targeting NVIDIA Jetson Orin Nano deployment.

## Target Platform
- Jetson: JetPack 6.2 / CUDA 12.6 / TensorRT 10.3 / Python 3.10
- Desktop training: Windows, RTX 4070 Ti, Python 3.10, PyTorch 2.14.0+cu126

## Dataset
NEU-DET (6 classes, 1800 images). Not in repo due to size.

## Baseline (YOLOv8n, 100 epochs, FP32)
mAP50 0.736, mAP50-95 0.409

## Setup
See requirements-desktop.txt. Edit the path in configs/neu-det.yaml to your local dataset location.
