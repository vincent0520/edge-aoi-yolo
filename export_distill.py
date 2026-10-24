import sys
sys.path.insert(0, 'scripts')
import yolov8_pruning
import __main__
__main__.C2f_v2 = yolov8_pruning.C2f_v2
from ultralytics import YOLO

m = YOLO(r"runs\detect\mvtec_l_prune_distill_v2\weights\best.pt")
m.export(format="onnx", imgsz=640, opset=18, simplify=True)
print("ONNX 匯出完成")