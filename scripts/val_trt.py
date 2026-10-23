# val_trt.py —— 用 onnxruntime TensorRT FP16 量剪枝模型精度（跟速度同基準）
import os
os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"
os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
os.environ["ORT_TENSORRT_CACHE_PATH"] = "/home/jetson/edge-aoi-yolo/pruned/trt_cache"

from ultralytics import YOLO

# 用 ONNX 建立 YOLO 物件，並強制用 TensorRT provider
model = YOLO("pruned/mvtec_l_pruned.onnx", task="detect")

# 關鍵：把 onnxruntime session 的 provider 指定成 TensorRT FP16
# ultralytics 的 ONNX 後端存在 model.predictor / model.model，覆寫它的 providers
import onnxruntime as ort
print("可用 providers:", ort.get_available_providers())

# 直接用 ultralytics val，但先確認它建 session 時用 TRT
# 若 ultralytics 不讓指定，改用下面的手動驗證
metrics = model.val(
    data="configs/mvtec.yaml",
    imgsz=640,
    batch=1,
    device=0,
)
print("mAP50 =", metrics.box.map50)
print("mAP50-95 =", metrics.box.map)
