# diag4.py —— 繞過 DetectionModel 外殼，直接對內部 Sequential 建圖
import torch, torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
import torch.nn as nn

PT = r"D:\edge_aoi_yolo\runs\detect\runs\mvtec_yolov8l_baseline\weights\best.pt"
yolo = YOLO(PT)
model = yolo.model          # DetectionModel
model.eval()

# 看內部結構：model.model 是 nn.Sequential，裡面才是真正的層
inner = model.model
print("inner type:", type(inner).__name__, "len:", len(inner))
print("前 3 層:", [type(inner[i]).__name__ for i in range(3)])

example = torch.randn(1,3,640,640)

# 關鍵：對「整個 model」建圖，但用 unwrapped_parameters 處理 YOLO 特殊層
# 並明確告訴 DepGraph 從哪些 Conv 開始追
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=example)

# 列出圖裡實際有幾個可剪的 module（診斷圖到底建了多少）
prunable = []
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        try:
            g = DG.get_pruning_group(m, tp.prune_conv_out_channels,
                                     idxs=[0])
            if DG.check_pruning_group(g):
                prunable.append(m)
        except Exception:
            pass
print("圖中可剪的 Conv/Linear 數:", len(prunable), "/ 總 Conv 數:",
      sum(1 for m in model.modules() if isinstance(m, nn.Conv2d)))