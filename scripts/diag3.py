# diag3.py —— 桌機跑，測試修正建圖
import torch, torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
import torch.nn as nn

PT = r"D:\edge_aoi_yolo\runs\detect\runs\mvtec_yolov8l_baseline\weights\best.pt"
yolo = YOLO(PT); model = yolo.model
model.eval()

# 關鍵修正：關掉 YOLO 的特殊 forward 分支
for m in model.modules():
    if isinstance(m, Detect):
        m.export = True       # export 模式 → 走乾淨 forward，不含後處理
        m.format = "onnx"

example = torch.randn(1,3,640,640)

# 重新建圖
DG = tp.DependencyGraph().build_dependency(model, example_inputs=example)
ign = [m for m in model.modules() if isinstance(m, Detect)]
groups = list(DG.get_all_groups(ignored_layers=ign))
print("修正後可剪群組數:", len(groups))

# 試自動剪
imp = tp.importance.GroupMagnitudeImportance(p=2)
pruner = tp.pruner.GroupNormPruner(
    model, example, importance=imp,
    pruning_ratio=0.5, ignored_layers=ign, global_pruning=True,
)
b = tp.utils.count_ops_and_params(model, example)[1]
pruner.step()
a = tp.utils.count_ops_and_params(model, example)[1]
print(f"剪枝: {b/1e6:.2f}M -> {a/1e6:.2f}M  (-{(1-a/b)*100:.1f}%)")