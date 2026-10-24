# diag_prune.py —— 診斷用，桌機跑
import torch, torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

PT = r"D:\edge_aoi_yolo\runs\detect\runs\mvtec_yolov8l_baseline\weights\best.pt"
yolo = YOLO(PT); model = yolo.model; model.eval()
example = torch.randn(1,3,640,640)

print("torch-pruning", tp.__version__)
# 1) 看 importance / pruner 這些類別在 1.6.0 存不存在
print("has GroupMagnitudeImportance:", hasattr(tp.importance, "GroupMagnitudeImportance"))
print("has MagnitudeImportance:", hasattr(tp.importance, "MagnitudeImportance"))
print("has GroupNormPruner:", hasattr(tp.pruner, "GroupNormPruner"))
print("has MetaPruner:", hasattr(tp.pruner, "MetaPruner"))

# 2) 看 Detect 層抓到幾個（忽略層）
ign = [m for m in model.modules() if isinstance(m, Detect)]
print("Detect layers found:", len(ign))

# 3) 實際建 pruner 跑一次，印出它回報剪了什麼
imp = tp.importance.GroupMagnitudeImportance(p=2) if hasattr(tp.importance,"GroupMagnitudeImportance") else tp.importance.MagnitudeImportance(p=2)
pruner = tp.pruner.MetaPruner(
    model, example, importance=imp,
    pruning_ratio=0.5, ignored_layers=ign, global_pruning=True,
)
b_macs,b_par = tp.utils.count_ops_and_params(model, example)
pruner.step()
a_macs,a_par = tp.utils.count_ops_and_params(model, example)
print(f"MetaPruner: {b_par/1e6:.2f}M -> {a_par/1e6:.2f}M")