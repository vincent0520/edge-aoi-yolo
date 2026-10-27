# diag2.py —— 桌機跑
import torch, torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

PT = r"D:\edge_aoi_yolo\runs\detect\runs\mvtec_yolov8l_baseline\weights\best.pt"
yolo = YOLO(PT); model = yolo.model
example = torch.randn(1,3,640,640)

# 測試 A：DepGraph 到底能不能建圖、追蹤到幾個可剪群組
DG = tp.DependencyGraph().build_dependency(model, example_inputs=example)
ign = [m for m in model.modules() if isinstance(m, Detect)]
groups = list(DG.get_all_groups(ignored_layers=ign))
print("可剪群組數:", len(groups))

# 測試 B：手動對「第一個卷積」剪一半，看 DepGraph 能不能真的執行
import torch.nn as nn
first_conv = None
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        first_conv = m; break
print("第一個 Conv:", first_conv.out_channels, "channels")

g = DG.get_pruning_group(first_conv, tp.prune_conv_out_channels, idxs=list(range(0, first_conv.out_channels//2)))
print("這個 group 可剪嗎:", DG.check_pruning_group(g))
b = tp.utils.count_ops_and_params(model, example)[1]
g.prune()
a = tp.utils.count_ops_and_params(model, example)[1]
print(f"手動剪第一層後: {b/1e6:.2f}M -> {a/1e6:.2f}M")