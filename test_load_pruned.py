# test_load_pruned.py —— 測試在主環境載入剪枝模型（C2f_v2）
import sys, types
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, Bottleneck

# ---- 把 C2f_v2 的定義搬過來（從 scripts/yolov8_pruning.py 複製）----
class C2f_v2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# ---- 關鍵：把 C2f_v2 註冊到 pickle 找得到的地方 ----
# pickle 存模型時是在 __main__ 下，所以註冊到 __main__
import __main__
__main__.C2f_v2 = C2f_v2

# 也註冊到 ultralytics.nn.modules（保險）
import ultralytics.nn.modules as M
M.C2f_v2 = C2f_v2

# ---- 試載入 ----
from ultralytics import YOLO
PT = r"runs\detect\step_7_finetune\weights\best.pt"
m = YOLO(PT)
print("✅ 載入成功！參數:", round(sum(p.numel() for p in m.model.parameters())/1e6, 2), "M")