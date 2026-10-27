# distill_pruned.py —— 剪枝 v8l 學生 + v8l 老師 蒸餾（Windows 安全版）
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, Bottleneck

# ---- C2f_v2 定義（放在最外層，子進程 import 時也認得）----
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

# 註冊（最外層執行，子進程也會跑到）
import __main__
__main__.C2f_v2 = C2f_v2
import ultralytics.nn.modules as M
M.C2f_v2 = C2f_v2


def main():
    from ultralytics import YOLO
    STUDENT = r"runs\detect\step_7_finetune\weights\best.pt"
    TEACHER = r"runs\detect\runs\mvtec_yolov8l_baseline\weights\best.pt"

    model = YOLO(STUDENT)
    print("學生參數:", round(sum(p.numel() for p in model.model.parameters())/1e6, 2), "M")

    model.train(
        data="configs/mvtec.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        seed=0,
        name="mvtec_l_prune_distill",
        distill_model=TEACHER,
        pretrained=False,
        exist_ok=True,
        workers=0,          # Windows + 自訂類別，workers=0 最穩
    )
    print("蒸餾完成")


if __name__ == '__main__':
    main()