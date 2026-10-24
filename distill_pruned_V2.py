import sys, types
sys.path.insert(0, 'scripts')
import torch
import torch.nn as nn
import torch.nn.functional as F

import yolov8_pruning
import __main__
__main__.C2f_v2 = yolov8_pruning.C2f_v2

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

STUDENT = r"runs\detect\step_7_finetune\weights\best.pt"
TEACHER = r"runs\detect\runs\mvtec_yolov8l_baseline\weights\best.pt"
DATA    = r"D:/edge_aoi_yolo/configs/mvtec.yaml"
EPOCHS  = 50
DISTILL_ALPHA = 1.0

_TEACHER = {"model": None}
_ORIG_LOSS = DetectionModel.loss   # 原始類別方法


def loss_with_distill(self, batch, preds=None):
    dev = next(self.parameters()).device
    if getattr(self, 'criterion', None) is not None and hasattr(self.criterion, 'proj'):
        self.criterion.proj = self.criterion.proj.to(dev)
    total, items = _ORIG_LOSS(self, batch, preds)
    tm = _TEACHER["model"]
    if tm is not None and self.training:   # 只在訓練時蒸餾
        try:
            imgs = batch["img"].to(dev).float()
            imgs = imgs / 255.0 if imgs.max() > 1.5 else imgs
            with torch.no_grad():
                t_out = tm(imgs)
            s_out = self(imgs)
            def flat(o):
                if isinstance(o, (list, tuple)):
                    return torch.cat([x.flatten().float() for x in o if torch.is_tensor(x)])
                return o.flatten().float()
            sf, tf = flat(s_out), flat(t_out)
            n = min(sf.numel(), tf.numel())
            dloss = F.mse_loss(sf[:n], tf[:n].to(sf.device)) * DISTILL_ALPHA
            total = total + dloss * imgs.shape[0]
        except Exception as e:
            print("[distill 跳過]", e)
    return total, items


def main():
    student = YOLO(STUDENT)
    student.__setattr__("train_v2", yolov8_pruning.train_v2.__get__(student))
    sm = student.model
    print("學生:", round(sum(p.numel() for p in sm.parameters()) / 1e6, 2), "M")

    teacher = YOLO(TEACHER)
    _TEACHER["model"] = teacher.model.cuda().float().eval()
    for p in _TEACHER["model"].parameters():
        p.requires_grad = False
    print("老師:", round(sum(p.numel() for p in _TEACHER["model"].parameters()) / 1e6, 2), "M")

    # 覆寫整個類別的 loss（不綁實例，pickle 不受影響）
    DetectionModel.loss = loss_with_distill

    sm.cuda()
    for p in sm.parameters():
        p.requires_grad = True

    student.train_v2(
        pruning=True,
        data=DATA, epochs=EPOCHS, imgsz=640, batch=16, seed=0,
        name="mvtec_l_prune_distill_v2", workers=0, exist_ok=True, device=0,
    )
    # 還原類別方法
    DetectionModel.loss = _ORIG_LOSS
    print("蒸餾完成")


if __name__ == '__main__':
    main()