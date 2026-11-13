# -*- coding: utf-8 -*-
"""
方向 C：Cost-sensitive 重訓（MVTec v8l）
目標：降低難類別（cable/capsule/hazelnut）漏檢，不改模型結構（FPS 不變）

相對 baseline 的改動（其餘全對齊原始 args.yaml）：
  1. cls 權重 0.5 → 1.0       # 加重分類懲罰，模型更重視「抓到」
  2. 資料增強針對難類別打開：
       degrees   0 → 10       # 旋轉（cable 細長物件）
       flipud    0 → 0.5      # 上下翻
       mixup     0 → 0.1      # 混合
       copy_paste 0 → 0.3     # 小瑕疵複製貼上（增加難類別樣本）

用法（先啟動 .venv）：
  cd D:\\edge_aoi_yolo
  .venv\\Scripts\\activate
  python train_cost_sensitive.py          # 預設 50 epochs 快速驗證
  python train_cost_sensitive.py --epochs 100   # 跑滿 100
"""
import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50, help="先 50 驗證，有效再 100")
    ap.add_argument("--model", default="yolov8l.pt")
    ap.add_argument("--data", default="configs/mvtec.yaml")
    ap.add_argument("--name", default="mvtec_v8l_costsens")
    ap.add_argument("--cls", type=float, default=1.0, help="分類 loss 權重（baseline=0.5）")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        project="runs",
        name=args.name,
        seed=0,
        deterministic=True,
        pretrained=True,
        optimizer="auto",
        patience=50,
        close_mosaic=10,
        amp=True,
        # ---- cost-sensitive 核心：加重分類懲罰 ----
        box=7.5,
        cls=args.cls,     # 0.5 → 1.0
        dfl=1.5,
        # ---- 針對難類別的資料增強 ----
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0,     # baseline 0
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,       # baseline 0
        mosaic=1.0,
        mixup=0.1,        # baseline 0
        copy_paste=0.3,   # baseline 0
    )
    print("\n訓練完成。權重在：runs/%s/weights/best.pt" % args.name)
    print("下一步：export engine → 跑 underkill_eval 看 cable 漏檢是否下降。")

if __name__ == "__main__":
    main()
