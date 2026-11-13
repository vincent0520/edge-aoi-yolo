# -*- coding: utf-8 -*-
"""
方向 C 第二版：Cost-sensitive 重訓 v2（MVTec v8l）
根據 v1 結果調整：
  v1 問題：flipud=0.5 傷到 transistor（方向敏感元件），整體漏檢反升
  v2 修正：
    - flipud 0.5 → 0     關掉上下翻（救回 transistor）
    - cls     1.0 → 1.5   加重分類懲罰（強化 cost-sensitive）
    - degrees 10 → 5      旋轉改小（避免再傷方向敏感類別）
    - copy_paste 0.3 保留（複製小瑕疵，對 cable/hazelnut 有效）
    - mixup   0.1 → 0.05  略降

用法：
  cd D:\\edge_aoi_yolo
  .venv\\Scripts\\activate
  python train_cost_sensitive_v2.py            # 50 epochs
  python train_cost_sensitive_v2.py --epochs 100
"""
import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--model", default="yolov8l.pt")
    ap.add_argument("--data", default="configs/mvtec.yaml")
    ap.add_argument("--name", default="mvtec_v8l_costsens_v2")
    ap.add_argument("--cls", type=float, default=1.5, help="分類 loss 權重（v1=1.0, baseline=0.5）")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640, batch=16, device=0, workers=8,
        project="runs", name=args.name,
        seed=0, deterministic=True, pretrained=True,
        optimizer="auto", patience=50, close_mosaic=10, amp=True,
        # ---- cost-sensitive：加重分類懲罰 ----
        box=7.5, cls=args.cls, dfl=1.5,
        # ---- 增強（修正版：關 flipud、降 degrees）----
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0,       # v1=10 → 5（降低，避免傷方向敏感類別）
        translate=0.1, scale=0.5,
        fliplr=0.5,
        flipud=0.0,        # v1=0.5 → 0（關掉，救回 transistor）
        mosaic=1.0,
        mixup=0.05,        # v1=0.1 → 0.05
        copy_paste=0.3,    # 保留（對小瑕疵有效）
    )
    print("\n訓練完成。權重：runs/detect/runs/%s/weights/best.pt" % args.name)
    print("下一步：python underkill_eval.py --engine runs/detect/runs/%s/weights/best.pt "
          "--data configs/mvtec.yaml --imgsz 640 --batch 8 --conf 0.25 --name uk_v2" % args.name)

if __name__ == "__main__":
    main()
