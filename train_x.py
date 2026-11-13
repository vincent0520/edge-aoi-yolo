# -*- coding: utf-8 -*-
# 方向 4：換更大模型 YOLOv8x 重訓（測試模型容量對漏檢的影響）
# 用 baseline 乾淨設定，只換 l -> x，用原始資料（非擴增），讓「模型大小」成為唯一變因
from ultralytics import YOLO
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    model = YOLO("yolov8x.pt")
    model.train(
        data="configs/mvtec.yaml",          # 原始資料（跟 baseline 同條件）
        epochs=args.epochs,
        imgsz=640, batch=16, device=0, workers=0,   # workers=0 避免 Windows 多進程問題
        project="runs", name="mvtec_v8x_baseline",
        seed=0, deterministic=True, pretrained=True,
        optimizer="auto", patience=50, close_mosaic=10, amp=True,
        # 全用 baseline 預設（讓模型大小成為唯一變因）
        box=7.5, cls=0.5, dfl=1.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5,
        fliplr=0.5, flipud=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0,
    )
    print("\n訓練完成。權重：runs/detect/runs/mvtec_v8x_baseline/weights/best.pt")
    print('量漏檢（記得確認實際資料夾名，可能有 -N 後綴）：')
    print('  python underkill_eval.py --engine "runs/detect/runs/mvtec_v8x_baseline/weights/best.pt" --data configs/mvtec.yaml --imgsz 640 --batch 8 --conf 0.25 --name uk_x')

if __name__ == "__main__":
    main()
