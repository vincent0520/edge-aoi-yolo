# -*- coding: utf-8 -*-
# 擴增資料重訓（驗證：增加難類別資料量能否降漏檢）
# 對齊 baseline 設定，只有「擴增資料」是變因
from ultralytics import YOLO
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    model = YOLO("yolov8l.pt")
    model.train(
        data="configs/mvtec_aug.yaml",
        epochs=args.epochs,
        imgsz=640, batch=16, device=0, workers=8,
        project="runs", name="mvtec_v8l_aug",
        seed=0, deterministic=True, pretrained=True,
        optimizer="auto", patience=50, close_mosaic=10, amp=True,
        # 全用 baseline 預設，讓「擴增資料」成為唯一變因
        box=7.5, cls=0.5, dfl=1.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5,
        fliplr=0.5, flipud=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0,
    )
    print("\n訓練完成。權重：runs/detect/runs/mvtec_v8l_aug/weights/best.pt")
    print('量漏檢：python underkill_eval.py --engine "runs/detect/runs/mvtec_v8l_aug/weights/best.pt" --data configs/mvtec.yaml --imgsz 640 --batch 8 --conf 0.25 --name uk_aug')

if __name__ == "__main__":
    main()
