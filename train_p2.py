# -*- coding: utf-8 -*-
"""
方向 D：加 P2 小目標偵測層（MVTec v8l-p2）
診斷依據：v2 混淆矩陣顯示 cable(18)/hazelnut(19) 漏檢幾乎全掉進 background
         → 是「小目標漏偵測」，非「誤分類」→ 對症解法是 P2 層（非 SlideLoss）

模型：yolov8l-p2.yaml（多一層 P2/4，專抓小目標）+ 載入 yolov8l.pt 預訓練權重
增強：保持接近 baseline（不加會誤傷 transistor 的 flipud/大旋轉），
      只留 copy_paste 幫小瑕疵；讓「P2 層本身」成為主要變因，對照乾淨。

用法：
  cd D:\\edge_aoi_yolo
  .venv\\Scripts\\activate
  python train_p2.py              # 50 epochs
  python train_p2.py --epochs 100
"""
import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--cfg", default="yolov8l-p2.yaml", help="P2 結構")
    ap.add_argument("--weights", default="yolov8l.pt", help="預訓練權重")
    ap.add_argument("--data", default="configs/mvtec.yaml")
    ap.add_argument("--name", default="mvtec_v8l_p2")
    args = ap.parse_args()

    # 用 P2 結構建模型，再載入 l 的預訓練權重（結構不同的層會自動略過）
    model = YOLO(args.cfg).load(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640, batch=16, device=0, workers=8,
        project="runs", name=args.name,
        seed=0, deterministic=True, pretrained=True,
        optimizer="auto", patience=50, close_mosaic=10, amp=True,
        # loss 權重用 baseline 預設（讓 P2 結構成為唯一主變因）
        box=7.5, cls=0.5, dfl=1.5,
        # 增強保持接近 baseline，只加 copy_paste 幫小瑕疵；不加 flipud/大旋轉
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0, mixup=0.0,
        copy_paste=0.2,
    )
    print("\n訓練完成。權重：runs/detect/runs/%s/weights/best.pt" % args.name)
    print("下一步量漏檢：")
    print('  python underkill_eval.py --engine "runs/detect/runs/%s/weights/best.pt" '
          '--data configs/mvtec.yaml --imgsz 640 --batch 8 --conf 0.25 --name uk_p2' % args.name)

if __name__ == "__main__":
    main()
