# -*- coding: utf-8 -*-
"""
漏檢率 / 誤檢率量測（Underkill / Overkill）
在 Jetson 上執行，吃現成的 .engine，跑 yolo val 取得混淆矩陣，
依官方定義 matrix[predicted, true]、index nc = background 計算：
  FN (漏檢/Underkill) = matrix[nc, gc]  真實瑕疵被判成背景（沒抓到）
  FP (誤檢/Overkill)  = matrix[dc, nc]  背景被判成瑕疵（抓錯）
  TP                  = matrix[c, c]

用法：
  python underkill_eval.py --engine best.engine --data mvtec.yaml --imgsz 640
"""
import argparse, os, sys, csv
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="TensorRT .engine 路徑（也可放 .pt 比對）")
    ap.add_argument("--data",   required=True, help="資料集 yaml")
    ap.add_argument("--imgsz",  type=int, default=640)
    ap.add_argument("--batch",  type=int, default=1, help="INT8/靜態 engine 請用 1")
    ap.add_argument("--conf",   type=float, default=0.25, help="信心閾值（產線一般 0.25）")
    ap.add_argument("--iou",    type=float, default=0.7)
    ap.add_argument("--name",   default="underkill", help="輸出資料夾名")
    ap.add_argument("--out",    default="underkill_report.csv")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.engine, task="detect")

    # plots=True 會同時存 confusion_matrix.png / *_normalized.png 方便放報告
    r = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch,
                  conf=args.conf, iou=args.iou, plots=True, name=args.name, verbose=False)

    cm = r.confusion_matrix.matrix          # shape (nc+1, nc+1), [predicted, true]
    cm = np.array(cm, dtype=np.float64)
    nc = cm.shape[0] - 1                     # 最後一軸是 background
    names = r.names if hasattr(r, "names") and r.names else model.names
    names = [names[i] for i in range(nc)]

    rows = []
    tot_tp = tot_fn = tot_fp = 0
    for c in range(nc):
        tp = cm[c, c]                        # 對角線：抓對
        fn = cm[nc, c]                        # 最後一列：真實=c 卻判成背景 → 漏檢
        fp = cm[c, nc]                        # 最後一欄：真實=背景 卻判成 c → 誤檢
        gt = cm[:, c].sum()                   # 該類真實總數（整欄）
        underkill = fn / gt if gt > 0 else 0.0           # 漏檢率
        recall    = tp / gt if gt > 0 else 0.0           # 召回（= 1 - 漏檢率 的概念）
        pred_c    = cm[c, :].sum()                        # 被判成 c 的總數（整列）
        precision = tp / pred_c if pred_c > 0 else 0.0
        rows.append([names[c], int(gt), int(tp), int(fn), int(fp),
                     round(underkill, 4), round(recall, 4), round(precision, 4)])
        tot_tp += tp; tot_fn += fn; tot_fp += fp

    tot_gt = tot_tp + tot_fn
    overall_underkill = tot_fn / tot_gt if tot_gt > 0 else 0.0
    overall_recall    = tot_tp / tot_gt if tot_gt > 0 else 0.0

    # 印出
    hdr = ["類別", "真實數GT", "抓對TP", "漏檢FN", "誤檢FP", "漏檢率", "召回率", "精確率"]
    w = [14, 9, 8, 8, 8, 8, 8, 8]
    def line(cells):
        return "  ".join(str(x).ljust(w[i]) for i, x in enumerate(cells))
    print("\n" + "=" * 90)
    print(f"漏檢/誤檢分析  engine={os.path.basename(args.engine)}  conf={args.conf}  iou={args.iou}")
    print("=" * 90)
    print(line(hdr))
    print("-" * 90)
    for row in rows:
        print(line(row))
    print("-" * 90)
    print(line(["【整體】", int(tot_gt), int(tot_tp), int(tot_fn), int(tot_fp),
                round(overall_underkill, 4), round(overall_recall, 4), ""]))
    print("=" * 90)
    print(f"整體漏檢率 (Underkill) = {overall_underkill:.2%}   "
          f"整體誤檢數 (Overkill FP) = {int(tot_fp)}")
    print("混淆矩陣圖已存於 runs/detect/%s/confusion_matrix.png" % args.name)

    # 寫 CSV
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(hdr)
        wr.writerows(rows)
        wr.writerow(["整體", int(tot_gt), int(tot_tp), int(tot_fn), int(tot_fp),
                     round(overall_underkill, 4), round(overall_recall, 4), ""])
    print("CSV 已存：%s" % args.out)

if __name__ == "__main__":
    main()
