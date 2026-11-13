# -*- coding: utf-8 -*-
"""
精確診斷三個難類別（cable/capsule/hazelnut）現有資料，找出漏用的瑕疵型態。
不修改任何資料，純統計。

用法：
  cd D:\\edge_aoi_yolo
  .venv\\Scripts\\activate
  python diagnose_hard_classes.py
"""
import os, glob
from collections import defaultdict

HARD = ["cable", "capsule", "hazelnut"]
SRC = "datasets/mvtec_yolo"
RAW = "datasets/mvtec"   # 原始 MVTec（若存在，用來比對官方完整型態）

def defect_type(fname):
    # cable_cut_inner_insulation_000.txt -> cut_inner_insulation
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    # 去掉開頭類別名、結尾編號(或 augK)
    body = parts[1:]
    while body and (body[-1].isdigit() or body[-1].startswith("aug")):
        body = body[:-1]
    return "_".join(body) if body else "?"

def scan(split):
    """回傳 {類別: {型態: 張數}} 和 {類別: 總框數}"""
    per_type = {c: defaultdict(int) for c in HARD}
    nboxes   = {c: 0 for c in HARD}
    nimgs    = {c: 0 for c in HARD}
    lbldir = os.path.join(SRC, "labels", split)
    for txt in glob.glob(os.path.join(lbldir, "*.txt")):
        name = os.path.basename(txt)
        cls = name.split("_")[0]
        if cls not in HARD:
            continue
        nimgs[cls]+=1
        per_type[cls][defect_type(txt)] += 1
        with open(txt) as f:
            nboxes[cls]+= sum(1 for l in f if l.strip())
    return per_type, nboxes, nimgs

def scan_raw_defect_types():
    """掃原始 MVTec（datasets/mvtec/<cls>/test/<defect>/）看官方有哪些型態與張數"""
    out = {c: {} for c in HARD}
    if not os.path.isdir(RAW):
        return None
    for c in HARD:
        testdir = os.path.join(RAW, c, "test")
        if not os.path.isdir(testdir): continue
        for d in sorted(os.listdir(testdir)):
            dp = os.path.join(testdir, d)
            if os.path.isdir(dp) and d != "good":
                n = len([f for f in os.listdir(dp) if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))])
                out[c][d] = n
    return out

def main():
    print("="*70)
    print("難類別資料診斷（cable / capsule / hazelnut）")
    print("="*70)

    tr_type, tr_box, tr_img = scan("train")
    va_type, va_box, va_img = scan("val")

    for c in HARD:
        print(f"\n### {c} ###")
        print(f"  現用資料集  train: {tr_img[c]} 張 / {tr_box[c]} 框　"
              f"val: {va_img[c]} 張 / {va_box[c]} 框　"
              f"合計 {tr_img[c]+va_img[c]} 張")
        # 合併型態
        types = defaultdict(lambda:[0,0])
        for t,n in tr_type[c].items(): types[t][0]+=n
        for t,n in va_type[c].items(): types[t][1]+=n
        print(f"  現用瑕疵型態（{len(types)} 種）：")
        for t in sorted(types):
            print(f"      {t:28s} train={types[t][0]:2d}  val={types[t][1]:2d}")

    # 跟原始 MVTec 比對
    raw = scan_raw_defect_types()
    if raw is None:
        print("\n[提示] 找不到原始 datasets/mvtec/ 目錄，無法自動比對官方完整型態。")
        print("       若要補資料，需從 MVTec 官方下載原始 cable/capsule/hazelnut。")
    else:
        print("\n" + "="*70)
        print("對照官方原始 MVTec（datasets/mvtec/<類別>/test/）")
        print("="*70)
        for c in HARD:
            used = set()
            for t in tr_type[c]: used.add(t)
            for t in va_type[c]: used.add(t)
            print(f"\n### {c} ### 官方 test 瑕疵型態：")
            total_official=0
            for d,n in raw[c].items():
                total_official+=n
                mark = "✓已用" if d in used else "✗未用 ★"
                print(f"      {d:28s} 官方 {n:2d} 張   {mark}")
            print(f"  官方瑕疵圖合計 {total_official} 張；你現用 {tr_img[c]+va_img[c]} 張")
            missing = total_official - (tr_img[c]+va_img[c])
            if missing>0:
                print(f"  >>> 可能可補 {missing} 張真實瑕疵圖")

if __name__=="__main__":
    main()
