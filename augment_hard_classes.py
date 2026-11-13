# -*- coding: utf-8 -*-
"""
難類別 copy-paste 擴增（cable / capsule / hazelnut）
目的：驗證「增加難類別資料能否降低漏檢」

做法：
  1. 統計三類在 train 的現有數量（診斷）
  2. 對每張難類別訓練圖，用 copy-paste 生成 N 張變體：
     - 把該圖的瑕疵框區域裁下來
     - 貼回同一張圖的隨機位置（模擬瑕疵出現在不同處）
     - 加入位置抖動、亮度/對比微調、小角度旋轉
     - 同步生成新標註
  3. 只擴增 train，val 完全不動（避免評估作弊）
  4. 輸出到 datasets/mvtec_aug/（原始資料不動）+ 新 yaml

用法：
  cd D:\\edge_aoi_yolo
  .venv\\Scripts\\activate
  python augment_hard_classes.py                 # 預設每張生成 4 個變體
  python augment_hard_classes.py --per 6         # 每張 6 個
"""
import argparse, os, shutil, random, glob
import numpy as np
import cv2

HARD = ["cable", "capsule", "hazelnut"]
random.seed(0); np.random.seed(0)

def yolo_to_xyxy(box, W, H):
    _, xc, yc, w, h = box
    x1 = (xc - w/2)*W; y1 = (yc - h/2)*H
    x2 = (xc + w/2)*W; y2 = (yc + h/2)*H
    return int(max(0,x1)), int(max(0,y1)), int(min(W,x2)), int(min(H,y2))

def xyxy_to_yolo(cls, x1,y1,x2,y2, W,H):
    xc = (x1+x2)/2/W; yc=(y1+y2)/2/H
    w=(x2-x1)/W; h=(y2-y1)/H
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/mvtec_yolo")
    ap.add_argument("--dst", default="datasets/mvtec_aug")
    ap.add_argument("--per", type=int, default=4, help="每張難類別圖生成幾個變體")
    args=ap.parse_args()

    src_img=os.path.join(args.src,"images","train")
    src_lbl=os.path.join(args.src,"labels","train")
    dst_img=os.path.join(args.dst,"images","train")
    dst_lbl=os.path.join(args.dst,"labels","train")
    os.makedirs(dst_img,exist_ok=True); os.makedirs(dst_lbl,exist_ok=True)

    # 1. 先複製整個原始 train（原資料全保留）
    print("複製原始 train ...")
    for f in glob.glob(os.path.join(src_img,"*")):
        shutil.copy(f, dst_img)
    for f in glob.glob(os.path.join(src_lbl,"*.txt")):
        shutil.copy(f, dst_lbl)
    # 複製 val（擴增版 yaml 的 val 指向同一份，不動）
    for sub in ["val"]:
        os.makedirs(os.path.join(args.dst,"images",sub),exist_ok=True)
        os.makedirs(os.path.join(args.dst,"labels",sub),exist_ok=True)
        for f in glob.glob(os.path.join(args.src,"images",sub,"*")):
            shutil.copy(f, os.path.join(args.dst,"images",sub))
        for f in glob.glob(os.path.join(args.src,"labels",sub,"*.txt")):
            shutil.copy(f, os.path.join(args.dst,"labels",sub))

    # 2. 統計 + 擴增
    stats={c:0 for c in HARD}; made={c:0 for c in HARD}
    train_imgs=glob.glob(os.path.join(src_img,"*"))
    for imgpath in train_imgs:
        base=os.path.splitext(os.path.basename(imgpath))[0]
        cls_name=base.split("_")[0]
        if cls_name not in HARD:
            # 有些類別名含底線（metal_nut），這裡 HARD 都是單字，安全
            continue
        stats[cls_name]+=1
        lblpath=os.path.join(src_lbl, base+".txt")
        if not os.path.exists(lblpath): continue
        img=cv2.imread(imgpath)
        if img is None: continue
        H,W=img.shape[:2]
        boxes=[]
        for line in open(lblpath):
            p=line.split()
            if len(p)==5: boxes.append([int(p[0])]+[float(x) for x in p[1:]])
        if not boxes: continue

        for k in range(args.per):
            canvas=img.copy()
            new_lines=[l.strip() for l in open(lblpath) if l.strip()]
            for box in boxes:
                x1,y1,x2,y2=yolo_to_xyxy(box,W,H)
                if x2-x1<5 or y2-y1<5: continue
                patch=img[y1:y2, x1:x2].copy()
                # 亮度/對比微調
                alpha=random.uniform(0.85,1.15); beta=random.uniform(-15,15)
                patch=cv2.convertScaleAbs(patch,alpha=alpha,beta=beta)
                ph,pw=patch.shape[:2]
                # 貼到隨機位置（避免貼出界）
                nx=random.randint(0,max(0,W-pw)); ny=random.randint(0,max(0,H-ph))
                # 用羽化邊緣貼上（簡單 alpha 混合，避免硬邊）
                roi=canvas[ny:ny+ph, nx:nx+pw]
                if roi.shape[:2]!=patch.shape[:2]: continue
                blended=cv2.addWeighted(roi,0.1,patch,0.9,0)
                canvas[ny:ny+ph, nx:nx+pw]=blended
                new_lines.append(xyxy_to_yolo(box[0], nx,ny,nx+pw,ny+ph, W,H))
            outbase=f"{base}_aug{k}"
            cv2.imwrite(os.path.join(dst_img,outbase+".jpg"), canvas)
            with open(os.path.join(dst_lbl,outbase+".txt"),"w") as f:
                f.write("\n".join(new_lines)+"\n")
            made[cls_name]+=1

    # 3. 產生新 yaml（複製原 yaml 改 path）
    print("\n===== 診斷：難類別原始 train 張數 =====")
    for c in HARD: print(f"  {c:10s}: {stats[c]} 張 → 擴增生成 {made[c]} 張（共 {stats[c]+made[c]}）")

    print("\n擴增完成。新資料集：%s" % args.dst)
    print("請建立 configs/mvtec_aug.yaml（把 path 指向 datasets/mvtec_aug，其餘同 mvtec.yaml）")
    print("注意：val 用原始資料，評估才公平。")

if __name__=="__main__":
    main()
