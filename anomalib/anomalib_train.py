# -*- coding: utf-8 -*-
"""
EfficientAD 異常偵測訓練（MVTec）
逐類訓練，存每類的 Image AUROC（與 YOLO 漏檢對照）。
可續跑：已完成的類別會跳過。

用法（在 anomalib 子資料夾裡跑）：
  cd D:\\edge_aoi_yolo\\anomalib
  ..\\.venv_anomalib\\Scripts\\activate
  python anomalib_train.py --cats cable                 # 先試 1 類
  python anomalib_train.py --all                         # 全 15 類
  python anomalib_train.py --cats cable capsule hazelnut # 指定幾類
"""
import argparse, os, json, warnings
warnings.filterwarnings("ignore")

ALL_CATS = ["bottle","cable","capsule","carpet","grid","hazelnut","leather",
            "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]

RESULT_JSON = "anomalib_results.json"

def load_results():
    if os.path.exists(RESULT_JSON):
        return json.load(open(RESULT_JSON, encoding="utf-8"))
    return {}

def save_results(r):
    json.dump(r, open(RESULT_JSON,"w",encoding="utf-8"), indent=2, ensure_ascii=False)

def train_one(cat, epochs, root):
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd

    datamodule = MVTecAD(
        root=root,
        category=cat,
        train_batch_size=1,   # EfficientAd 要求 batch=1
        eval_batch_size=16,
        num_workers=0,         # Windows 穩定
    )
    model = EfficientAd()
    engine = Engine(max_epochs=epochs, accelerator="gpu", devices=1)
    engine.fit(datamodule=datamodule, model=model)
    test_results = engine.test(datamodule=datamodule, model=model)
    # test_results 通常是 list[dict]
    print(f"\n[{cat}] raw test_results:", test_results)
    metrics = {}
    if isinstance(test_results, list) and test_results:
        metrics = test_results[0]
    elif isinstance(test_results, dict):
        metrics = test_results
    # 抓 image AUROC（鍵名可能不同，全部存下來）
    return {k: (float(v) if hasattr(v,"__float__") else v) for k,v in metrics.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--root", default="../datasets/mvtec")  # 在 anomalib/ 子資料夾跑，往上一層找資料集
    args = ap.parse_args()

    cats = ALL_CATS if args.all else (args.cats or ["cable"])
    results = load_results()

    for cat in cats:
        if cat in results:
            print(f"[跳過] {cat} 已完成（AUROC 記錄存在）")
            continue
        print("\n" + "="*60)
        print(f"訓練 EfficientAD：{cat}  ({args.epochs} epochs)")
        print("="*60)
        try:
            m = train_one(cat, args.epochs, args.root)
            results[cat] = m
            save_results(results)
            print(f"[完成] {cat} → {m}")
        except Exception as e:
            print(f"[失敗] {cat}: {e}")
            import traceback; traceback.print_exc()
            break  # 出錯就停，方便你貼錯誤給我

    # 總表
    print("\n" + "="*60)
    print("已完成類別的指標摘要")
    print("="*60)
    for cat, m in results.items():
        # 嘗試找 image AUROC 的鍵
        auroc = None
        for k,v in m.items():
            if "image" in k.lower() and ("auroc" in k.lower() or "auc" in k.lower()):
                auroc = v
        print(f"  {cat:12s}  image_AUROC={auroc}   (all keys: {list(m.keys())})")

if __name__ == "__main__":
    main()
