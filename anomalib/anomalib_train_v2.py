# -*- coding: utf-8 -*-
"""
異常偵測訓練（MVTec）— 預設 PatchCore（不需 imagenette，最省事）
逐類訓練，存每類的 Image AUROC（與 YOLO 漏檢對照）。可續跑。

用法（在 anomalib 子資料夾、.venv_anomalib 環境）：
  cd D:\\edge_aoi_yolo\\anomalib
  ..\\.venv_anomalib\\Scripts\\activate
  python anomalib_train.py --cats cable                  # 先試 1 類
  python anomalib_train.py --all                          # 全 15 類
  python anomalib_train.py --model efficientad --all      # 改用 EfficientAD（需 imagenette）
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

def build_model(name):
    if name == "patchcore":
        from anomalib.models import Patchcore
        return Patchcore()           # 不需 imagenette
    elif name == "efficientad":
        from anomalib.models import EfficientAd
        return EfficientAd()         # 需 imagenette
    elif name == "fastflow":
        from anomalib.models import Fastflow
        return Fastflow()
    else:
        raise ValueError("unknown model: " + name)

def train_one(cat, epochs, root, model_name):
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine

    # PatchCore 用較大 batch 可加速；EfficientAD 要 batch=1
    tb = 1 if model_name == "efficientad" else 32
    datamodule = MVTecAD(
        root=root, category=cat,
        train_batch_size=tb, eval_batch_size=16, num_workers=0,
    )
    model = build_model(model_name)
    # PatchCore 是 memory-bank，1 epoch 即可；EfficientAD 要多 epoch
    me = 1 if model_name in ("patchcore",) else epochs
    engine = Engine(max_epochs=me, accelerator="gpu", devices=1)
    engine.fit(datamodule=datamodule, model=model)
    test_results = engine.test(datamodule=datamodule, model=model)
    print(f"\n[{cat}] raw test_results:", test_results)
    metrics = {}
    if isinstance(test_results, list) and test_results:
        metrics = test_results[0]
    elif isinstance(test_results, dict):
        metrics = test_results
    return {k: (float(v) if hasattr(v,"__float__") else v) for k,v in metrics.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="patchcore",
                    choices=["patchcore","efficientad","fastflow"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--root", default="../datasets/mvtec")
    args = ap.parse_args()

    cats = ALL_CATS if args.all else (args.cats or ["cable"])
    results = load_results()

    for cat in cats:
        key = f"{args.model}:{cat}"
        if key in results:
            print(f"[跳過] {key} 已完成")
            continue
        print("\n" + "="*60)
        print(f"訓練 {args.model}：{cat}")
        print("="*60)
        try:
            m = train_one(cat, args.epochs, args.root, args.model)
            results[key] = m
            save_results(results)
            print(f"[完成] {key} → {m}")
        except Exception as e:
            print(f"[失敗] {key}: {e}")
            import traceback; traceback.print_exc()
            break

    print("\n" + "="*60)
    print(f"已完成摘要（model={args.model}）")
    print("="*60)
    for key, m in results.items():
        if not key.startswith(args.model + ":"): continue
        auroc = None
        for k,v in m.items():
            if "image" in k.lower() and ("auroc" in k.lower() or "auc" in k.lower()):
                auroc = v
        print(f"  {key:24s} image_AUROC={auroc}   keys={list(m.keys())}")

if __name__ == "__main__":
    main()
