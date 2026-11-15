# -*- coding: utf-8 -*-
"""
PatchCore 雙重優化：resnet18 輕量 backbone + coreset 0.01 小 memory bank。
把兩個加速法疊加，看 Jetson 速度能否再往下，AUROC 是否維持。
訓練 → 之後用 export 腳本匯出 ONNX → 傳 Jetson 量速度。
用法：
  python train_r18_coreset.py --cat cable --backbone resnet18 --ratio 0.01
"""
import argparse, json, os, warnings
warnings.filterwarnings("ignore")

RESULT_JSON = "r18_coreset_results.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="cable")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    ap.add_argument("--ratio", type=float, default=0.01)
    ap.add_argument("--root", default="../datasets/mvtec")
    args = ap.parse_args()

    from anomalib.data import MVTecAD
    from anomalib.models import Patchcore
    from anomalib.engine import Engine

    datamodule = MVTecAD(root=args.root, category=args.cat,
                         train_batch_size=32, eval_batch_size=16, num_workers=0)
    # 雙重優化：輕量 backbone + 小 coreset
    model = Patchcore(backbone=args.backbone, layers=args.layers,
                      coreset_sampling_ratio=args.ratio)
    tag = f"{args.backbone}_cs{args.ratio}"
    engine = Engine(max_epochs=1, accelerator="gpu", devices=1,
                    default_root_dir=f"results_r18coreset/{tag}")
    engine.fit(datamodule=datamodule, model=model)
    res = engine.test(datamodule=datamodule, model=model)
    m = res[0] if isinstance(res, list) and res else res
    auroc = m.get("image_AUROC")
    print(f"\n[完成] backbone={args.backbone} coreset={args.ratio} cat={args.cat}")
    print(f"       image_AUROC = {auroc}")

    results = json.load(open(RESULT_JSON, encoding="utf-8")) if os.path.exists(RESULT_JSON) else {}
    results[f"{args.cat}:{tag}"] = {
        "backbone": args.backbone, "ratio": args.ratio,
        **{k: (float(v) if hasattr(v, "__float__") else v) for k, v in m.items()}
    }
    json.dump(results, open(RESULT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\ncheckpoint 在 results_r18coreset/{tag}/ 底下")
    print(f"下一步 export：python export_r18coreset.py --backbone {args.backbone} --ratio {args.ratio} --cat {args.cat}")

if __name__ == "__main__":
    main()
