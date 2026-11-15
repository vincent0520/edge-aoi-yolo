# -*- coding: utf-8 -*-
"""
PatchCore coreset_sampling_ratio 掃描 —— 針對真正的瓶頸(距離 matmul)對症下藥。
memory bank 越小 → 距離矩陣越小 → matmul 越快。看速度能降多少、AUROC 掉多少。
用法：
  python coreset_sweep.py --cat cable --ratios 0.1 0.05 0.01
"""
import argparse, json, os, warnings
warnings.filterwarnings("ignore")

RESULT_JSON = "coreset_sweep_results.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="cable")
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.05, 0.01])
    ap.add_argument("--root", default="../datasets/mvtec")
    args = ap.parse_args()

    from anomalib.data import MVTecAD
    from anomalib.models import Patchcore
    from anomalib.engine import Engine

    results = json.load(open(RESULT_JSON, encoding="utf-8")) if os.path.exists(RESULT_JSON) else {}

    for ratio in args.ratios:
        print(f"\n===== coreset_sampling_ratio = {ratio} =====")
        datamodule = MVTecAD(root=args.root, category=args.cat,
                             train_batch_size=32, eval_batch_size=16, num_workers=0)
        model = Patchcore(coreset_sampling_ratio=ratio)
        engine = Engine(max_epochs=1, accelerator="gpu", devices=1,
                        default_root_dir=f"results_coreset/r{ratio}")
        engine.fit(datamodule=datamodule, model=model)
        res = engine.test(datamodule=datamodule, model=model)
        m = res[0] if isinstance(res, list) and res else res
        auroc = m.get("image_AUROC")
        print(f"       ratio={ratio}  image_AUROC={auroc}")
        results[f"{args.cat}:ratio{ratio}"] = {
            "ratio": ratio,
            **{k: (float(v) if hasattr(v, "__float__") else v) for k, v in m.items()}
        }
        json.dump(results, open(RESULT_JSON, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)

    print("\n[完成] 掃描結束。checkpoint 在 results_coreset/ 底下，之後各自 export 量 Jetson 速度。")

if __name__ == "__main__":
    main()
