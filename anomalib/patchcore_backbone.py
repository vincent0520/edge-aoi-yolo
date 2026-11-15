# -*- coding: utf-8 -*-
"""
PatchCore 輕量 backbone 實驗（優化速度）
預設 wide_resnet50_2 很重，換成輕量 backbone 看能否加速。
訓練 + 驗證 AUROC，之後 export ONNX 到 Jetson 量速度。

用法（anomalib 資料夾、.venv_anomalib）：
  cd D:\\edge_aoi_yolo\\anomalib
  ..\\.venv_anomalib\\Scripts\\activate
  python patchcore_backbone.py --backbone resnet18 --cat cable
  python patchcore_backbone.py --backbone wide_resnet50_2 --cat cable   # 原始對照
"""
import argparse, os, json, warnings
warnings.filterwarnings("ignore")

RESULT_JSON = "patchcore_backbone_results.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18",
                    help="resnet18 / resnet34 / wide_resnet50_2 等")
    ap.add_argument("--cat", default="cable")
    ap.add_argument("--layers", nargs="+", default=["layer2","layer3"])
    ap.add_argument("--root", default="../datasets/mvtec")
    args = ap.parse_args()

    from anomalib.data import MVTecAD
    from anomalib.models import Patchcore
    from anomalib.engine import Engine

    datamodule = MVTecAD(root=args.root, category=args.cat,
                         train_batch_size=32, eval_batch_size=16, num_workers=0)
    # 換 backbone
    model = Patchcore(backbone=args.backbone, layers=args.layers)
    engine = Engine(max_epochs=1, accelerator="gpu", devices=1,
                    default_root_dir=f"results_backbone/{args.backbone}")
    engine.fit(datamodule=datamodule, model=model)
    res = engine.test(datamodule=datamodule, model=model)
    m = res[0] if isinstance(res, list) and res else res
    auroc = m.get("image_AUROC")
    print(f"\n[完成] PatchCore backbone={args.backbone} cat={args.cat}")
    print(f"       image_AUROC = {auroc}")

    # 存結果
    results = json.load(open(RESULT_JSON, encoding="utf-8")) if os.path.exists(RESULT_JSON) else {}
    results[f"{args.backbone}:{args.cat}"] = {k:(float(v) if hasattr(v,'__float__') else v) for k,v in m.items()}
    json.dump(results, open(RESULT_JSON,"w",encoding="utf-8"), indent=2, ensure_ascii=False)

    # 找 checkpoint 路徑提示（供 export）
    print("\n下一步：export ONNX → 傳 Jetson 量速度，跟 wide_resnet50_2 原始版對照。")
    print("checkpoint 在 results_backbone/%s/ 底下" % args.backbone)

if __name__ == "__main__":
    main()
