# -*- coding: utf-8 -*-
"""匯出 resnet18 + coreset 疊加版 PatchCore 的 ONNX。
用法：
  python export_r18coreset.py --backbone resnet18 --ratio 0.01 --cat cable
"""
import argparse, warnings
warnings.filterwarnings("ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    ap.add_argument("--ratio", type=float, default=0.01)
    ap.add_argument("--cat", default="cable")
    ap.add_argument("--imgsz", type=int, default=256)
    args = ap.parse_args()

    from anomalib.models import Patchcore
    from anomalib.engine import Engine
    from anomalib.deploy import ExportType

    tag = f"{args.backbone}_cs{args.ratio}"
    ckpt = (f"results_r18coreset/{tag}/Patchcore/MVTecAD/"
            f"{args.cat}/latest/weights/lightning/model.ckpt")
    out_root = f"onnx_export_{tag}"

    model = Patchcore(backbone=args.backbone, layers=args.layers,
                      coreset_sampling_ratio=args.ratio)
    engine = Engine()
    engine.export(
        model=model,
        export_type=ExportType.ONNX,
        export_root=out_root,
        ckpt_path=ckpt,
        input_size=(args.imgsz, args.imgsz),
    )
    print(f"\n[完成] ONNX 已匯出 {tag}")
    print(f"       輸出位置：{out_root}（onnx 在 weights/onnx/ 下，若有 .data 檔要一起傳 Jetson）")

if __name__ == "__main__":
    main()
