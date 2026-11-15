# -*- coding: utf-8 -*-
"""匯出 resnet18 版 PatchCore 的 ONNX，供 Jetson 量速度對照用。"""
import argparse, warnings
warnings.filterwarnings("ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--cat", default="cable")
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    args = ap.parse_args()

    from anomalib.models import Patchcore
    from anomalib.engine import Engine
    from anomalib.deploy import ExportType

    ckpt = (f"results_backbone/{args.backbone}/Patchcore/MVTecAD/"
            f"{args.cat}/latest/weights/lightning/model.ckpt")
    out_root = f"onnx_export_{args.backbone}"

    model = Patchcore(backbone=args.backbone, layers=args.layers)
    engine = Engine()
    engine.export(
        model=model,
        export_type=ExportType.ONNX,
        export_root=out_root,
        ckpt_path=ckpt,
        input_size=(args.imgsz, args.imgsz),
    )
    print(f"\n[完成] ONNX 已匯出")
    print(f"       backbone={args.backbone} cat={args.cat} imgsz={args.imgsz}")
    print(f"       輸出位置：{out_root}")

if __name__ == "__main__":
    main()