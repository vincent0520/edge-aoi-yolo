# -*- coding: utf-8 -*-
"""匯出 coreset 掃描版 PatchCore 的 ONNX，供 Jetson 量速度對照用。
用法：
  python export_coreset.py --ratio 0.01 --cat cable
"""
import argparse, warnings
warnings.filterwarnings("ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, required=True,
                    help="coreset_sampling_ratio，要跟訓練時一致")
    ap.add_argument("--cat", default="cable")
    ap.add_argument("--imgsz", type=int, default=256)
    args = ap.parse_args()

    from anomalib.models import Patchcore
    from anomalib.engine import Engine
    from anomalib.deploy import ExportType

    ckpt = (f"results_coreset/r{args.ratio}/Patchcore/MVTecAD/"
            f"{args.cat}/latest/weights/lightning/model.ckpt")
    out_root = f"onnx_export_coreset_r{args.ratio}"

    # coreset_sampling_ratio 要跟訓練一致，才能正確載入 memory bank
    model = Patchcore(coreset_sampling_ratio=args.ratio)
    engine = Engine()
    engine.export(
        model=model,
        export_type=ExportType.ONNX,
        export_root=out_root,
        ckpt_path=ckpt,
        input_size=(args.imgsz, args.imgsz),
    )
    print(f"\n[完成] ONNX 已匯出 coreset ratio={args.ratio}")
    print(f"       輸出位置：{out_root}")

if __name__ == "__main__":
    main()
