# -*- coding: utf-8 -*-
"""
Export anomalib 模型為 ONNX（供 Jetson TensorRT 部署）
從訓練好的 checkpoint 載入，export 成 ONNX。

用法（.venv_anomalib 環境，在 anomalib 資料夾）：
  cd D:\\edge_aoi_yolo\\anomalib
  ..\\.venv_anomalib\\Scripts\\activate
  python anomalib_export.py --model patchcore --cat cable
  python anomalib_export.py --model efficientad --cat cable
  python anomalib_export.py --model fastflow --cat cable
"""
import argparse, os, warnings
warnings.filterwarnings("ignore")

# anomalib 的 results 子資料夾名（注意大小寫）
MODEL_DIR = {"patchcore":"Patchcore", "efficientad":"EfficientAd", "fastflow":"Fastflow"}

def build_model(name):
    if name == "patchcore":
        from anomalib.models import Patchcore; return Patchcore()
    elif name == "efficientad":
        from anomalib.models import EfficientAd; return EfficientAd()
    elif name == "fastflow":
        from anomalib.models import Fastflow; return Fastflow()
    raise ValueError(name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["patchcore","efficientad","fastflow"])
    ap.add_argument("--cat", required=True)
    ap.add_argument("--imgsz", type=int, default=256, help="anomalib 預設 256")
    ap.add_argument("--out", default="onnx_export")
    args = ap.parse_args()

    from anomalib.engine import Engine
    from anomalib.deploy import ExportType

    mdir = MODEL_DIR[args.model]
    ckpt = f"results/{mdir}/MVTecAD/{args.cat}/v0/weights/lightning/model.ckpt"
    if not os.path.exists(ckpt):
        # 試 latest
        ckpt2 = f"results/{mdir}/MVTecAD/{args.cat}/latest/weights/lightning/model.ckpt"
        ckpt = ckpt2 if os.path.exists(ckpt2) else ckpt
    print(f"載入 checkpoint: {ckpt}")
    if not os.path.exists(ckpt):
        print(f"[錯誤] 找不到 checkpoint，請確認已訓練 {args.model}:{args.cat}")
        return

    model = build_model(args.model)
    engine = Engine()
    out_root = os.path.join(args.out, args.model, args.cat)
    os.makedirs(out_root, exist_ok=True)

    # export ONNX
    exported = engine.export(
        model=model,
        export_type=ExportType.ONNX,
        export_root=out_root,
        ckpt_path=ckpt,
        input_size=(args.imgsz, args.imgsz),
    )
    print(f"\n[完成] ONNX 已 export 到: {exported}")
    print("下一步：用 MobaXterm 把這個 .onnx 傳到 Jetson，再轉 TensorRT engine。")

if __name__ == "__main__":
    main()
