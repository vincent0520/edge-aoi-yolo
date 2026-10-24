# prune_mvtec_l.py  —— 桌機（4070Ti）執行
# 結構化剪枝 YOLOv8l @ MVTec，含微調與 ONNX 匯出
import torch
import torch_pruning as tp
from ultralytics import YOLO

# ========== 設定（改成你桌機的實際路徑）==========
PT     = r"D:\edge_aoi_yolo\runs\detect\runs\mvtec_yolov8l_baseline\weights\best.pt"
DATA   = r"D:\edge_aoi_yolo\configs\mvtec.yaml"
RATIO  = 0.5        # 剪 50% 通道
IMGSZ  = 640
EPOCHS = 100
NAME   = "mvtec_l_prune50"

# ========== 1) 載入並剪枝 ==========
yolo = YOLO(PT)
model = yolo.model
model.eval()
example = torch.randn(1, 3, IMGSZ, IMGSZ)

base_macs, base_params = tp.utils.count_ops_and_params(model, example)
print(f"[剪枝前] 參數 {base_params/1e6:.2f}M  MACs {base_macs/1e9:.2f}G")

# 忽略偵測頭（Detect 的 cv2/cv3 輸出層不能剪，會改變輸出維度）
from ultralytics.nn.modules import Detect
ignored_layers = [m for m in model.modules() if isinstance(m, Detect)]

imp = tp.importance.GroupMagnitudeImportance(p=2)
pruner = tp.pruner.GroupNormPruner(
    model, example,
    importance=imp,
    pruning_ratio=RATIO,
    ignored_layers=ignored_layers,
    isomorphic=True,          # YOLO 有 concat/split，需要同構剪枝保持維度一致
    global_pruning=True,
)
pruner.step()

pr_macs, pr_params = tp.utils.count_ops_and_params(model, example)
print(f"[剪枝後] 參數 {pr_params/1e6:.2f}M  MACs {pr_macs/1e9:.2f}G")
print(f"參數 -{(1-pr_params/base_params)*100:.1f}%  MACs -{(1-pr_macs/base_macs)*100:.1f}%")

# 驗證剪枝真的生效（參數必須明顯下降，否則是假剪枝）
assert pr_params < base_params * 0.9, "剪枝沒生效！參數沒明顯減少"

# 前向測試一次，確認剪枝後模型還能跑
with torch.no_grad():
    y = model(example)
print("[OK] 剪枝後前向測試通過")

# ========== 2) 微調（避開 yaml 重建陷阱）==========
# 把剪枝後的 module 塞回 YOLO 物件，並關掉會觸發重建的設定
yolo.model = model
# 關鍵：清掉 yaml 相關，讓 trainer 用現有 module 而非重建
yolo.model.args = yolo.overrides if hasattr(yolo, "overrides") else {}

yolo.train(
    data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=16,
    name=NAME, seed=0, pretrained=False,
    exist_ok=True,
)

# ========== 3) 匯出 ONNX（傳去 Jetson 建 engine）==========
best = yolo  # train 後 yolo 內部已是 best
best.export(format="onnx", imgsz=IMGSZ, opset=18, simplify=True)
print(f"[完成] 微調後模型與 ONNX 已存在 runs/detect/{NAME}/")