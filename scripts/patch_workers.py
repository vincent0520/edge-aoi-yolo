# patch_workers.py
p = r"scripts\yolov8_pruning.py"
s = open(p, encoding="utf-8").read()

# 在 epochs 設定後面加一行 workers=0（Windows 最穩）
old = "pruning_cfg['epochs'] = 50"
new = "pruning_cfg['epochs'] = 50\n    pruning_cfg['workers'] = 0"
assert old in s, "找不到目標行"
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("patched workers=0")