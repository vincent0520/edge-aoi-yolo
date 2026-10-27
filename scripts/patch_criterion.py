# patch_criterion.py
p = r"scripts\yolov8_pruning.py"
s = open(p, encoding="utf-8").read()

old = "        model.train_v2(pruning=True, **pruning_cfg)"
new = (
    "        # 剪枝後 criterion 可能被設為 None，刪掉讓 forward 時自動重建\n"
    "        if hasattr(model.model, 'criterion'):\n"
    "            try:\n"
    "                del model.model.criterion\n"
    "            except Exception:\n"
    "                model.model.criterion = None\n"
    "        model.train_v2(pruning=True, **pruning_cfg)"
)

assert old in s, "找不到目標行！縮排可能不符"
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("patched OK")