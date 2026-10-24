# patch_mvtec.py
p = r"scripts\yolov8_pruning.py"
s = open(p, encoding="utf-8").read()

s = s.replace(
    'pruning_cfg[\'data\'] = "coco128.yaml"',
    'pruning_cfg[\'data\'] = r"D:/edge_aoi_yolo/configs/mvtec.yaml"'
)
s = s.replace(
    "pruning_cfg['epochs'] = 10",
    "pruning_cfg['epochs'] = 50"
)

open(p, "w", encoding="utf-8").write(s)
print("patched OK")