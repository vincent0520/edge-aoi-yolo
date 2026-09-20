import cv2, os, glob, random, shutil, yaml
random.seed(0)

SRC = "datasets/mvtec"
DST = "datasets/mvtec_yolo"
VAL_RATIO = 0.2

# 15 objects -> class id (alphabetical)
objects = sorted([d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d))])
cls_id = {name: i for i, name in enumerate(objects)}
print("Classes:", cls_id)

# prepare output dirs
for split in ["train", "val"]:
    os.makedirs(f"{DST}/images/{split}", exist_ok=True)
    os.makedirs(f"{DST}/labels/{split}", exist_ok=True)

def mask_to_boxes(mask_path):
    m = cv2.imread(mask_path, 0)
    if m is None:
        return None, None, None
    h, w = m.shape
    _, mbin = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mbin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw*bh < 25:   # skip tiny noise
            continue
        cx, cy = (x+bw/2)/w, (y+bh/2)/h
        boxes.append((cx, cy, bw/w, bh/h))
    return boxes, h, w

pairs = []  # (image_path, mask_path, class_id)
for obj in objects:
    cid = cls_id[obj]
    test_dir = os.path.join(SRC, obj, "test")
    gt_dir = os.path.join(SRC, obj, "ground_truth")
    if not os.path.isdir(test_dir):
        continue
    for defect in os.listdir(test_dir):
        if defect == "good":
            continue
        for img in glob.glob(os.path.join(test_dir, defect, "*.png")):
            base = os.path.splitext(os.path.basename(img))[0]
            mask = os.path.join(gt_dir, defect, base + "_mask.png")
            if os.path.exists(mask):
                pairs.append((img, mask, cid, obj, defect, base))

print(f"Total defect images: {len(pairs)}")
random.shuffle(pairs)
n_val = int(len(pairs) * VAL_RATIO)

for i, (img, mask, cid, obj, defect, base) in enumerate(pairs):
    split = "val" if i < n_val else "train"
    boxes, h, w = mask_to_boxes(mask)
    if not boxes:
        continue
    uid = f"{obj}_{defect}_{base}"
    shutil.copy(img, f"{DST}/images/{split}/{uid}.png")
    with open(f"{DST}/labels/{split}/{uid}.txt", "w") as f:
        for (cx, cy, bw, bh) in boxes:
            f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

# write yaml
data = {
    "path": os.path.abspath(DST).replace("\\", "/"),
    "train": "images/train",
    "val": "images/val",
    "nc": len(objects),
    "names": objects,
}
with open("configs/mvtec.yaml", "w") as f:
    yaml.dump(data, f, sort_keys=False, allow_unicode=True)
print("Wrote configs/mvtec.yaml")
print("Done.")
