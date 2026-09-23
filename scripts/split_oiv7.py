import os, glob, random, shutil
random.seed(0)

base = "datasets/openimages_yolo"
imgs = [f for f in glob.glob(f"{base}/images/val/*") if os.path.isfile(f)]
random.shuffle(imgs)

n_val = int(len(imgs) * 0.2)
val_set = set(imgs[:n_val])

for sub in ["images/train","labels/train"]:
    os.makedirs(f"{base}/{sub}", exist_ok=True)

moved = 0
for img in imgs:
    if img in val_set:
        continue
    name = os.path.basename(img)
    stem = os.path.splitext(name)[0]
    lbl = f"{base}/labels/val/{stem}.txt"
    shutil.move(img, f"{base}/images/train/{name}")
    if os.path.exists(lbl):
        shutil.move(lbl, f"{base}/labels/train/{stem}.txt")
    moved += 1

print(f"Total {len(imgs)}, moved to train {moved}, kept in val {n_val}")
