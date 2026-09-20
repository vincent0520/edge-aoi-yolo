import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 五个模型：正确资料夹 + 参数量
runs = {
    "n": ("runs/detect/runs/mvtec_v8n_baseline", 3.2),
    "s": ("runs/detect/runs/mvtec_v8s_baseline", 11.2),
    "m": ("runs/detect/runs/mvtec_yolov8m_baseline-3", 25.9),
    "l": ("runs/detect/runs/mvtec_yolov8l_baseline", 43.7),
    "x": ("runs/detect/runs/mvtec_yolov8x_baseline", 68.2),
}

def last_map(path):
    with open(os.path.join(path, "results.csv")) as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    epoch = list(last.values())[0].strip()
    m50 = m5095 = None
    for k in last:
        kk = k.strip()
        if kk == "metrics/mAP50(B)":
            m50 = float(last[k])
        if kk == "metrics/mAP50-95(B)":
            m5095 = float(last[k])
    return epoch, m50, m5095

data = []
for size, (path, params) in runs.items():
    if os.path.exists(os.path.join(path, "results.csv")):
        epoch, m50, m5095 = last_map(path)
        data.append([size, params, m50, m5095])
        print(f"{size}: epoch={epoch}  params={params}M  mAP50={m50:.4f}  mAP50-95={m5095:.4f}")
    else:
        print(f"{size}: PATH NOT FOUND -> {path}")

# 写 CSV
os.makedirs("results/csv", exist_ok=True)
with open("results/csv/mvtec_model_size_sweep.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["model","params_M","mAP50","mAP50_95"])
    for d in data:
        w.writerow([d[0], d[1], round(d[2],4), round(d[3],4)])
print("Wrote results/csv/mvtec_model_size_sweep.csv")

# 画饱和曲线
sizes=[d[0] for d in data]; params=[d[1] for d in data]; map50=[d[2] for d in data]
plt.figure(figsize=(8,5))
plt.plot(params, map50, "o-", linewidth=2, markersize=9)
for i,s in enumerate(sizes):
    plt.annotate(f"  {s}  {map50[i]:.3f}", (params[i], map50[i]), fontsize=10)
plt.xlabel("Parameters (M)"); plt.ylabel("mAP@50")
plt.title("MVTec: Model Size vs Accuracy")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("results/model_size_sweep.png", dpi=120)
print("Wrote results/model_size_sweep.png")
