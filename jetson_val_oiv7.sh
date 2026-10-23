#!/usr/bin/env bash
set -e
# 先修 openimages.yaml 的 path（若已是正確路徑，這行無害）
sed -i 's|^path:.*|path: /home/jetson/edge-aoi-yolo/datasets/openimages_yolo|' \
  /home/jetson/edge-aoi-yolo/configs/openimages.yaml

DATA_YAML="/home/jetson/edge-aoi-yolo/configs/openimages.yaml"
IMGSZ=640
RUNS="/home/jetson/edge-aoi-yolo/runs/detect/runs"

declare -A DIR=(
  [s]="oiv7_50k_yolov8s"
  [m]="oiv7_50k_yolov8m"
  [l]="oiv7_50k_yolov8l"
  [x]="oiv7_50k_yolov8x-2"
)

OUT="val_result_OIV7.txt"; : > "$OUT"
for SZ in s m l x; do
  PT="${RUNS}/${DIR[$SZ]}/weights/best.pt"
  ENG="${PT%.pt}.engine"
  echo "==================== OIV7 / ${SZ} ===================="
  if [ ! -f "$PT" ]; then echo "[略過] 找不到 $PT"; echo "${SZ}  mAP50=NA  mAP50-95=NA" >> "$OUT"; continue; fi
  if [ ! -f "$ENG" ]; then
    echo "[export] $PT → engine (FP16)"
    yolo export model="$PT" format=engine half=True imgsz=$IMGSZ device=0
  else
    echo "[export] 已有 $ENG，略過"
  fi
  echo "[val] $ENG"
  LOG=$(yolo val model="$ENG" data="$DATA_YAML" imgsz=$IMGSZ device=0 2>&1)
  echo "$LOG"
  LINE=$(echo "$LOG" | grep -E "^\s*all" | tail -1)
  echo "${SZ}  mAP50=$(echo "$LINE" | awk '{print $(NF-1)}')  mAP50-95=$(echo "$LINE" | awk '{print $NF}')" >> "$OUT"
done
echo ""; echo "########## OIV7 彙總（貼回來）##########"; cat "$OUT"
