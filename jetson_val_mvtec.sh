#!/usr/bin/env bash
set -e
DATA_YAML="/home/jetson/edge-aoi-yolo/configs/mvtec.yaml"   # ← 確認指向 Jetson 本機 MVTe
IMGSZ=640
RUNS="/home/jetson/edge-aoi-yolo/runs/detect/runs"

declare -A DIR=(
  [s]="mvtec_v8s_baseline"
  [m]="mvtec_yolov8m_baseline-3"
  [l]="mvtec_yolov8l_baseline"
  [x]="mvtec_yolov8x_baseline"
)

OUT="val_result_MVTec.txt"; : > "$OUT"

for SZ in s m l x; do
  PT="${RUNS}/${DIR[$SZ]}/weights/best.pt"
  ENG="${PT%.pt}.engine"
  echo "==================== MVTec / ${SZ} ===================="

  if [ ! -f "$PT" ]; then
    echo "[略過] 找不到 $PT"; echo "${SZ}  mAP50=NA  mAP50-95=NA" >> "$OUT"; continue
  fi

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

echo ""; echo "########## MVTec 彙總（貼回來）##########"; cat "$OUT"
