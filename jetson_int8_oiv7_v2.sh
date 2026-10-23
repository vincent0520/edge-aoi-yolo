#!/usr/bin/env bash
DATA_YAML="/home/jetson/edge-aoi-yolo/configs/openimages.yaml"
IMGSZ=640
RUNS="/home/jetson/edge-aoi-yolo/runs/detect/runs"
declare -A DIR=(
  [n]="oiv7_50k_v8n"
  [s]="oiv7_50k_yolov8s"
  [m]="oiv7_50k_yolov8m"
  [l]="oiv7_50k_yolov8l"
  [x]="oiv7_50k_yolov8x-2"
)
OUT="int8_result_OIV7.txt"; : > "$OUT"
for SZ in n s m l x; do
  PT="${RUNS}/${DIR[$SZ]}/weights/best.pt"
  ENG="${PT%.pt}_int8.engine"
  echo "==================== MVTec INT8 / ${SZ} ===================="
  # 砍掉舊的（batch=8 或 9/20 的假檔），一律重建
  rm -f "$ENG"
  echo "[export INT8 batch=1] $PT"
  yolo export model="$PT" format=engine quantize=8 data="$DATA_YAML" imgsz=$IMGSZ batch=1 device=0
  mv "${PT%.pt}.engine" "$ENG"
  echo "[val] $ENG"
  LOG=$(yolo val model="$ENG" data="$DATA_YAML" imgsz=$IMGSZ batch=1 device=0 2>&1)
  echo "$LOG" | tail -4
  LINE=$(echo "$LOG" | grep -E "^\s*all" | tail -1)
  echo "${SZ}  P=$(echo "$LINE"|awk '{print $(NF-3)}')  R=$(echo "$LINE"|awk '{print $(NF-2)}')  mAP50=$(echo "$LINE"|awk '{print $(NF-1)}')  mAP50-95=$(echo "$LINE"|awk '{print $NF}')" >> "$OUT"
done
echo ""; echo "########## MVTec INT8 彙總（貼回來）##########"; cat "$OUT"
