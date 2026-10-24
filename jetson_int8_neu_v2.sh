#!/usr/bin/env bash
DATA_YAML="/home/jetson/edge-aoi-yolo/configs/neu-det.yaml"
IMGSZ=640
RUNS="/home/jetson/edge-aoi-yolo/runs/detect/runs"
declare -A DIR=(
  [n]="neu_v8n_baseline"
  [s]="neu_v8s_baseline-2"
  [m]="neu_yolov8m_baseline"
  [l]="neu_yolov8l_baseline"
  [x]="neu_yolov8x_baseline"
)
OUT="int8_result_NEU.txt"; : > "$OUT"
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
