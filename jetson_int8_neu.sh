#!/usr/bin/env bash
DATA_YAML="/home/jetson/edge-aoi-yolo/configs/neu-det.yaml"
IMGSZ=640; BATCH=8
RUNS="/home/jetson/edge-aoi-yolo/runs/detect/runs"
declare -A DIR=(
  [n]="neu_v8n_baseline" [s]="neu_v8s_baseline-2" [m]="neu_yolov8m_baseline"
  [l]="neu_yolov8l_baseline" [x]="neu_yolov8x_baseline"
)
OUT="int8_result_MVTec.txt"; : > "$OUT"
for SZ in n s m l x; do
  PT="${RUNS}/${DIR[$SZ]}/weights/best.pt"
  ENG="${PT%.pt}_int8.engine"      # INT8 engine 另存，不蓋掉 FP16 的 best.engine
  echo "==================== MVTec INT8 / ${SZ} ===================="
  if [ ! -f "$PT" ]; then echo "[略過] 找不到 $PT"; echo "${SZ}  mAP50=NA  mAP50-95=NA" >> "$OUT"; continue; fi
  if [ ! -f "$ENG" ]; then
    echo "[export INT8] $PT"
    # 先 export 成預設檔名，再改名為 _int8.engine
    yolo export model="$PT" format=engine quantize=8 data="$DATA_YAML" imgsz=$IMGSZ batch=$BATCH device=0
    mv "${PT%.pt}.engine" "$ENG"
  else
    echo "[export] 已有 $ENG，略過"
  fi
  echo "[val INT8] $ENG"
  LOG=$(yolo val model="$ENG" data="$DATA_YAML" imgsz=$IMGSZ device=0 2>&1)
  echo "$LOG" | tail -6
  LINE=$(echo "$LOG" | grep -E "^\s*all" | tail -1)
  P=$(echo "$LINE" | awk '{print $(NF-3)}'); R=$(echo "$LINE" | awk '{print $(NF-2)}')
  echo "${SZ}  P=${P}  R=${R}  mAP50=$(echo "$LINE" | awk '{print $(NF-1)}')  mAP50-95=$(echo "$LINE" | awk '{print $NF}')" >> "$OUT"
done
echo ""; echo "########## MVTec INT8 彙總（貼回來）##########"; cat "$OUT"