#!/usr/bin/env bash
# Rebuild results/detection.csv with the added precision / recall / mean-IoU columns.
#
# The CSV is REBUILT, not appended to: three new columns mean an appended row would be
# wider than the existing header, and DictWriter emits values positionally, so every new
# row would silently land in the wrong columns.
#
# Predictions run on CPU on purpose - the GPU is training Faster R-CNN, and a second
# CUDA context would compete for the 6 GB it is already most of the way through.
set -u
PY=.venv/Scripts/python.exe
mkdir -p results/predictions

for cfg in strict balanced tight loose paired; do
  out="results/predictions/classical_${cfg}.json"
  [ -f "$out" ] || { echo "predict classical_${cfg}"; \
    $PY scripts/predict_to_coco.py --family classical --config "$cfg" \
        --out "$out" --device cpu 2>&1 | tail -2; }
done

gn="results/predictions/ssd_small_960_groupnorm.json"
[ -f "$gn" ] || { echo "predict ssd_small_960_groupnorm"; \
  $PY scripts/predict_to_coco.py --family ssd --weights runs/ssd/ssd_small_960_gn/best.pt \
      --imgsz 960 --out "$gn" --device cpu 2>&1 | tail -2; }

[ -f results/detection.csv ] && mv results/detection.csv results/detection.csv.bak
echo
for f in results/predictions/*.json; do
  name=$(basename "$f" .json)
  echo "score ${name}"
  $PY scripts/evaluate_detection.py --predictions "$f" --name "$name" 2>&1 \
    | grep -E "precision |mAP@\[" | sed 's/^/    /'
done
echo "DONE"
