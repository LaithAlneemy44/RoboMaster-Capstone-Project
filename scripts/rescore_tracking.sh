#!/usr/bin/env bash
# Recompute the 7-clip detector-fed tracking rows.
#
# The originals were produced before make_detector applied the "car"-only class filter,
# so the tracker was fed armor plates, bases and watchers alongside robots while the
# ground truth contains only robots. On arc04 that inflated false positives to 503 and
# track count to 51; with the filter they are 15 and 21, and MOTA moves 0.196 -> 0.481.
# The rows described a pipeline that no longer exists.
#
# CPU inference on purpose: the GPU is training, and this is accuracy, not timing, so
# contention costs wall-clock and nothing else.
set -u
PY=.venv/Scripts/python.exe
DET=runs/detect/yolo_960/weights/best.pt
for seq in arc01 arc02 arc03 arc04 arc05 arc06 arc07; do
  for trk in classical sort; do
    echo "=== ${seq} / ${trk} ==="
    $PY scripts/run_trackers.py --seq "data/tracking/${seq}" --tracker "$trk" \
        --detector "$DET" --device cpu 2>&1 | tail -2
  done
done
echo "ALL RUNS DONE"
