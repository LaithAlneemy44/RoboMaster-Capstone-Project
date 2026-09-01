# Tracking — CPU cost and accuracy

## CPU cost (the headline — needs no labels)

Averaged over sequences. `RT` marks >= 30 FPS. `track %` is the tracker's share of the detector+tracker pipeline.

| cores | tracker | detect ms | track ms | total ms | FPS | RT | track % | tracks/frame | CPU% of cap | RAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `sort` | 402.5 | 0.4 | 402.8 | 2.5 |  | 0% | 5.1 | 98 | 859 |
| 1 | `classical` | 404.0 | 0.1 | 404.2 | 2.5 |  | 0% | 5.1 | 97 | 830 |
| 1 | `vit` | 402.2 | 94.0 | 496.2 | 2.0 |  | 19% | 5.1 | 97 | 854 |
| 1 | `goturn` | 405.7 | 4309.4 | 4715.1 | 0.2 |  | 91% | 5.1 | 98 | 7292 |
| 2 | `sort` | 256.1 | 0.4 | 256.5 | 3.9 |  | 0% | 5.1 | 96 | 863 |
| 2 | `classical` | 261.5 | 0.1 | 261.6 | 3.8 |  | 0% | 5.1 | 96 | 833 |
| 2 | `vit` | 243.3 | 95.7 | 338.9 | 3.0 |  | 28% | 5.1 | 96 | 864 |
| 2 | `goturn` | 248.4 | 4192.4 | 4440.7 | 0.2 |  | 94% | 5.1 | 55 | 7340 |
| 4 | `sort` | 152.0 | 0.4 | 152.3 | 6.6 |  | 0% | 5.1 | 96 | 870 |
| 4 | `classical` | 187.6 | 0.1 | 187.7 | 5.4 |  | 0% | 5.1 | 97 | 833 |
| 4 | `vit` | 181.8 | 96.3 | 278.1 | 3.6 |  | 34% | 5.1 | 95 | 870 |
| 4 | `goturn` | 170.0 | 4179.0 | 4349.0 | 0.2 |  | 96% | 5.1 | 31 | 7347 |
| 6 | `classical` | 143.5 | 0.1 | 143.7 | 7.0 |  | 0% | 5.1 | 98 | 838 |
| 6 | `sort` | 148.6 | 0.4 | 149.0 | 6.7 |  | 0% | 5.1 | 98 | 869 |
| 6 | `vit` | 155.7 | 101.8 | 257.5 | 3.9 |  | 39% | 5.1 | 97 | 878 |
| 6 | `goturn` | 153.2 | 4178.4 | 4331.6 | 0.2 |  | 96% | 5.1 | 23 | 7347 |

## Accuracy (secondary — see the caveat below)

| tracker | sequence | MOTA | 95% CI | IDF1 | ID switches | MT / ML |
|---|---|---|---|---|---|---|
| `classical_det` | arc01 | 0.397 | [0.357, 0.434] | 0.641 | 8 | 7 / 0 |
| `classical_det` | arc01 | 0.397 | [0.357, 0.434] | 0.641 | 8 | 7 / 0 |
| `classical_det` | arc02 | 0.767 | [0.743, 0.791] | 0.856 | 5 | 7 / 1 |
| `classical_det` | arc02 | 0.767 | [0.743, 0.791] | 0.856 | 5 | 7 / 1 |
| `classical_det` | arc03 | 0.332 | [0.310, 0.355] | 0.660 | 6 | 9 / 0 |
| `classical_det` | arc03 | 0.332 | [0.310, 0.355] | 0.660 | 6 | 9 / 0 |
| `classical_det` | arc04 | 0.196 | [0.164, 0.227] | 0.466 | 13 | 2 / 1 |
| `classical_det` | arc04 | 0.196 | [0.164, 0.227] | 0.466 | 13 | 2 / 1 |
| `classical_det` | arc05 | 0.628 | [0.604, 0.648] | 0.750 | 7 | 10 / 3 |
| `classical_det` | arc05 | 0.628 | [0.604, 0.648] | 0.750 | 7 | 10 / 3 |
| `classical_det` | arc06 | 0.306 | [0.267, 0.347] | 0.475 | 24 | 5 / 6 |
| `classical_det` | arc06 | 0.306 | [0.267, 0.347] | 0.475 | 24 | 5 / 6 |
| `classical_det` | arc07 | 0.508 | [0.453, 0.553] | 0.560 | 10 | 5 / 0 |
| `classical_det` | arc07 | 0.508 | [0.453, 0.553] | 0.560 | 10 | 5 / 0 |
| `sort_det` | arc01 | 0.398 | [0.357, 0.435] | 0.641 | 8 | 7 / 0 |
| `sort_det` | arc01 | 0.398 | [0.357, 0.435] | 0.641 | 8 | 7 / 0 |
| `sort_det` | arc02 | 0.769 | [0.747, 0.794] | 0.857 | 6 | 7 / 1 |
| `sort_det` | arc02 | 0.769 | [0.747, 0.794] | 0.857 | 6 | 7 / 1 |
| `sort_det` | arc03 | 0.330 | [0.308, 0.352] | 0.656 | 8 | 9 / 0 |
| `sort_det` | arc03 | 0.330 | [0.308, 0.352] | 0.656 | 8 | 9 / 0 |
| `sort_det` | arc04 | 0.197 | [0.165, 0.228] | 0.480 | 17 | 2 / 1 |
| `sort_det` | arc04 | 0.197 | [0.165, 0.228] | 0.480 | 17 | 2 / 1 |
| `sort_det` | arc05 | 0.639 | [0.617, 0.660] | 0.751 | 9 | 9 / 3 |
| `sort_det` | arc05 | 0.639 | [0.617, 0.660] | 0.751 | 9 | 9 / 3 |
| `sort_det` | arc06 | 0.310 | [0.272, 0.353] | 0.491 | 25 | 5 / 6 |
| `sort_det` | arc06 | 0.310 | [0.272, 0.353] | 0.491 | 25 | 5 / 6 |
| `sort_det` | arc07 | 0.522 | [0.468, 0.569] | 0.594 | 10 | 6 / 0 |
| `sort_det` | arc07 | 0.522 | [0.468, 0.569] | 0.594 | 10 | 6 / 0 |

**Caveat.** Ground truth here was generated automatically (`scripts/auto_label.py`), because hand-labelling was not affordable within the project. Labels derived from motion association share their assumptions with SORT and the classical tracker and not with the appearance-based GOTURN and VitTrack, so this table is biased toward the former. The CPU table above needs no labels and is unaffected.

Evaluation is detector-fed, not ground-truth-fed. Feeding trackers the reference boxes proved degenerate: they returned the labeller's own tracks and scored ~0.99 MOTA regardless of which tracker ran. Detector-fed asks a real question instead — how much an online tracker loses against an offline reference that saw the whole clip.

## Label quality (no ground truth exists, so this measures implausibility)

| sequence | tracks | median track len | tracks/robot | p99 step/size | flags |
|---|---|---|---|---|---|
| arc01 | 14 | 129.5 | 3.3 | 0.092 | 3.3 tracks per robot on screen - fragmentation |
| arc02 | 9 | 245 | 1.6 | 0.104 | ok |
| arc03 | 13 | 141 | 2.7 | 0.156 | ok |
| arc04 | 9 | 183 | 3.1 | 0.084 | 3.1 tracks per robot on screen - fragmentation |
| arc05 | 17 | 94 | 3.6 | 0.139 | 3.6 tracks per robot on screen - fragmentation |
| arc06 | 19 | 96 | 3.9 | 0.165 | 3.9 tracks per robot on screen - fragmentation |
| arc07 | 12 | 99.5 | 3.4 | 0.117 | 3.4 tracks per robot on screen - fragmentation |
