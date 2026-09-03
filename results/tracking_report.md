# Tracking — CPU cost and accuracy

## CPU cost of each perception model (the headline — needs no labels)

One row per detector+tracker pairing, averaged over sequences. `RT` marks >= 30 FPS. `track %` is the tracker's share of the pipeline: a tracker that is cheap beside its detector is a completely different deployment proposition from one that dominates the frame budget.

| cores | detector | tracker | detect ms | track ms | total ms | FPS | RT | track % | tracks/frame | CPU% of cap | RAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `strict` | `classical` | 30.3 | 0.6 | 30.9 | 33.9 | OK | 2% | 12.5 | 98 | 534 |
| 1 | `strict` | `sort` | 30.7 | 1.1 | 31.8 | 33.0 | OK | 3% | 12.5 | 97 | 570 |
| 1 | `ssd_small_960_anchor` | `classical` | 143.0 | 0.1 | 143.1 | 7.0 |  | 0% | 4.7 | 98 | 745 |
| 1 | `ssd_small_960_anchor` | `sort` | 145.1 | 0.4 | 145.4 | 6.9 |  | 0% | 4.7 | 98 | 779 |
| 1 | `fast_960` | `sort` | 158.2 | 0.4 | 158.6 | 6.3 |  | 0% | 5.4 | 96 | 788 |
| 1 | `fast_960` | `classical` | 162.8 | 0.1 | 163.0 | 6.1 |  | 0% | 5.4 | 96 | 754 |
| 1 | `ssd_small_960_anchor` | `vit` | 142.6 | 87.6 | 230.2 | 4.4 |  | 37% | 4.7 | 97 | 774 |
| 1 | `fast_960` | `vit` | 157.7 | 101.6 | 259.3 | 3.9 |  | 39% | 5.4 | 98 | 790 |
| 1 | `strict` | `vit` | 31.1 | 324.2 | 355.2 | 3.5 |  | 90% | 11.6 | 98 | 727 |
| 1 | `yolo_960` | `sort` | 402.5 | 0.4 | 402.8 | 2.5 |  | 0% | 5.1 | 98 | 859 |
| 1 | `yolo_960` | `classical` | 404.0 | 0.1 | 404.2 | 2.5 |  | 0% | 5.1 | 97 | 830 |
| 1 | `yolo_960` | `vit` | 402.2 | 94.0 | 496.2 | 2.0 |  | 19% | 5.1 | 97 | 854 |
| 1 | `ssd_small_960_anchor` | `goturn` | 154.4 | 3685.0 | 3839.4 | 0.3 |  | 96% | 4.4 | 98 | 5933 |
| 1 | `fast_960` | `goturn` | 163.5 | 4308.7 | 4472.2 | 0.2 |  | 96% | 5.0 | 98 | 7211 |
| 1 | `yolo_960` | `goturn` | 405.7 | 4309.4 | 4715.1 | 0.2 |  | 91% | 5.1 | 98 | 7292 |
| 1 | `strict` | `goturn` | 140.1 | 93322.1 | 93462.1 | 0.0 |  | 100% | 13.3 | 69 | 17476 |
| 2 | `yolo_960` | `sort` | 256.1 | 0.4 | 256.5 | 3.9 |  | 0% | 5.1 | 96 | 863 |
| 2 | `yolo_960` | `classical` | 261.5 | 0.1 | 261.6 | 3.8 |  | 0% | 5.1 | 96 | 833 |
| 2 | `yolo_960` | `vit` | 243.3 | 95.7 | 338.9 | 3.0 |  | 28% | 5.1 | 96 | 864 |
| 2 | `yolo_960` | `goturn` | 248.4 | 4192.4 | 4440.7 | 0.2 |  | 94% | 5.1 | 55 | 7340 |
| 4 | `yolo_960` | `sort` | 152.0 | 0.4 | 152.3 | 6.6 |  | 0% | 5.1 | 96 | 870 |
| 4 | `yolo_960` | `classical` | 187.6 | 0.1 | 187.7 | 5.4 |  | 0% | 5.1 | 97 | 833 |
| 4 | `yolo_960` | `vit` | 181.8 | 96.3 | 278.1 | 3.6 |  | 34% | 5.1 | 95 | 870 |
| 4 | `yolo_960` | `goturn` | 170.0 | 4179.0 | 4349.0 | 0.2 |  | 96% | 5.1 | 31 | 7347 |
| 6 | `strict` | `classical` | 22.3 | 0.6 | 22.9 | 47.9 | OK | 2% | 12.5 | 28 | 541 |
| 6 | `strict` | `sort` | 23.1 | 1.1 | 24.3 | 45.7 | OK | 4% | 12.5 | 28 | 575 |
| 6 | `fast_960` | `sort` | 66.9 | 0.4 | 67.3 | 14.9 |  | 1% | 5.4 | 98 | 786 |
| 6 | `fast_960` | `classical` | 70.9 | 0.1 | 71.0 | 14.2 |  | 0% | 5.4 | 98 | 755 |
| 6 | `ssd_small_960_anchor` | `classical` | 91.0 | 0.1 | 91.2 | 11.0 |  | 0% | 4.7 | 100 | 758 |
| 6 | `ssd_small_960_anchor` | `sort` | 93.0 | 0.4 | 93.4 | 10.7 |  | 0% | 4.7 | 99 | 792 |
| 6 | `yolo_960` | `classical` | 143.5 | 0.1 | 143.7 | 7.0 |  | 0% | 5.1 | 98 | 838 |
| 6 | `yolo_960` | `sort` | 148.6 | 0.4 | 149.0 | 6.7 |  | 0% | 5.1 | 98 | 869 |
| 6 | `fast_960` | `vit` | 67.0 | 108.4 | 175.4 | 5.7 |  | 61% | 5.4 | 97 | 796 |
| 6 | `ssd_small_960_anchor` | `vit` | 95.3 | 83.0 | 178.4 | 5.7 |  | 45% | 4.7 | 98 | 798 |
| 6 | `strict` | `vit` | 22.5 | 278.0 | 300.5 | 4.2 |  | 92% | 11.6 | 34 | 735 |
| 6 | `yolo_960` | `vit` | 155.7 | 101.8 | 257.5 | 3.9 |  | 39% | 5.1 | 97 | 878 |
| 6 | `ssd_small_960_anchor` | `goturn` | 102.0 | 3398.2 | 3500.2 | 0.3 |  | 97% | 4.4 | 26 | 5943 |
| 6 | `yolo_960` | `goturn` | 153.2 | 4178.4 | 4331.6 | 0.2 |  | 96% | 5.1 | 23 | 7347 |
| 6 | `fast_960` | `goturn` | 67.0 | 4215.6 | 4282.7 | 0.2 |  | 98% | 5.0 | 21 | 7272 |
| 6 | `strict` | `goturn` | 131.3 | 72267.8 | 72399.1 | 0.0 |  | 100% | 13.3 | 18 | 16249 |

Fastest pairing at each core level:

- **1 core(s)** — `strict` + `classical`, 30.9 ms (33.9 FPS) — best of 16 pairings measured at this level
- **2 core(s)** — `yolo_960` + `sort`, 256.5 ms (3.9 FPS) — best of 4 pairings measured at this level
- **4 core(s)** — `yolo_960` + `sort`, 152.3 ms (6.6 FPS) — best of 4 pairings measured at this level
- **6 core(s)** — `strict` + `classical`, 22.9 ms (47.9 FPS) — best of 16 pairings measured at this level

## Accuracy (secondary — see the caveat below)

**Read IDF1 and ID switches, not MOTA.** Every tracker in a given row group is fed by the same detector, and MOTA is dominated by that detector's misses and false positives, which are therefore identical across trackers; only the ID-switch term varies, and it is small next to the others. On arc01 `classical`, `sort` and `vit` return MOTA 0.570 and IDF1 0.651 alike — the trackers genuinely agreed on that clip's identity assignment. Where they diverge, IDF1 separates them: GOTURN scores 0.493 against 0.651 on the same frames, with twice the ID switches.

| tracker | sequence | MOTA | 95% CI | IDF1 | ID switches | MT / ML |
|---|---|---|---|---|---|---|
| `classical_det` | arc01 | 0.397 | [0.357, 0.434] | 0.641 | 8 | 7 / 0 |
| `classical_det` | arc02 | 0.767 | [0.743, 0.791] | 0.856 | 5 | 7 / 1 |
| `classical_det` | arc03 | 0.332 | [0.310, 0.355] | 0.660 | 6 | 9 / 0 |
| `classical_det` | arc04 | 0.196 | [0.164, 0.227] | 0.466 | 13 | 2 / 1 |
| `classical_det` | arc05 | 0.628 | [0.604, 0.648] | 0.750 | 7 | 10 / 3 |
| `classical_det` | arc06 | 0.306 | [0.267, 0.347] | 0.475 | 24 | 5 / 6 |
| `classical_det` | arc07 | 0.508 | [0.453, 0.553] | 0.560 | 10 | 5 / 0 |
| `classical_det150` | arc01 | 0.570 | [0.542, 0.597] | 0.651 | 7 | 4 / 1 |
| `classical_det150` | arc02 | 0.848 | [0.832, 0.864] | 0.907 | 1 | 6 / 0 |
| `classical_det150` | arc05 | 0.673 | [0.649, 0.697] | 0.782 | 4 | 10 / 3 |
| `goturn_det150` | arc01 | 0.557 | [0.529, 0.586] | 0.493 | 14 | 4 / 1 |
| `goturn_det150` | arc02 | 0.838 | [0.821, 0.857] | 0.653 | 7 | 6 / 0 |
| `goturn_det150` | arc05 | 0.658 | [0.635, 0.682] | 0.706 | 15 | 9 / 3 |
| `sort_det` | arc01 | 0.398 | [0.357, 0.435] | 0.641 | 8 | 7 / 0 |
| `sort_det` | arc02 | 0.769 | [0.747, 0.794] | 0.857 | 6 | 7 / 1 |
| `sort_det` | arc03 | 0.330 | [0.308, 0.352] | 0.656 | 8 | 9 / 0 |
| `sort_det` | arc04 | 0.197 | [0.165, 0.228] | 0.480 | 17 | 2 / 1 |
| `sort_det` | arc05 | 0.639 | [0.617, 0.660] | 0.751 | 9 | 9 / 3 |
| `sort_det` | arc06 | 0.310 | [0.272, 0.353] | 0.491 | 25 | 5 / 6 |
| `sort_det` | arc07 | 0.522 | [0.468, 0.569] | 0.594 | 10 | 6 / 0 |
| `sort_det150` | arc01 | 0.570 | [0.542, 0.597] | 0.651 | 7 | 4 / 1 |
| `sort_det150` | arc02 | 0.846 | [0.829, 0.862] | 0.905 | 2 | 6 / 0 |
| `sort_det150` | arc05 | 0.686 | [0.665, 0.707] | 0.782 | 6 | 9 / 3 |
| `vit_det150` | arc01 | 0.570 | [0.542, 0.597] | 0.651 | 7 | 4 / 1 |
| `vit_det150` | arc02 | 0.843 | [0.827, 0.860] | 0.745 | 5 | 6 / 0 |
| `vit_det150` | arc05 | 0.661 | [0.637, 0.685] | 0.677 | 12 | 9 / 3 |

**Caveat.** Ground truth here was generated automatically (`scripts/auto_label.py`), because hand-labelling was not affordable within the project. Labels derived from motion association share their assumptions with SORT and the classical tracker and not with the appearance-based GOTURN and VitTrack, so this table is biased toward the former. The CPU table above needs no labels and is unaffected.

Evaluation is detector-fed, not ground-truth-fed. Feeding trackers the reference boxes proved degenerate, and this was measured rather than assumed: on arc01 the classical tracker and SORT both scored MOTA 0.991651 — identical to six decimal places, which is not two trackers agreeing but one answer arrived at twice. Handed boxes that were themselves produced by association, a Kalman tracker returns the labeller's own tracks. Detector-fed asks a real question instead — how much an online tracker loses against an offline reference that saw the whole clip.

Rows ending `_det150` are the four-way head-to-head: all four trackers on the same 3 clips x 150 frames, fed by `yolo_960`. Rows ending `_det` are the broader 7-clip sample, which only `classical` and `sort` are cheap enough to run in full — GOTURN alone would need ~17 hours for it. Compare within a suffix, not across.

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
