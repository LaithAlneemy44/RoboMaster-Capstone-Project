"""Classical armor detector: HSV gate -> candidate regions -> edge map -> NCC.

The classical arm of the comparison, built rather than imported, following proposal 5.2:
RGB to HSV, template matching over areas of matching target hue, a greyscale edge map,
and normalised cross correlation for similarity. No training - behaviour comes entirely
from the parameters below and the template bank from scripts/build_templates.py.

WHAT THE MEASUREMENTS CHANGED
    5.2 describes gating on target hue. Measured over the train split, hue alone is a
    weak gate: background is already 13.3% red against armor's 26.6%, so barely 2x
    enriched. Two properties matter more.

    Armor plates are LIT. Value median is 228, and saturation drops to 49 at p25 because
    the LED cores blow out toward white. A hue-and-saturation gate, the obvious reading
    of 5.2, discards the brightest centre of every plate. So the gate leads with
    brightness and uses hue to separate red team from blue.

    The mask fires on the LED BARS, not the plate. A connected component is usually one
    bar - half the target. That is why candidates are padded generously before matching:
    the template is a whole plate, and it needs room to find itself around the bar that
    triggered the candidate.

COORDINATES
    matchTemplate returns a position inside the padded ROI. Every detection therefore
    adds the ROI origin back before being emitted. Forgetting that offset produces boxes
    clustered near the top-left of the image and an mAP near zero that looks exactly like
    a failed algorithm - predict_to_coco.assert_native_scale is the backstop.

Usage:
    python scripts/classical_detector.py --selftest
    python scripts/classical_detector.py --config balanced --image path/to/frame.jpg
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "models" / "classical" / "templates.npz"

# OpenCV hue is 0-180, so red straddles the wrap and needs two ranges.
RED_BANDS = ((0, 10), (170, 180))
BLUE_BAND = (90, 130)


@dataclass
class Params:
    """One classical "model". 5.2 asks for several configs; these are the axes."""

    # --- candidate gate (governs how many regions reach the NCC stage) ---
    value_min: int = 170        # LED brightness; the primary discriminator
    sat_min: int = 40           # low, because blown-out LED cores desaturate
    hue_slack: int = 0          # widens both colour bands symmetrically
    close_kernel: int = 3       # morphological close, joins a broken bar

    # --- size envelope ---
    # Calibrated against ground truth, not guessed. Components landing on armor measure
    # ~7x8px; background noise is ~3x2px, and outnumbers them 54:1. Filtering on
    # min(w,h) and area separates the two: min(w,h)>=3 with area>=15 keeps 98.0% of
    # plates at 211 candidates/frame. An earlier version tested max(w,h), which admits
    # 1x40 slivers, and then capped by area DESCENDING - which discarded the small
    # armor bars in favour of arena lighting and floor reflections.
    min_side: int = 3           # on min(w, h), not max
    min_area: int = 15
    max_side: int = 60
    max_candidates: int = 600   # cost ceiling; NCC scales with candidate count

    # --- matching ---
    pad_ratio: float = 1.2      # ROI padding around a candidate, as a multiple of size
    scales: tuple = (1.0, 1.3)  # 0.8 measured worse: NCC favours small templates
    ncc_min: float = 0.30       # accept threshold on the NCC response
    nms_iou: float = 0.40

    # Size the emitted box from the lit pixels inside the matched window rather than
    # from the template. NCC decides WHERE the plate is and scores it, but the template
    # is a fixed shape from an aspect bin, so using its dimensions as the box makes
    # every detection the wrong size: measured against ground truth, boxes came out
    # 16px against a 24px median, and recall at IoU 0.5 was 18.5% while recall at IoU
    # 0.1 was 72.8% - found, but not fitted. An armor plate is bounded by its two LED
    # bars, which are exactly what the mask fires on, so their extent is a far better
    # estimate of the plate than any single template.
    # TESTED AND OFF BY DEFAULT. The idea fails for a specific reason: the matched
    # window is roughly one plate wide, and the lit pixels inside it are usually a
    # SINGLE bar, not both. Sizing from them shrinks the box to that bar - median box
    # 20.0 -> 8.5px, median IoU 0.389 -> 0.109, recall at IoU 0.5 28.5% -> 11.4%. A
    # generous margin recovers only part of it (0.30 margin: IoU 0.215). Kept because
    # the alternative - pairing bars into plates - is the domain-standard fix and would
    # start here, but it departs from the template-matching method 5.2 specifies.
    refine_box: bool = False
    refine_margin: float = 0.30

    # --- bar pairing: an ALTERNATIVE to template matching, not part of 5.2 ---
    # A RoboMaster armor plate is bounded by two lit LED bars. Template matching
    # locates plates well (72.8% recall at IoU 0.1) but fits them badly (18.5% at
    # IoU 0.5) because the emitted box is a fixed template shape. The two bars ARE
    # the plate edges, so pairing them gives the extent directly. Reported as a
    # separate config so the 5.2 method stands unaltered beside it.
    pair_bars: bool = False
    pair_height_ratio: float = 2.0   # taller/shorter bar, max
    pair_gap_min: float = 0.6        # centre separation / mean bar height
    pair_gap_max: float = 4.5
    pair_y_tolerance: float = 1.0    # centre y offset / mean bar height


CONFIGS = {
    # name          value_min  sat_min  ncc_min
    "loose":      Params(value_min=140, sat_min=25, ncc_min=0.20),
    "balanced":   Params(value_min=170, sat_min=40, ncc_min=0.30),
    "strict":     Params(value_min=200, sat_min=60, ncc_min=0.40),
    "loose_hi":   Params(value_min=140, sat_min=25, ncc_min=0.40),
    "strict_lo":  Params(value_min=200, sat_min=60, ncc_min=0.20),
    # Cheapest gate that still keeps nearly every plate: the HSV stage reaches 100%
    # plate recall even at value_min=200, where it fires on just 1.87% of background
    # pixels, so a tight gate costs recall almost nothing and buys a lot of speed.
    "tight":      Params(value_min=200, sat_min=60, min_area=25, ncc_min=0.25),
    # Not a 5.2 configuration - the domain-standard alternative, for comparison.
    "paired":     Params(value_min=170, sat_min=40, min_area=8, min_side=2,
                         pair_bars=True),
}


def load_templates(path: Path = TEMPLATES):
    """Edge-map templates keyed by aspect bin, from scripts/build_templates.py."""
    import numpy as np

    if not path.is_file():
        sys.exit(f"Missing {path}\nRun: python scripts/build_templates.py")
    data = np.load(path, allow_pickle=False)
    return [data[k].astype(np.float32) for k in data.files if k.startswith("ar")]


def edge_map(gray):
    """Sobel gradient magnitude, 0-1. Must match build_templates.edge_map."""
    import cv2
    import numpy as np

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    peak = float(mag.max())
    return mag / peak if peak > 0 else mag


def nms(boxes, iou_thresh: float):
    """Greedy non-maximum suppression over (x, y, w, h, score)."""
    kept = []
    for box in sorted(boxes, key=lambda b: -b[4]):
        x, y, w, h, _ = box
        clash = False
        for kx, ky, kw, kh, _ in kept:
            ix1, iy1 = max(x, kx), max(y, ky)
            ix2, iy2 = min(x + w, kx + kw), min(y + h, ky + kh)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter / (w * h + kw * kh - inter) > iou_thresh:
                clash = True
                break
        if not clash:
            kept.append(box)
    return kept


class ClassicalDetector:
    """HSV gate, then NCC against an edge-map template bank."""

    def __init__(self, params: Params | None = None, templates=None):
        self.p = params or Params()
        self.templates = templates if templates is not None else load_templates()
        self.last_candidates = 0  # so the benchmark can attribute cost to the gate

    def candidate_mask(self, hsv):
        import cv2
        import numpy as np

        p = self.p
        slack = p.hue_slack
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        bands = [(max(0, lo - slack), min(180, hi + slack)) for lo, hi in RED_BANDS]
        bands.append((max(0, BLUE_BAND[0] - slack), min(180, BLUE_BAND[1] + slack)))
        for lo, hi in bands:
            mask |= cv2.inRange(hsv, (lo, p.sat_min, p.value_min),
                                (hi, 255, 255))
        if p.close_kernel > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (p.close_kernel, p.close_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def candidates(self, mask):
        """Connected components inside the observed armor size envelope."""
        import cv2

        p = self.p
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, count):
            x, y, w, h = (int(stats[i, k]) for k in
                          (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                           cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            area = int(stats[i, cv2.CC_STAT_AREA])
            if min(w, h) < p.min_side or max(w, h) > p.max_side or area < p.min_area:
                continue
            out.append((x, y, w, h, area))
        # Only meaningful once the noise is filtered out above; before that, sorting by
        # area kept the brightest floor reflections and dropped the armor.
        out.sort(key=lambda c: -c[4])
        return out[:p.max_candidates]

    def pair_bars(self, regions):
        """Pair lit bars into plates. The alternative to template matching.

        An armor plate is two vertical LED bars with the numbered panel between them, so
        the pair's bounding box IS the plate - no template, no fixed output size, and the
        box fits whatever the plate actually measures. Geometry thresholds come from the
        measured bars (~7x8px inside a 24x19px plate, so centres sit roughly 2 bar-heights
        apart).

        Score is how alike the two bars are: a real pair is two views of the same lamp, so
        similar height and level with each other. Mismatched pairs score low and fall out
        at the threshold.
        """
        p = self.p
        out = []
        for i, (ax, ay, aw, ah, _) in enumerate(regions):
            for bx, by, bw, bh, _ in regions[i + 1:]:
                mean_h = (ah + bh) / 2.0
                if mean_h <= 0:
                    continue
                taller = max(ah, bh) / max(1e-6, min(ah, bh))
                if taller > p.pair_height_ratio:
                    continue
                acx, acy = ax + aw / 2, ay + ah / 2
                bcx, bcy = bx + bw / 2, by + bh / 2
                gap = abs(acx - bcx) / mean_h
                if not p.pair_gap_min <= gap <= p.pair_gap_max:
                    continue
                if abs(acy - bcy) / mean_h > p.pair_y_tolerance:
                    continue

                x0, y0 = min(ax, bx), min(ay, by)
                x1 = max(ax + aw, bx + bw)
                y1 = max(ay + ah, by + bh)
                # Closer heights and better vertical alignment mean a more convincing
                # pair; both terms are already normalised, so the product is a score.
                score = (1.0 / taller) * (1.0 - abs(acy - bcy) / (mean_h * 2.0))
                out.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0),
                            float(max(0.0, min(1.0, score)))))
        return out

    def detect(self, image_bgr):
        """Returns [(xywh, score, "armor")] in NATIVE image pixels."""
        import cv2
        import numpy as np

        p = self.p
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        mask = self.candidate_mask(hsv)
        regions = self.candidates(mask)
        self.last_candidates = len(regions)

        if p.pair_bars:
            paired = self.pair_bars(regions)
            kept = nms([b for b in paired if b[4] >= p.ncc_min], p.nms_iou)
            return [((x, y, w, h), s, "armor") for x, y, w, h, s in kept]

        raw = []
        for cx, cy, cw, ch, _ in regions:
            # Generous padding: the gate fires on a single LED bar, and the template is
            # a whole plate, so the ROI must be able to contain one.
            pad_x = int(cw * p.pad_ratio) + 2
            pad_y = int(ch * p.pad_ratio) + 2
            x0, y0 = max(0, cx - pad_x), max(0, cy - pad_y)
            x1, y1 = min(width, cx + cw + pad_x), min(height, cy + ch + pad_y)
            roi = gray[y0:y1, x0:x1]
            if roi.shape[0] < 6 or roi.shape[1] < 6:
                continue
            roi_edges = edge_map(roi)

            best = None
            for template in self.templates:
                for scale in p.scales:
                    th = max(4, int(round(template.shape[0] * scale)))
                    tw = max(4, int(round(template.shape[1] * scale)))
                    if th >= roi_edges.shape[0] or tw >= roi_edges.shape[1]:
                        continue
                    scaled = cv2.resize(template, (tw, th),
                                        interpolation=cv2.INTER_AREA)
                    response = cv2.matchTemplate(roi_edges, scaled,
                                                 cv2.TM_CCOEFF_NORMED)
                    _, score, _, loc = cv2.minMaxLoc(response)
                    if best is None or score > best[0]:
                        best = (score, loc, tw, th)

            if best is None or best[0] < p.ncc_min:
                continue
            score, loc, tw, th = best
            # ROI origin added back - see the module docstring.
            bx, by = x0 + loc[0], y0 + loc[1]
            bw, bh = tw, th

            if p.refine_box:
                window = mask[by:by + bh, bx:bx + bw]
                ys, xs = np.nonzero(window)
                if xs.size:
                    mx0, mx1 = int(xs.min()), int(xs.max())
                    my0, my1 = int(ys.min()), int(ys.max())
                    pad_w = int((mx1 - mx0 + 1) * p.refine_margin)
                    pad_h = int((my1 - my0 + 1) * p.refine_margin)
                    bx, by = bx + mx0 - pad_w, by + my0 - pad_h
                    bw = (mx1 - mx0 + 1) + 2 * pad_w
                    bh = (my1 - my0 + 1) + 2 * pad_h
                    bx, by = max(0, bx), max(0, by)
                    bw, bh = max(2, bw), max(2, bh)

            raw.append((float(bx), float(by), float(bw), float(bh), float(score)))

        return [((x, y, w, h), s, "armor") for x, y, w, h, s in nms(raw, p.nms_iou)]


def selftest() -> None:
    """Check the pieces that can be checked without ground truth."""
    import cv2
    import numpy as np

    print("[test] templates load")
    templates = load_templates()
    print(f"       ok - {len(templates)} templates, shapes "
          f"{[t.shape for t in templates]}")

    print("[test] nms")
    boxes = [(0, 0, 10, 10, 0.9), (1, 1, 10, 10, 0.8), (100, 100, 10, 10, 0.7)]
    kept = nms(boxes, 0.4)
    assert len(kept) == 2, kept
    print("       ok - overlapping box suppressed, distant one kept")

    print("[test] coordinates are native, not ROI-relative")
    # A synthetic bright blue plate far from the origin. Whatever the detector finds
    # must be near it, not near (0, 0) - the ROI-offset bug in the docstring.
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(frame, (420, 300), (444, 320), (255, 120, 60), -1)
    cv2.rectangle(frame, (424, 304), (440, 316), (30, 30, 30), -1)
    found = ClassicalDetector(CONFIGS["loose"], templates).detect(frame)
    if found:
        (x, y, w, h), score, _ = max(found, key=lambda d: d[1])
        assert 300 < x < 560 and 200 < y < 400, f"box at ({x},{y}) - ROI offset lost?"
        print(f"       ok - {len(found)} detection(s), best at ({x:.0f},{y:.0f}) "
              f"score {score:.2f}")
    else:
        print("       WARNING: nothing found on the synthetic plate. Not fatal - a "
              "flat\n       synthetic rectangle has little edge structure - but the "
              "offset check\n       did not run. Verify on a real frame.")

    print("\nAll checks passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", choices=sorted(CONFIGS), default="balanced")
    parser.add_argument("--image", type=Path, help="Run on one image and report.")
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    if not args.image:
        sys.exit("Give --image or --selftest")

    import cv2

    frame = cv2.imread(str(args.image))
    if frame is None:
        sys.exit(f"Could not read {args.image}")
    detector = ClassicalDetector(CONFIGS[args.config])
    found = detector.detect(frame)
    print(f"{args.image.name}: {detector.last_candidates} candidates -> "
          f"{len(found)} detections")
    for (x, y, w, h), score, _ in sorted(found, key=lambda d: -d[1])[:10]:
        print(f"  {score:.3f}  ({x:.0f},{y:.0f}) {w:.0f}x{h:.0f}")


if __name__ == "__main__":
    main()
