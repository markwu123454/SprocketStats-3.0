#!/usr/bin/env python3
"""
Step 2 -- Find every region of the broadcast frame that contains a CHANGING
NUMBER. Nothing else: no roles, no alliance colours, no icons, no OCR, no
expectations about what any region should read.

This REPLACES the old 02_ocr_score.py, which was hallucinated (written
speculatively, never run -- see broadcast/data/archive/ for the pre-rewrite
snapshot, kept only because broadcast/ is untracked by git).

Why this file no longer assigns meaning
---------------------------------------
The previous version of THIS file did assign meaning: it classified each
region as blue_score / red_score / clock / badge using the chip colour plus
an OCR read of the pre-match frame that had to equal a known starting value
(0 for a score, 0:AUTO_DURATION for the clock). Measured across 9 matches,
that layer produced 9 wrong boxes out of ~44, and every one of its failure
modes was a semantic assumption meeting a broadcast that did not share it:

* "the badge is a bare integer" -- 2026's badge reads "0 / 100", so the
  merged region OCR'd as '07100', failed the ==0 test, and match4 and
  match10 got NO blue badge at all.
* "numerically equal to 0 means it started at 0" -- match4's red badge
  confirmed on the STATIC denominator "100" (OCR'd '000', int 0), so the
  box landed on the wrong half of the field.
* "blue is on the left" -- match6 has red on the left.
* "the pre-match clock reads 0:20" -- 2025 reads 0:00.
* match8's overlay is a different style altogether, and the file still
  emitted confirmed:true on a blue chip containing no digits.

None of that is fixable by better thresholds, because the information needed
to decide arrives LATER: what a region means is far easier to see from how
its value behaves across a whole match than from one frame before it starts.
So identity moved to 06_identify.py, which reads 05's timelines, and this
file was cut back to the part it can actually measure.

The split, concretely: this file may use pixel colour as a SIMILARITY signal
(two pockets sit on the same chip, so they are one field) but never as a
NAME (this chip is blue, therefore blue alliance) -- naming is a
season-specific convention and belongs in 06. Colour-as-similarity stays
entirely internal to the merge step (merge_pockets/background_stats) and is
not exported: 06_identify.py has its own frame access now and samples
colour itself, from however many frames it needs, when it decides what a
region means. Dropping the naming also drops easyocr/torch from this file
-- the whole pipeline is now OCR-free apart from 03_calibrate.py's template
harvesting.

Algorithm
---------
1. Sample frames spread across the video and compute a per-pixel temporal
   activity range. Static pixels land near 0; anything that changes lands
   high. Uses a PERCENTILE range (hi_pctl - lo_pctl), not literal max-min:
   found the hard way on match2, a produced VOD bookended by a title card
   and celebration b-roll, where max-min is wrecked by a single outlier
   frame per pixel and reported 100% of the frame as dynamic. A percentile
   range tolerates a minority of contaminated samples.
2. Connected components of the DYNAMIC mask, keeping "pockets": components
   that do not touch the frame border (real camera motion almost always
   reaches an edge) and fall in a plausible digit-sized area range.
3. Merge pockets into regions. A multi-digit number does not reliably
   survive step 2 as one component -- whether the static gap between glyphs
   stays below STATIC_THRESHOLD is font- and rendering-specific (match1's
   2-digit score merged on its own; match2's 3-digit score split into three).
   Two pockets merge when they are row-aligned, close horizontally, and sit
   on a background of the same colour.
4. Emit every surviving region with what was measured about it: box,
   n_pockets it was merged from, activity. No colour, no interpretation --
   colour was only ever needed to decide the merge itself (step 3).

MERGE_MAX_GAP_FRAC: the constant that cost two matches their clock
--------------------------------------------------------------------
The merge gap used to be MERGE_MAX_GAP_PX = 15, an absolute pixel count,
applied to an overlay whose rendering scale varies by 2x across broadcasts.
Measured inter-glyph gaps inside the clock chip:

    glyph height 31-32px (match1/2/4/6)  ->  gaps 6-11px
    glyph height 63-65px (match7/9)      ->  gaps 13-20px

The gap is a constant 0.31 x glyph height at every scale, and a fixed 15px
threshold falls between the two. So minutes+colon merged with seconds on the
small overlay and SPLIT on the large one, the orphaned minutes digit lost
role assignment to the wider seconds pocket, and match7/match9's clock boxes
covered "53" out of "1:53" -- permanently, with no way for any later stage to
recover the minutes digit. Expressing the gap as a fraction of pocket height
is the fix. The safe band is wide: real inter-glyph gaps sit at 0.31x, and
the nearest neighbouring field measured 1.9x, so the value below is roughly
the geometric middle of a range spanning a factor of six.

Output
------
data/<match>_regions.json:
{
  "match": "match1", "img_w":.., "img_h":.., "fps":..,
  "regions": [{"id": "r00", "box": [x0,y0,x1,y1], "area_frac":..,
               "n_pockets":.., "activity":..}, ...],
  "pockets": [ ... pre-merge, for --viz debugging ... ]
}
Regions are ordered left-to-right, top-to-bottom, and `id` is positional
only -- it carries no meaning and is not stable across reruns if the
detection changes. 05_extract.py reads every region; 06_identify.py decides
which ones matter.

Known limitations / unvalidated constants
------------------------------------------
- MIN/MAX_POCKET_AREA_FRAC, STATIC_THRESHOLD, ACTIVITY_LOW/HIGH_PCTL and
  BG_MATCH_DIST are tuned from real footage but not swept. Treat each as
  independently replaceable: if a region comes up missing or wrong, check
  which stage (activity, pocket geometry, or merge) actually failed first.
- The percentile trim handles a MINORITY of contaminated frames, not an
  arbitrary amount. A video where non-match content dominates the sampled
  window would still break it.
- Regions whose value never changes are invisible to this file by
  construction -- activity is the only detector. A static denominator ("/
  100") is only captured because the numerator beside it re-flows the whole
  string when it gains a digit, dragging the denominator's pixels with it.
  A truly static number sharing a chip with nothing dynamic will be missed.
- This file no longer rejects regions that are not numbers at all (match6
  has a pocket on a field-wall sign reading "REBUILT"). That is deliberate:
  05_extract.py's triage answers "does this decode as digits" far better
  than any geometric test here could, and it does so by measurement.

Usage
-----
  python pipeline/02_detect_overlay.py --video match1.mp4 --save --viz
"""

import argparse, json, pathlib, sys

import cv2
import numpy as np

ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

N_SAMPLE_FRAMES = 240

# Starts much earlier than homography/00's 0.25 default: that range exists
# there to dodge intro/outro CG for a different problem. Here the pre-match
# state is part of what should be detected, and on match1 the overlay is on
# screen by ~1% into the video.
SAMPLE_START_FRAC = 0.01
SAMPLE_END_FRAC = 0.85

# ---------------------------------------------------------------------------
# Activity / pocket detection
# ---------------------------------------------------------------------------

# Reused verbatim from homography/pipeline/00_split_views.py's
# STATIC_THRESHOLD -- same signal and same source of truth for what "static"
# means on compressed H.264 broadcast footage (static ~2-5, camera ~30+).
STATIC_THRESHOLD = 12

# Percentile pair for the activity range instead of literal max-min. Trims
# the most extreme 8% of samples per side per pixel, so a minority of
# contaminated frames is treated as outliers rather than dominating.
ACTIVITY_LOW_PCTL  = 8
ACTIVITY_HIGH_PCTL = 92

# Measured areas at 1920x1080 (area / frame_area). A merged 2-3 digit score
# is ~0.0012-0.0017; the small per-alliance edge numbers ~0.0004-0.0005. MIN
# exists only to reject near-noise components: it was lowered to 0.0001 after
# a single "1" (the clock's minutes place, sparser than a "0") measured
# 0.00036 and was being excluded, and again after one of match3's edge
# numbers measured 0.000156 -- still 6x above the actual per-frame noise floor
# in that region (stray components there measure 0.000001-0.000025, confirmed
# from connectedComponentsWithStats output). MAX is a generous ceiling, not a
# tight fit.
MIN_POCKET_AREA_FRAC = 0.0001
MAX_POCKET_AREA_FRAC = 0.02

# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

# See the module docstring section on this constant -- it is the one that
# cost match7 and match9 their minutes digit when it was absolute.
MERGE_MAX_GAP_FRAC = 0.7

# Vertical misalignment still counting as "the same text baseline". Observed
# y0 jitter between adjacent real digits is 0-1px at every scale seen, so
# this is loose; proportional with a floor so it does not become tight on a
# large overlay or meaningless on a small one.
MERGE_MAX_Y_OFFSET_FRAC = 0.20
MERGE_MAX_Y_OFFSET_MIN_PX = 3

# How far outside a pocket to sample the background chip colour. The pocket's
# own pixels are dominated by the changing glyph, so the chip has to be
# sampled around it.
BG_RING_MARGIN = 6

# Max Euclidean BGR distance between two pockets' background colours for them
# to count as sitting on the same chip. This is deliberately a SIMILARITY
# test with no colour names attached (see the module docstring). Measured
# separations are enormous relative to this: two digits of the same number
# differ by ~0-5, while the blue score chip and the white clock chip beside
# it differ by 200+. Any value in roughly [30, 120] behaves identically on
# the current footage.
BG_MATCH_DIST = 50


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_frames(video_path: pathlib.Path, n: int, start_frac: float, end_frac: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[error] cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lo = max(0, int(total * start_frac))
    hi = min(total - 1, int(total * end_frac))
    step = max(1, (hi - lo) // max(1, n - 1))
    indices = list(range(lo, hi + 1, step))[:n]

    grays, bgrs = [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        bgrs.append(frame)
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return grays, bgrs, fps, w, h


def accumulate_range(frames_gray: list) -> np.ndarray:
    """Per-pixel temporal activity: (high percentile - low percentile) across
    sampled frames. See ACTIVITY_LOW_PCTL/HIGH_PCTL and the module docstring
    for why this is not max-min. Stacks all frames as uint8 (~430MB for 240
    frames at 1080p) -- a bounded one-time cost; the incremental running
    min/max trick homography/00 uses is unavailable once outlier-robustness
    is the point."""
    stack = np.stack(frames_gray)
    lo, hi = np.percentile(stack, [ACTIVITY_LOW_PCTL, ACTIVITY_HIGH_PCTL], axis=0)
    return (hi - lo).astype(np.float32)


# ---------------------------------------------------------------------------
# Pockets
# ---------------------------------------------------------------------------

def find_pockets(range_img: np.ndarray, static_threshold: float,
                 min_area_frac: float, max_area_frac: float) -> list:
    """Connected components of the DYNAMIC mask, keeping only components that
    (a) do not touch the frame border -- a real camera view's activity almost
    always reaches an edge, which rejects it without needing to isolate the
    overlay panel first -- and (b) fall inside a plausible area range."""
    h, w = range_img.shape
    total = h * w
    dynamic = (range_img > static_threshold).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(dynamic, connectivity=8)

    pockets = []
    for label in range(1, n_labels):
        x, y, bw, bh, area = stats[label]
        if x <= 0 or y <= 0 or (x + bw) >= w or (y + bh) >= h:
            continue
        frac = area / total
        if not (min_area_frac <= frac <= max_area_frac):
            continue
        pockets.append({"box": [int(x), int(y), int(x + bw), int(y + bh)],
                        "area": int(area), "frac": round(float(frac), 6)})
    return pockets


# ---------------------------------------------------------------------------
# Background colour -- a similarity signal, never a name
# ---------------------------------------------------------------------------

def background_stats(frame_bgr: np.ndarray, box: list, margin: int = BG_RING_MARGIN):
    """Median BGR and per-channel spread of the ring just outside `box`.
    Returns (median_bgr, std) or (None, None). Used only internally by
    merge_pockets, as a SIMILARITY signal for deciding which pockets belong
    to the same chip -- not exported. 06_identify.py now has its own frame
    access and samples colour itself when it needs to decide what a region
    means."""
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = box
    ox0, oy0 = max(0, x0 - margin), max(0, y0 - margin)
    ox1, oy1 = min(w, x1 + margin), min(h, y1 + margin)
    outer = frame_bgr[oy0:oy1, ox0:ox1]
    if outer.size == 0:
        return None, None
    ring_mask = np.ones(outer.shape[:2], dtype=bool)
    ring_mask[y0 - oy0:y1 - oy0, x0 - ox0:x1 - ox0] = False
    ring = outer[ring_mask]
    if ring.size == 0:
        return None, None
    return np.median(ring, axis=0), float(np.mean(np.std(ring, axis=0)))


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_pockets(pockets: list, ref_bgr: np.ndarray, frame_area: int) -> list:
    """Merge row-aligned, horizontally-close pockets that sit on the same
    background into one region per number field.

    Both the gap and the row tolerance scale with pocket height -- see the
    module docstring on MERGE_MAX_GAP_FRAC for the measurement that forced
    this. Background similarity is the second, independent condition: it is
    what keeps a score chip from merging with the clock chip beside it when
    the two happen to fall within the gap."""
    for p in pockets:
        med, _ = background_stats(ref_bgr, p["box"])
        p["_bg"] = med

    order = sorted(range(len(pockets)), key=lambda i: pockets[i]["box"][0])
    used = [False] * len(pockets)
    regions = []
    for oi in order:
        if used[oi]:
            continue
        members = [oi]
        used[oi] = True
        x0, y0, x1, y1 = pockets[oi]["box"]
        grew = True
        while grew:
            grew = False
            for oj in order:
                if used[oj]:
                    continue
                qx0, qy0, qx1, qy1 = pockets[oj]["box"]
                h_ref = max(y1 - y0, qy1 - qy0, 1)
                y_tol = max(MERGE_MAX_Y_OFFSET_MIN_PX, MERGE_MAX_Y_OFFSET_FRAC * h_ref)
                gap_tol = MERGE_MAX_GAP_FRAC * h_ref
                same_row = abs(qy0 - y0) <= y_tol and abs(qy1 - y1) <= y_tol
                close_x = (qx0 - x1) <= gap_tol and (x0 - qx1) <= gap_tol
                a, b = pockets[oi].get("_bg"), pockets[oj].get("_bg")
                same_bg = a is not None and b is not None and float(np.linalg.norm(a - b)) <= BG_MATCH_DIST
                if same_row and close_x and same_bg:
                    x0, y0, x1, y1 = min(x0, qx0), min(y0, qy0), max(x1, qx1), max(y1, qy1)
                    members.append(oj)
                    used[oj] = True
                    grew = True
        area = (x1 - x0) * (y1 - y0)
        regions.append({
            "box": [int(x0), int(y0), int(x1), int(y1)],
            "area_frac": round(area / frame_area, 6),
            "n_pockets": len(members),
            "activity": round(float(np.mean([pockets[i]["frac"] for i in members])), 6),
        })
    regions.sort(key=lambda r: (r["box"][1] // 20, r["box"][0]))
    for i, r in enumerate(regions):
        r["id"] = f"r{i:02d}"
    return regions


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(ref_bgr: np.ndarray, range_img: np.ndarray,
              pockets: list, regions: list) -> np.ndarray:
    h, w = ref_bgr.shape[:2]
    vis = ref_bgr.copy()
    for p in pockets:
        x0, y0, x1, y1 = p["box"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (140, 140, 140), 1)
    for r in regions:
        x0, y0, x1, y1 = r["box"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 230, 0), 2)
        cv2.putText(vis, r["id"], (x0, max(12, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 0), 1, cv2.LINE_AA)
    scaled = np.clip(range_img / max(float(range_img.max()), 1.0) * 255, 0, 255).astype(np.uint8)
    range_bgr = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    scale = 0.6
    return np.hstack([cv2.resize(vis, (int(w * scale), int(h * scale))),
                      cv2.resize(range_bgr, (int(w * scale), int(h * scale)))])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def detect(video_path: pathlib.Path) -> dict:
    grays, bgrs, fps, w, h = sample_frames(video_path, N_SAMPLE_FRAMES,
                                           SAMPLE_START_FRAC, SAMPLE_END_FRAC)
    if not grays:
        sys.exit(f"[error] no frames sampled from {video_path}")
    range_img = accumulate_range(grays)
    pockets = find_pockets(range_img, STATIC_THRESHOLD,
                           MIN_POCKET_AREA_FRAC, MAX_POCKET_AREA_FRAC)
    ref = bgrs[len(bgrs) // 2]
    regions = merge_pockets(pockets, ref, w * h)
    print(f"[detect] {video_path.stem}: {len(pockets)} pockets -> {len(regions)} regions", file=sys.stderr)
    return {"match": video_path.stem, "img_w": w, "img_h": h, "fps": fps,
            "regions": regions,
            "pockets": [{k: v for k, v in p.items() if not k.startswith("_")} for p in pockets]}, ref, range_img, pockets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--viz", action="store_true")
    args = ap.parse_args()

    video_path = pathlib.Path(args.video)
    if not video_path.exists():
        video_path = ROOT.parent / args.video
    if not video_path.exists():
        sys.exit(f"[error] video not found: {args.video}")

    result, ref, range_img, pockets = detect(video_path)
    for r in result["regions"]:
        x0, y0, x1, y1 = r["box"]
        print(f"  {r['id']}  {x1-x0:>4}x{y1-y0:<4} @({x0},{y0})  "
              f"pockets={r['n_pockets']}", file=sys.stderr)

    if args.viz:
        vis = visualize(ref, range_img, pockets, result["regions"])
        out = DATA_DIR / f"{result['match']}_regions.jpg"
        cv2.imwrite(str(out), vis)
        print(f"[viz] -> {out}", file=sys.stderr)

    if args.save:
        out = DATA_DIR / f"{result['match']}_regions.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"[save] -> {out}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
