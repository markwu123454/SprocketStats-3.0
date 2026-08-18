#!/usr/bin/env python3
"""
Step 2 -- Detect the CG scoreboard overlay's live numeric regions (blue
score, red score, match clock, and each alliance's small edge badge)
directly from a broadcast video. No manual --calibrate step: the layout is
found per match from the video itself.

This REPLACES the old 02_ocr_score.py, which is hallucinated (written
speculatively, never run/validated -- see broadcast/data/archive/ for the
pre-rewrite snapshot, kept only because broadcast/ is untracked by git and
there's no other safety net). Nothing in this file's design carries over
from it.

Algorithm
---------
The scorebug is a single global CG overlay (not per-camera), rendered at a
fixed pixel position for the whole video, with 3 live numbers on an
otherwise-static graphic (labels, team logos, borders). This exploits that
directly: find every part of the overlay that's static except for a few
enclosed pockets of change, then use each pocket's own background color and
known starting value to say which of the 3 it is.

1. Sample frames spread across most of the video and compute a per-pixel
   temporal activity range: static pixels land near 0, anything that changes
   (camera motion, a digit swap) lands high. This isn't a new idea: 00's own
   BAND_MAX_TROUGH_FRAC comment already measured, on match1, that scoreboard
   digits spike row_var to 1.3-3.4% of peak against a real camera-seam
   separator's 0.08-0.1% -- i.e. this exact "static overlay except a few
   numbers" signature was already observed there as an unwanted side effect
   (it exists to filter that spike back OUT). Here it's the primary signal.

   UNLIKE homography/00_split_views.py's accumulate_range, this uses a
   PERCENTILE range (hi_pctl - lo_pctl across sampled frames), not literal
   max-min. Found the hard way on match2: that video isn't a clean
   single-segment capture like match1 -- it's a produced broadcast VOD that
   bookends the actual match with a static title card and post-match
   celebration b-roll. Those few frames of totally different content are
   enough to drive max-min to ~255 across effectively the WHOLE frame
   (confirmed: 100% of pixels registered "dynamic", one connected component
   covering the entire image) because max-min is wrecked by a single outlier
   frame per pixel. A percentile range tolerates a minority of contaminated
   samples instead of being destroyed by them -- see ACTIVITY_LOW_PCTL/
   ACTIVITY_HIGH_PCTL.
2. Threshold the range image into a static/dynamic mask, then take connected
   components of the DYNAMIC side and keep only "pockets": components that
   don't touch the frame border (real camera-view motion almost always
   reaches the edge and gets rejected by this alone) and fall inside a
   plausible digit-sized area range.
3. Classify each pocket by the STATIC color in a ring immediately around it
   (not the pocket itself, which is dominated by the changing glyph pixels).
   Solid blue -> candidate blue score. Solid red -> candidate red score.
   Solid white -> candidate clock. All 3 colors were confirmed by eye against
   real match1 frames before being hardcoded (see data/archive/ discussion /
   memory) -- this isn't a guess.
3b. Merge same-color pockets that are row-aligned and horizontally adjacent
   into one box per number field (merge_adjacent_pockets). A multi-digit
   score doesn't reliably survive step 2 as ONE component -- whether the
   static gap between two digit glyphs stays below STATIC_THRESHOLD for the
   whole sampled window turned out to be font/rendering-specific: match1's
   2-digit score merged on its own, match2's 3-digit score split into 3
   separate ~30px pockets, one per digit. Without this step, whichever
   single digit slot happened to be OCR-readable in step 4 would get used as
   "the" box -- fine pre-match when every slot reads the same thing, wrong
   once the real score grows past that one digit.
4. Disambiguate the real score/clock box from any other same-colored UI
   element by reading the earliest few sampled frames with OCR and requiring
   the value to actually match the known pre-match state: 0 for a score, and
   0:AUTO_DURATION_SEC for the clock (2026's auto period is 20s, confirmed
   against real footage -- see AUTO_DURATION_SEC below). Both match1 and
   match2 have same-colored elements that also start at 0 and are NOT
   reliably smaller than the real score digits, so area alone can't reject
   them -- what actually does is assign_roles trying same-colored candidates
   largest-first and stopping at the first one whose value confirms, so the
   real (bigger) score box wins even when a same-colored decoy also
   confirms. This also rejects same-colored elements that aren't even the
   right KIND of thing -- match1 has a teleop-only shift-change counter that
   also sits on white, and match2's largest blue-colored candidate turned
   out to be part of the "FIRST 2026 CHAMPIONSHIP" branding text -- neither
   reads the expected pre-match value, so both get excluded without the code
   needing to know they exist.
5. Separately, find each alliance's small edge badge (blue_badge/red_badge):
   a same-colored, also-starts-at-0 element, but small and positioned way
   out at the frame's left (blue) / right (red) margin rather than centrally
   -- see EDGE_ZONE_FRAC. This is the thing step 4 would otherwise have
   discarded as a same-colored decoy; it's tracked as its own box instead
   because POSITION, not size, is what actually identifies it, and because
   it isn't always redundant with the main score -- 2026's game scores
   exactly 1 point per scoring action, so a scoring-actions counter and a
   point total happen to read the same number this year, but that's a
   property of this year's game, not something to assume holds in general.
   NOT assumed to be exactly one badge per side either: 2025's REEFSCAPE
   (match3) has TWO, side by side with different icons, both starting at 0
   -- every confirmed edge-zone candidate is kept (blue_badges/red_badges
   are lists), not just the largest/first.

Frame sampling deliberately starts close to the beginning of the video
(SAMPLE_START_FRAC), NOT homography/00's default 0.25-0.75 -- that range
exists there to dodge intro/outro CG graphics for a different problem (view
splitting). Here the pre-match 0/(clock) state is exactly what step 4 needs
to see, and on match1 it's already on screen by ~1% into the video, well
before any intro graphic would need dodging. The pre-match CLOCK value
itself is year-specific, not just its duration -- 2026 previews the auto
length ("0:20"), 2025 (match3) instead reads "0:00" pre-match, confirmed by
extracting and viewing an actual frame rather than assumed -- pass
`--auto-duration 0` for that.

Usage
-----
  python pipeline/02_detect_overlay.py --video match1.mp4 --save --viz

Output (JSON to stdout, and data/<match>_overlay_boxes.json with --save)
------------------------------------------------------------------------
{
  "match": "match1", "img_w":.., "img_h":..,
  "boxes": {"blue_score": {"box":[x0,y0,x1,y1], "confirmed":true, "area_frac":..},
            "red_score": {...}, "clock": {...},
            "blue_badges": [{"box":.., "confirmed":true, "area_frac":..,
                              "icon_box":[x0,y0,x1,y1]}, ...],
            "red_badges": [...]},
  "candidates": [ {"box":.., "area":.., "frac":.., "color":.., "role":..,
                    "expected":.., "confirmed":.., "votes":[..]} , ... ]
}
`boxes` may have fewer keys than shown -- a role with no confirmed candidate
is just absent, not null. blue_score/red_score/clock are always a single box
dict when present; blue_badges/red_badges are always a LIST (possibly of
length 1, as in 2026) when present, never a bare box dict -- check `isinstance`
before assuming which, same as `visualize()` does. Each badge entry's
`icon_box` is the location of the game-piece icon beside it (see
find_icon_box) -- WHERE it is only, not what it depicts; `icon_box` is null
if no confident white blob was found for that badge. `candidates` lists
every pocket that survived the geometric filter, not just the winners --
useful for --viz debugging when a role comes up missing or with the wrong
count.

Known limitations / unvalidated constants
------------------------------------------
- MIN/MAX_POCKET_AREA_FRAC, the HSV color ranges, STATIC_THRESHOLD, and
  ACTIVITY_LOW_PCTL/HIGH_PCTL are tuned from two matches' footage (match1's
  in-match capture, match2's title-card+celebration-bookended VOD) plus 00's
  existing value for STATIC_THRESHOLD -- not yet swept broadly. Per the
  agreed design, treat each as independently replaceable: if a match's boxes
  come up missing or wrong, check which stage (pocket-finding, color, or
  initial-value confirmation) actually failed before changing anything else.
- The percentile trim handles a MINORITY of contaminated frames (a title
  card, some post-match b-roll), not an arbitrary amount -- a video where
  non-match content is a large fraction of the sampled window would still
  break this. Tested against match1 (no contamination) and match2 (~5-10%
  estimated, from a title card at the very start plus celebration footage
  past the match's real end); not tested against anything worse.
- Calibration (a digit template bank, so a later extraction step doesn't
  need OCR/GPU per frame) and extraction (the actual per-match score-change
  timeline) are separate, not-yet-built pipeline stages -- this file only
  answers "where are the 3 boxes", nothing about reading their live values
  over time.
- AUTO_DURATION_SEC is 2026-specific by design, same reasoning as why
  pipeline/01_audio.py's cue timing lives in data/cue_profiles/<year>.json
  rather than as a literal: the code stays year-agnostic (--auto-duration is
  a flag, not a constant baked into the logic), only the default value is
  year-specific. Move it into a proper year-keyed profile once a second
  year is supported.

Install: pip install opencv-python numpy easyocr torch
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

# See module docstring for why this starts much earlier than homography/00's
# 0.25 default -- on match1 (172s @ 1920x1080) the pre-match 0-0/0:20 state
# is already on screen by t=2s, i.e. ~1% in, so 0.01 leaves margin without
# needing to guess how long a pre-match hold runs on other matches.
SAMPLE_START_FRAC = 0.01
# Mirrors homography/00's reasoning for avoiding the tail: dodges post-match
# graphics that could otherwise dilute the activity signal with a different
# frame composition than the in-match overlay layout.
SAMPLE_END_FRAC = 0.85

# ---------------------------------------------------------------------------
# Activity / pocket detection
# ---------------------------------------------------------------------------

# Starting point reused verbatim from homography/pipeline/00_split_views.py's
# STATIC_THRESHOLD (same signal, same source of truth for what "static" means
# on real compressed H.264 broadcast footage: static ~2-5, camera content ~30+).
STATIC_THRESHOLD = 12

# Percentile pair used for the activity range (hi_pctl - lo_pctl) instead of
# literal max-min -- see module docstring for why (match2's title-card +
# post-match-celebration bookends broke max-min outright). Trims the most
# extreme 8% of samples on each side per pixel before taking the spread, so a
# minority of contaminated frames gets treated as outliers rather than
# dominating the result. Not swept -- if a real digit change gets missed
# because it only shows up in a handful of the sampled frames (e.g. a score
# that changes very late within the sampled window), narrowing this trim is
# the first thing to try; if contamination bigger than what match2 has
# starts slipping through, widening it is the first thing to try.
ACTIVITY_LOW_PCTL  = 8
ACTIVITY_HIGH_PCTL = 92

# Real measured areas (1920x1080, area/frame_area), corrected from initial
# eyeballed estimates once actual candidates were inspected: a merged 2-3
# digit score field is ~0.0012-0.0017; the small per-alliance badge numbers
# in the far screen corners (also blue/red, also start at 0 -- see docstring)
# measure ~0.0004-0.0005, NOT ~0.0002 as originally guessed from the image --
# they are NOT reliably excludable by area alone, and in practice aren't
# excluded this way: assign_roles' largest-first + initial-value-confirm
# order is what actually keeps them from winning (they're smaller than the
# real merged score box, so the real one is tried first and wins the role).
# MIN exists only to reject near-noise-sized components -- found the hard
# way on match2 that it can't be set much above that: a single "1" digit
# (the clock's minutes place, sparser than a "0"/"8"-shaped glyph) measured
# 0.00036 and was being wrongly excluded at the previous 0.0004, silently
# truncating "1:20" down to "20" in the merged clock box. Lowered again on
# match3 (2025, REEFSCAPE): one of its two per-alliance edge badges (see
# EDGE_ZONE_FRAC) measured only 0.000156 -- narrower/less-active digits than
# any seen before, still a clean 6x+ above the actual per-frame noise floor
# in that same region (stray sub-pixel connected components measured
# 0.000001-0.000025 there, confirmed by inspecting connectedComponentsWithStats
# output directly, not guessed). MAX is a generous ceiling for a 3-digit
# score, not a tight fit -- exists only to reject something pocket-shaped
# but implausibly large.
MIN_POCKET_AREA_FRAC = 0.0001
MAX_POCKET_AREA_FRAC = 0.02

# How close to the frame's LEFT (blue) / RIGHT (red) edge a same-colored
# pocket needs to sit to count as the small per-alliance edge badge -- a
# distinct scoreboard element from the main score, same color, also reads 0
# pre-match, but positioned way out at the margins instead of centrally.
# Observed on both match1 and match2: the badge sits at roughly x=130-180
# (blue) / x=1780-1840 (red) out of a 1920px-wide frame -- comfortably
# inside a generous 20% zone, while the main score sits centrally around
# 40-60% across. In 2026 this badge is numerically redundant with the main
# score (this year's game scores exactly 1 point per scoring action, so a
# scoring-actions count and a point total happen to be the same number) --
# NOT assumed to hold in other years, which is the whole reason this is
# tracked as its own box rather than being treated as just another
# same-colored decoy to discard.
EDGE_ZONE_FRAC = 0.2

# ---------------------------------------------------------------------------
# Color classification
# ---------------------------------------------------------------------------

# Ring width (px) sampled around a candidate pocket to read its STATIC
# background chip color -- deliberately excludes the pocket's own pixels,
# which are dominated by the changing digit glyph rather than the chip.
COLOR_RING_MARGIN = 6

# OpenCV hue is 0-179. Blue/red/white ranges below were read off real match1
# frames (a bright, fully-saturated alliance-color chip; a bright, low-
# saturation white chip), not assumed.
HUE_BLUE_RANGE   = (95, 135)
HUE_RED_RANGES   = ((0, 10), (170, 179))   # red wraps around hue 0
SAT_MIN_COLOR    = 90    # blue/red chip: must be strongly saturated
VAL_MIN_COLOR    = 60
SAT_MAX_WHITE    = 60    # white chip: must be weakly saturated ...
VAL_MIN_WHITE    = 170   # ... and bright

# Fraction of ring pixels that must agree on one color for a confident call.
MIN_COLOR_MATCH_FRAC = 0.5

# ---------------------------------------------------------------------------
# OCR confirmation (initial-value check)
# ---------------------------------------------------------------------------

OCR_UPSCALE = 4

# How many of the earliest sampled frames to try reading per candidate, and
# how many of those must agree with the known pre-match value for it to
# count as confirmed. Deliberately lenient (2 of 5, not unanimous) since a
# single frame's OCR read is noisy; this is the same reject-one-off-misread
# idea as smoothing elsewhere in this pipeline, just scoped to a handful of
# frames instead of a whole match.
N_CONFIRM_FRAMES  = 5
MIN_CONFIRM_AGREE = 2

# 2026-specific: the clock reads 0:20 (the auto period length) immediately
# before the match starts -- confirmed against real match1 footage, not
# assumed. See module docstring for why this is a flag with a year-specific
# default rather than a hardcoded literal deeper in the logic.
AUTO_DURATION_SEC = 20


# ---------------------------------------------------------------------------
# Frame sampling
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


def accumulate_range(frames_gray: list[np.ndarray]) -> np.ndarray:
    """Per-pixel temporal activity: (high percentile - low percentile) across
    sampled frames, NOT literal max-min -- see ACTIVITY_LOW_PCTL/HIGH_PCTL
    and the module docstring for why (max-min is wrecked outright by a
    single outlier-content frame per pixel, which real produced broadcast
    VODs turned out to have -- match2's title-card/celebration bookends).
    Stacks all sampled frames as uint8 (not float32) to compute the
    percentile -- e.g. ~430MB for 240 frames at 1920x1080, a bounded
    one-time cost per run. homography/00_split_views.py's accumulate_range
    avoids exactly this stacking (for a true max-min, an incremental
    running min/max needs no history at all), but that shortcut isn't
    available once outlier-robustness is the point."""
    stack = np.stack(frames_gray)   # (n, h, w) uint8
    lo, hi = np.percentile(stack, [ACTIVITY_LOW_PCTL, ACTIVITY_HIGH_PCTL], axis=0)
    return (hi - lo).astype(np.float32)


# ---------------------------------------------------------------------------
# Pocket detection
# ---------------------------------------------------------------------------

def find_pockets(range_img: np.ndarray, static_threshold: float,
                 min_area_frac: float, max_area_frac: float) -> list[dict]:
    """Connected components of the DYNAMIC mask, keeping only components that
    (a) don't touch the frame border -- a real camera view's activity almost
    always reaches the edge, so this alone should reject it without needing
    to separately isolate the CG overlay panel first -- and (b) fall inside a
    plausible digit-box area range. See module docstring for where the area
    bounds came from."""
    h, w = range_img.shape
    total = h * w
    dynamic = (range_img > static_threshold).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(dynamic, connectivity=8)

    pockets = []
    for label in range(1, n_labels):   # label 0 is the background component
        x, y, bw, bh, area = stats[label]
        if x <= 0 or y <= 0 or (x + bw) >= w or (y + bh) >= h:
            continue   # touches the frame border -- not an enclosed pocket
        frac = area / total
        if not (min_area_frac <= frac <= max_area_frac):
            continue
        pockets.append({"box": [int(x), int(y), int(x + bw), int(y + bh)],
                        "area": int(area), "frac": round(float(frac), 6)})
    return pockets


# ---------------------------------------------------------------------------
# Color classification
# ---------------------------------------------------------------------------

def classify_color(frame_bgr: np.ndarray, box: list[int],
                   margin: int = COLOR_RING_MARGIN) -> str | None:
    """Classify a pocket by the STATIC background color in a ring around it
    (the pocket's own pixels are dominated by the changing digit glyph, not
    the chip it sits on -- sampling outside it is what actually isolates the
    chip color)."""
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = box
    ox0, oy0 = max(0, x0 - margin), max(0, y0 - margin)
    ox1, oy1 = min(w, x1 + margin), min(h, y1 + margin)
    outer = frame_bgr[oy0:oy1, ox0:ox1]
    if outer.size == 0:
        return None

    ring_mask = np.ones(outer.shape[:2], dtype=bool)
    ix0, iy0 = x0 - ox0, y0 - oy0
    ix1, iy1 = x1 - ox0, y1 - oy0
    ring_mask[iy0:iy1, ix0:ix1] = False
    ring = outer[ring_mask]
    if ring.size == 0:
        return None

    hsv = cv2.cvtColor(ring.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue, sat, val = hsv[:, 0].astype(np.int32), hsv[:, 1], hsv[:, 2]

    is_blue = ((hue >= HUE_BLUE_RANGE[0]) & (hue <= HUE_BLUE_RANGE[1]) &
              (sat >= SAT_MIN_COLOR) & (val >= VAL_MIN_COLOR))
    (rlo0, rlo1), (rhi0, rhi1) = HUE_RED_RANGES
    is_red = (((hue >= rlo0) & (hue <= rlo1)) | ((hue >= rhi0) & (hue <= rhi1))) & \
             (sat >= SAT_MIN_COLOR) & (val >= VAL_MIN_COLOR)
    is_white = (sat <= SAT_MAX_WHITE) & (val >= VAL_MIN_WHITE)

    fracs = {"blue": float(is_blue.mean()), "red": float(is_red.mean()),
             "white": float(is_white.mean())}
    best = max(fracs, key=fracs.get)
    return best if fracs[best] >= MIN_COLOR_MATCH_FRAC else None


# ---------------------------------------------------------------------------
# Adjacent-pocket merge
# ---------------------------------------------------------------------------

# How close two same-color, same-row pockets need to be (in px) to merge into
# one number field. Found necessary on match2: a growing multi-digit score
# doesn't reliably land as one connected component -- whether the single-px
# static gap between two digit glyphs stays below STATIC_THRESHOLD for the
# WHOLE sampled window (merging them) or not (splitting them) turned out to
# be font/rendering-specific and inconsistent (match1's 2-digit score merged
# on its own; match2's 3-digit score split into 3 separate ~30px pockets, one
# per digit, with real observed gaps of 0-1px between them). Generous
# relative to that observed gap since the point is not to miss a real split.
MERGE_MAX_GAP_PX = 15
# How much vertical misalignment between two pockets' top/bottom edges is
# still "the same text baseline" rather than a coincidentally-nearby,
# unrelated element -- observed y0 jitter between adjacent real digits was
# 0-1px.
MERGE_MAX_Y_OFFSET_PX = 6


def merge_adjacent_pockets(pockets: list[dict], frame_area: int) -> list[dict]:
    """Merge same-color, row-aligned, horizontally-close pockets into one
    combined box per number field. Without this, whichever single digit slot
    happened to be OCR-readable at initial-value-check time gets used as
    "the" score/clock box -- fine pre-match when every slot shows the same
    thing, but too narrow once the real score grows past that one digit. See
    MERGE_MAX_GAP_PX docstring for how this was discovered."""
    by_color: dict[str | None, list[dict]] = {}
    for p in pockets:
        by_color.setdefault(p.get("color"), []).append(p)

    merged: list[dict] = []
    for color, group in by_color.items():
        group = sorted(group, key=lambda p: p["box"][0])   # left to right
        used = [False] * len(group)
        for i in range(len(group)):
            if used[i]:
                continue
            x0, y0, x1, y1 = group[i]["box"]
            used[i] = True
            grew = True
            while grew:
                grew = False
                for j in range(len(group)):
                    if used[j]:
                        continue
                    qx0, qy0, qx1, qy1 = group[j]["box"]
                    same_row = (abs(qy0 - y0) <= MERGE_MAX_Y_OFFSET_PX and
                               abs(qy1 - y1) <= MERGE_MAX_Y_OFFSET_PX)
                    close_x = qx0 - x1 <= MERGE_MAX_GAP_PX and x0 - qx1 <= MERGE_MAX_GAP_PX
                    if same_row and close_x:
                        x0, y0, x1, y1 = min(x0, qx0), min(y0, qy0), max(x1, qx1), max(y1, qy1)
                        used[j] = True
                        grew = True
            area = (x1 - x0) * (y1 - y0)
            merged.append({"box": [x0, y0, x1, y1], "area": area,
                           "frac": round(area / frame_area, 6), "color": color})
    return merged


# ---------------------------------------------------------------------------
# OCR confirmation
# ---------------------------------------------------------------------------

_READER = None


def get_reader():
    global _READER
    if _READER is None:
        import easyocr
        print("[ocr] loading EasyOCR digit reader ...", file=sys.stderr)
        _READER = easyocr.Reader(["en"], gpu=True)
    return _READER


def preprocess_crop(frame: np.ndarray, box: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def read_digits_str(frame: np.ndarray, box: list[int], allow_colon: bool = False) -> str | None:
    """OCR a crop and return its digits as a string (colon optionally allowed
    in the OCR allowlist so it doesn't get misread as another digit, but
    always stripped from the result -- EasyOCR doesn't reliably isolate ':'
    as its own detection, so comparisons are done digit-only rather than
    depending on that)."""
    img = preprocess_crop(frame, box)
    if img.size == 0:
        return None
    reader = get_reader()
    allowlist = "0123456789:" if allow_colon else "0123456789"
    results = reader.readtext(img, allowlist=allowlist, detail=1)
    if not results:
        return None
    results.sort(key=lambda r: r[0][0][0])   # left-to-right by x position
    text = "".join(c for c in "".join(r[1] for r in results) if c.isdigit())
    return text or None


def parse_clock_seconds(digit_str: str | None) -> int | None:
    """Parse a digit-only clock read (colon already stripped) into total
    seconds. The last two digits are always seconds; anything before that is
    minutes -- so '020' and '20' both parse to 20s, which matters since OCR
    may or may not read a leading minute-digit '0'."""
    if not digit_str:
        return None
    if len(digit_str) <= 2:
        seconds = int(digit_str)
        minutes = 0
    else:
        minutes, seconds = int(digit_str[:-2]), int(digit_str[-2:])
    if seconds >= 60:
        return None
    return minutes * 60 + seconds


def confirm_initial_value(bgrs: list[np.ndarray], box: list[int], matches_fn,
                          allow_colon: bool = False, n: int = N_CONFIRM_FRAMES,
                          min_agree: int = MIN_CONFIRM_AGREE) -> tuple[bool, list]:
    """Read the earliest `n` sampled frames within `box` and check at least
    `min_agree` agree with `matches_fn`. This is what actually disambiguates
    the real score/clock box from another similarly-colored, similarly-sized
    UI element -- see module docstring."""
    votes = [read_digits_str(f, box, allow_colon=allow_colon) for f in bgrs[:n]]
    agree = sum(1 for v in votes if matches_fn(v))
    return agree >= min_agree, votes


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------

def assign_roles(pockets: list[dict], bgrs: list[np.ndarray], auto_duration: int,
                 frame_w: int) -> tuple[dict, list[dict]]:
    """For the main score/clock roles, try each color's candidate pockets
    largest-area first (the real score/clock digits are consistently bigger
    than any same-colored decoy seen so far -- see MIN_POCKET_AREA_FRAC
    docstring) and keep the first one whose initial value actually confirms
    -- exactly one winner per role, since there's only ever one real score
    box or clock. For the edge-badge roles (see EDGE_ZONE_FRAC), position --
    not size -- is the identifying signal, since that element is small by
    nature; UNLIKE the score/clock roles this keeps EVERY confirmed
    candidate on a side, not just one, since the count isn't assumed fixed
    (2026 has one badge per alliance, 2025's REEFSCAPE has two). Every
    attempt is recorded in the returned debug list regardless, so a missing
    or unexpectedly-counted role is easy to diagnose from --viz/output alone."""
    groups: dict[str, list[dict]] = {"blue": [], "red": [], "white": []}
    for p in pockets:
        if p.get("color") in groups:
            groups[p["color"]].append(p)
    for color in groups:
        groups[color].sort(key=lambda p: -p["area"])

    def is_zero(v):
        # handles "0", "00", ... -- how many digit slots the pre-match
        # display actually renders isn't something to assume
        return v is not None and v != "" and v.lstrip("0") == ""

    clock_expected = f"{auto_duration // 60}{auto_duration % 60:02d}"
    role_specs = {
        "blue":  ("blue_score", False, is_zero),
        "red":   ("red_score",  False, is_zero),
        "white": ("clock",      True,  lambda v: parse_clock_seconds(v) == auto_duration),
    }

    result: dict[str, dict] = {}
    debug: list[dict] = []
    for color, (role, allow_colon, matches_fn) in role_specs.items():
        expected_label = "0" if role != "clock" else clock_expected
        for p in groups[color]:
            ok, votes = confirm_initial_value(bgrs, p["box"], matches_fn, allow_colon=allow_colon)
            debug.append({**p, "role": role, "expected": expected_label,
                         "confirmed": ok, "votes": votes})
            if ok and role not in result:
                result[role] = {"box": p["box"], "confirmed": True, "area_frac": p["frac"]}

    # Edge badges: 2026 has one per alliance, 2025 (match3, REEFSCAPE) has
    # TWO side by side with different icons -- not assumed to be exactly one,
    # so every confirmed edge-zone candidate is kept, not just the first/
    # largest. Sorted left-to-right for a deterministic reading order rather
    # than by area, since picking "the winner" doesn't make sense once more
    # than one is expected to be real.
    #
    # Checks BOTH the left and right edge zone for EACH color, not "blue is
    # always left, red is always right" -- match1/2/3 all happened to put
    # blue on the left, but match7 mirrors it (red alliance displayed on the
    # left, blue on the right). A hardcoded side/color pairing found match7's
    # real, confirmed-reads-0 blue badge sitting at x=1612 (the right edge)
    # and never even tried it, because blue_badges only ever looked left.
    # Alliance-to-side assignment isn't assumed here at all -- whichever
    # edge a color's candidates actually confirm in is used.
    for color, role in (("blue", "blue_badges"), ("red", "red_badges")):
        edge_pockets = [p for p in groups[color]
                        if p["box"][2] <= frame_w * EDGE_ZONE_FRAC
                        or p["box"][0] >= frame_w * (1 - EDGE_ZONE_FRAC)]
        edge_pockets.sort(key=lambda p: p["box"][0])
        confirmed = []
        for p in edge_pockets:
            ok, votes = confirm_initial_value(bgrs, p["box"], is_zero)
            debug.append({**p, "role": role, "expected": "0", "confirmed": ok, "votes": votes})
            if ok:
                confirmed.append({"box": p["box"], "confirmed": True, "area_frac": p["frac"]})
        if confirmed:
            result[role] = confirmed

    # candidates whose color didn't match any role, kept for --viz/debugging
    for p in pockets:
        if p.get("color") not in groups:
            debug.append({**p, "role": None, "expected": None, "confirmed": False, "votes": []})
    return result, debug


# ---------------------------------------------------------------------------
# Icon (game-piece counter) bounding-box search
# ---------------------------------------------------------------------------
#
# Each edge badge sits next to a small icon identifying WHICH game-piece
# count it is (2026: a 6-dot "fuel" pyramid; 2025 REEFSCAPE: separate coral
# and algae icons) -- white background, alliance-color foreground, always
# adjacent to its digit (not necessarily touching -- match3's red badges
# have a real gap of solid-red pill padding before the icon starts), but NOT
# consistently on the same side across district broadcast layouts (left of
# the digit in some, right in others). This file's job is ONLY to find where
# that icon is, not what it depicts -- classifying the crop (which needs
# year-specific reference images, e.g. rasterized from an SVG source or a
# team's open-source arena software) is calibration's job, not detection's,
# same reasoning as why digit-template bootstrapping lives in calibration
# and not here.
#
# This can't reuse the activity-based pocket search at all: the icon is
# STATIC by definition (that's the whole reason it isn't found already), so
# accumulate_range/find_pockets are structurally blind to it. It also can't
# be found by searching a fixed-size window sized off the digit's own
# activity-box: that box's WIDTH varies with the currently-displayed value
# (a single-digit "8" measures 15px, a two-digit "16" measures ~30px), but
# the physical distance to the icon is a fixed property of the overlay's
# layout, not something that shrinks just because the number happens to be
# small right now -- confirmed the hard way on match3's narrow "8" badge,
# where a width-proportional search window undershot and missed the icon
# entirely even though it was clearly visible just a bit further out.
#
# Design instead: search outward from the digit in both directions, up to a
# generous but bounded distance (ICON_SEARCH_MAX_REACH_PX), and take the
# CLOSEST white connected-component that also validates as an icon (roughly
# square, with a meaningful fraction of alliance-colored pixels inside it --
# a plain white rectangle without that colored content isn't the icon, it's
# something else). This avoids needing to guess a precise reach at all.
# Deliberately not a frame-wide scan even at 500px: nothing is a candidate
# unless a connected white blob is found scanning outward from an
# already-validated digit, so an unrelated white UI element (a sponsor logo,
# other chrome) elsewhere in the overlay never enters consideration in the
# first place, and the validation step rejects anything that's white but
# isn't actually icon-shaped/icon-colored.
#
# Multi-badge pairing (2025's two side-by-side badges per alliance) is
# resolved GLOBALLY rather than by bounding each digit's search window:
# search every digit in BOTH directions independently, then check which
# single direction (all-left or all-right) pairs every digit with a
# DISTINCT icon and leaves none empty. The wrong direction reliably fails
# that check on its own -- e.g. for an icon-then-digit layout
# (icon_A digit_A icon_B digit_B), searching right from digit_A finds
# icon_B (not its own), and searching right from digit_B (the last element)
# finds nothing at all -- so "all-right" comes back with an empty and is
# rejected, while "all-left" pairs both correctly. No assumption about
# which side icons render on is needed; it falls out of the geometry.

# Outward search cap in px -- generous on purpose (this replaced an earlier
# version that tried to size the search off the digit box, which broke on
# a narrow single-digit value -- see design note above). Still bounded, not
# unbounded, so a pathological case can't walk arbitrarily far into
# unrelated content.
ICON_SEARCH_MAX_REACH_PX = 500

# Noise floor for a candidate white blob -- same role as MIN_POCKET_AREA_FRAC
# but for a static search window instead of a dynamic one. UNVALIDATED, no
# real icon-crop measurements yet (unlike MIN_POCKET_AREA_FRAC's, which came
# from inspecting real connectedComponentsWithStats output) -- starting
# point only, check against --viz output before trusting it.
ICON_MIN_AREA_FRAC = 0.00005

# A validated icon chip must be roughly square (width/height in this range)
# -- this is what rejects a plain wide white strip (e.g. part of a label
# background) that isn't actually the icon. Generous tolerance since the
# true aspect ratio hasn't been measured precisely yet.
ICON_ASPECT_RANGE = (0.6, 1.6)

# Minimum fraction of a candidate white blob's pixels that must be the
# alliance's own color for it to count as a real icon rather than a blank
# white rectangle -- confirms there's actually a colored glyph drawn inside
# it, not just proving the background is white. UNVALIDATED starting point.
ICON_MIN_COLOR_FRAC = 0.08

# STARTING vertical margin (px) added above/below the digit box's own
# y-range when building the icon search window -- the digit box is an
# ACTIVITY bbox (just the numeral strokes), which is shorter than the icon
# chip actually is. This is a starting point, not a fixed final size: see
# ICON_MAX_EXPAND_ITERS below for why a fixed margin isn't trustworthy no
# matter what value it's set to.
ICON_Y_MARGIN_PX = 10

# How many times to DOUBLE the search window and retry if the best
# candidate's box touches the window's own edge. A fixed margin can't be
# made "big enough" once and trusted -- measured directly on match1: even
# after ICON_Y_MARGIN_PX=10 was enough to pass the aspect-ratio check (the
# bug this was first added to fix), the reported box was STILL clipped 4px
# short on the top edge specifically (0px short on the other 3 sides) --
# not anti-aliasing, the connected white blob genuinely extended past the
# window's own boundary, which the component search can never see past
# regardless of threshold tuning. Detecting the clip and re-searching a
# larger window is what actually guarantees the full chip is captured,
# rather than a second guessed constant that could just as easily be wrong
# for some other icon's proportions. Same idea as MERGE_MAX_GAP_PX growing
# a pocket outward until nothing adjacent qualifies anymore, applied here to
# the search window instead of the merge result.
ICON_MAX_EXPAND_ITERS = 4


def _icon_candidates(ref_bgr: np.ndarray, window: list[int]) -> list[list[int]]:
    """Every connected white-background region inside `window` above the
    noise floor, as absolute-coordinate boxes (unordered)."""
    x0, y0, x1, y1 = window
    if x1 <= x0 or y1 <= y0:
        return []
    crop = ref_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white_mask = ((hsv[:, :, 1] <= SAT_MAX_WHITE) & (hsv[:, :, 2] >= VAL_MIN_WHITE)).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    frame_area = ref_bgr.shape[0] * ref_bgr.shape[1]
    boxes = []
    for lbl in range(1, n_labels):
        bx, by, bw, bh, barea = stats[lbl]
        if barea / frame_area < ICON_MIN_AREA_FRAC:
            continue
        boxes.append([x0 + int(bx), y0 + int(by), x0 + int(bx) + int(bw), y0 + int(by) + int(bh)])
    return boxes


def _is_icon(ref_bgr: np.ndarray, box: list[int], alliance_color: str) -> bool:
    """Roughly square, AND a meaningful fraction of alliance-colored pixels
    inside it -- see ICON_ASPECT_RANGE/ICON_MIN_COLOR_FRAC docstrings for
    why both checks exist (square alone would also accept a blank white
    swatch; color alone would also accept a wide label strip with some
    colored text in it)."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bh == 0 or not (ICON_ASPECT_RANGE[0] <= bw / bh <= ICON_ASPECT_RANGE[1]):
        return False
    crop = ref_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0].astype(np.int32), hsv[:, :, 1], hsv[:, :, 2]
    if alliance_color == "blue":
        is_color = ((hue >= HUE_BLUE_RANGE[0]) & (hue <= HUE_BLUE_RANGE[1]) &
                   (sat >= SAT_MIN_COLOR) & (val >= VAL_MIN_COLOR))
    else:
        (rlo0, rlo1), (rhi0, rhi1) = HUE_RED_RANGES
        is_color = (((hue >= rlo0) & (hue <= rlo1)) | ((hue >= rhi0) & (hue <= rhi1))) & \
                   (sat >= SAT_MIN_COLOR) & (val >= VAL_MIN_COLOR)
    return float(is_color.mean()) >= ICON_MIN_COLOR_FRAC


def _search_icon_direction(ref_bgr: np.ndarray, digit_box: list[int], direction: str,
                           alliance_color: str, frame_w: int) -> list[int] | None:
    """Closest validated icon in `direction` ("left"/"right") from
    `digit_box`, searching out to ICON_SEARCH_MAX_REACH_PX. Checks
    candidates nearest-first and returns the first one that validates, so a
    stray noise-sized-but-still-qualifying blob closer than the real icon
    can't win by proximity alone without also passing _is_icon.

    If the winning candidate's box touches the search window's own y-edge,
    that's a sign the connected-component search never saw the rest of the
    blob -- the window clipped it, not the blob's real extent. Doubles the
    window and retries (ICON_MAX_EXPAND_ITERS times) rather than trusting
    ICON_Y_MARGIN_PX to already be big enough; see that constant's docstring
    for why a fixed margin isn't good enough on its own."""
    x0, y0, x1, y1 = digit_box
    frame_h = ref_bgr.shape[0]
    y_margin = ICON_Y_MARGIN_PX
    box = None
    for _ in range(ICON_MAX_EXPAND_ITERS):
        y_lo, y_hi = max(0, y0 - y_margin), min(frame_h, y1 + y_margin)
        if direction == "left":
            window = [max(0, x0 - ICON_SEARCH_MAX_REACH_PX), y_lo, x0, y_hi]
            dist_fn = lambda b: x0 - b[2]
        else:
            window = [x1, y_lo, min(frame_w, x1 + ICON_SEARCH_MAX_REACH_PX), y_hi]
            dist_fn = lambda b: b[0] - x1

        candidates = sorted(_icon_candidates(ref_bgr, window), key=dist_fn)
        box = next((b for b in candidates if _is_icon(ref_bgr, b, alliance_color)), None)
        if box is None:
            return None
        clipped = box[1] <= window[1] or box[3] >= window[3]
        window_maxed = y_lo <= 0 and y_hi >= frame_h
        if not clipped or window_maxed:
            return box
        y_margin *= 2
    return box


def attach_icon_boxes(badges: list[dict], ref_bgr: np.ndarray, frame_w: int,
                      alliance_color: str) -> list[dict]:
    """Attach an `icon_box` (or None) to each badge in a same-side group,
    already sorted left-to-right by assign_roles. Searches every badge in
    both directions independently, then resolves which single direction is
    correct GLOBALLY via the emptiness/uniqueness check described in the
    section docstring above -- not by bounding each badge's own search
    window, which is what let two adjacent badges' windows overlap and
    both grab the same icon in an earlier version of this function."""
    if not badges:
        return badges

    left_found  = [_search_icon_direction(ref_bgr, b["box"], "left", alliance_color, frame_w) for b in badges]
    right_found = [_search_icon_direction(ref_bgr, b["box"], "right", alliance_color, frame_w) for b in badges]

    def viable(found: list[list[int] | None]) -> bool:
        if any(f is None for f in found):
            return False
        seen = set()
        for f in found:
            key = tuple(f)
            if key in seen:
                return False
            seen.add(key)
        return True

    left_ok, right_ok = viable(left_found), viable(right_found)
    if left_ok and not right_ok:
        chosen = left_found
    elif right_ok and not left_ok:
        chosen = right_found
    elif left_ok and right_ok:
        # Both directions produced a complete, distinct pairing -- shouldn't
        # normally happen (see section docstring), but if it does, prefer
        # whichever is closer on average rather than silently picking one.
        avg_dist = lambda found, key: sum(abs(f[key] - b["box"][key]) for f, b in zip(found, badges)) / len(badges)
        chosen = left_found if avg_dist(left_found, 2) <= avg_dist(right_found, 0) else right_found
    else:
        chosen = [None] * len(badges)

    return [{**b, "icon_box": icon} for b, icon in zip(badges, chosen)]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_ROLE_DRAW_COLOR = {"blue_score": (255, 150, 0), "red_score": (0, 0, 255), "clock": (0, 210, 210),
                    "blue_badges": (255, 210, 130), "red_badges": (130, 130, 255)}


def visualize(ref_bgr: np.ndarray, range_img: np.ndarray,
             pockets: list[dict], boxes: dict) -> np.ndarray:
    h, w = ref_bgr.shape[:2]
    vis = ref_bgr.copy()
    for p in pockets:
        x0, y0, x1, y1 = p["box"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (140, 140, 140), 1)
    for role, info in boxes.items():
        # blue_badges/red_badges are a LIST of boxes (2025's REEFSCAPE has 2
        # per alliance, not just 1) -- everything else is a single box dict.
        entries = info if isinstance(info, list) else [info]
        color = _ROLE_DRAW_COLOR.get(role, (0, 255, 0))
        for entry in entries:
            x0, y0, x1, y1 = entry["box"]
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            cv2.putText(vis, role, (x0, max(12, y0 - 6)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            icon_box = entry.get("icon_box")
            if icon_box:
                ix0, iy0, ix1, iy1 = icon_box
                cv2.rectangle(vis, (ix0, iy0), (ix1, iy1), (0, 255, 255), 1)

    scaled = np.clip(range_img / max(float(range_img.max()), 1.0) * 255, 0, 255).astype(np.uint8)
    range_bgr = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)

    scale = 0.6
    vis_s = cv2.resize(vis, (int(w * scale), int(h * scale)))
    range_s = cv2.resize(range_bgr, (int(w * scale), int(h * scale)))
    return np.hstack([vis_s, range_s])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, metavar="PATH")
    ap.add_argument("--n-frames", type=int, default=N_SAMPLE_FRAMES)
    ap.add_argument("--start-frac", type=float, default=SAMPLE_START_FRAC)
    ap.add_argument("--end-frac", type=float, default=SAMPLE_END_FRAC)
    ap.add_argument("--static-threshold", type=float, default=STATIC_THRESHOLD)
    ap.add_argument("--min-pocket-frac", type=float, default=MIN_POCKET_AREA_FRAC)
    ap.add_argument("--max-pocket-frac", type=float, default=MAX_POCKET_AREA_FRAC)
    ap.add_argument("--auto-duration", type=int, default=AUTO_DURATION_SEC,
                    help=f"seconds the clock reads before match start (default: "
                         f"{AUTO_DURATION_SEC}, 2026-specific -- see module docstring)")
    ap.add_argument("--save", action="store_true", help="write data/<match>_overlay_boxes.json")
    ap.add_argument("--viz", action="store_true")
    args = ap.parse_args()

    video_path = pathlib.Path(args.video)
    if not video_path.exists():
        sys.exit(f"[error] video not found: {video_path}")
    match_name = video_path.stem

    print(f"[sample] loading frames from {video_path} ...", file=sys.stderr)
    grays, bgrs, fps, w, h = sample_frames(video_path, args.n_frames, args.start_frac, args.end_frac)
    if len(grays) < 2:
        sys.exit("[error] fewer than 2 frames sampled -- check --video/--start-frac/--end-frac")
    print(f"[sample] {len(grays)} frames, {w}x{h} @ {fps:.1f}fps", file=sys.stderr)

    range_img = accumulate_range(grays)
    pockets = find_pockets(range_img, args.static_threshold, args.min_pocket_frac, args.max_pocket_frac)
    print(f"[pockets] {len(pockets)} candidate(s)", file=sys.stderr)

    ref_frame = bgrs[len(bgrs) // 2]
    for p in pockets:
        p["color"] = classify_color(ref_frame, p["box"])

    pockets = merge_adjacent_pockets(pockets, frame_area=w * h)
    print(f"[pockets] {len(pockets)} after merging adjacent same-number digits", file=sys.stderr)

    boxes, debug_candidates = assign_roles(pockets, bgrs, args.auto_duration, w)
    for role in ("blue_score", "red_score", "clock"):
        print(f"[detect] {role}: {'found ' + str(boxes[role]['box']) if role in boxes else 'MISSING'}",
              file=sys.stderr)
    for role in ("blue_badges", "red_badges"):
        if role in boxes:
            alliance_color = "blue" if role.startswith("blue") else "red"
            boxes[role] = attach_icon_boxes(boxes[role], ref_frame, w, alliance_color)
        found = boxes.get(role, [])
        icon_summary = [b["icon_box"] for b in found]
        print(f"[detect] {role}: {len(found)} found -- {[b['box'] for b in found]} "
              f"icons={icon_summary}", file=sys.stderr)

    output = {
        "match": match_name,
        "img_w": w, "img_h": h,
        "boxes": boxes,
        "candidates": debug_candidates,
    }

    if args.save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DATA_DIR / f"{match_name}_overlay_boxes.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"[save] -> {out_path}", file=sys.stderr)

    if args.viz:
        vis = visualize(ref_frame, range_img, pockets, boxes)
        viz_path = DATA_DIR / f"{match_name}_overlay_detect.jpg"
        cv2.imwrite(str(viz_path), vis)
        print(f"[viz] saved -> {viz_path}", file=sys.stderr)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
