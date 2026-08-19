#!/usr/bin/env python3
"""
Step 4 -- Identification: decide what each region 03_extract.py read actually
MEANS, and cut the timeline down to the match itself.

This is where every season-specific and broadcast-specific convention lives,
deliberately concentrated in one file. 02 finds regions, 03_extract.py reads them, and
neither knows what a score is. That split exists because of measured
failures: the old 02 assigned meaning from one pre-match frame plus an OCR
read, and produced 9 wrong boxes out of ~44 across 9 matches -- a badge that
reads "0 / 100" failing an ==0 test, a static denominator "100" OCR'ing as
'000' and passing as a score that starts at zero, "blue is on the left"
meeting a match where red is, a 2025 pre-match clock reading 0:00 instead of
0:20. None of those are threshold problems. They are all the same problem:
the evidence needed to decide arrives later.

By the time this file runs, that evidence exists. A region is identified by
how its value BEHAVES across a whole match -- thousands of reads -- rather
than by what it looked like once before the match began.

This file also owns every remaining pixel read the pipeline needs
------------------------------------------------------------------
02 used to compute each region's background colour and uniformity (bg_bgr/
bg_std) and hand them down through 03_extract.py untouched, purely so this
file could read them. Neither 02 nor 03_extract.py ever DID anything with
those values themselves
-- 02 only needed colour internally, as a merge-time similarity test, from
one single "middle of the sampled set" frame. That plumbing bought nothing:
this file is the only consumer, so it now opens the match video itself and
samples colour directly, from SEVERAL frames spread across the window
instead of trusting one. Icon identification (07, briefly a separate file)
folded in for the same reason -- it also just needed a frame near an
already-validated counter box, which this file can get itself instead of
another stage re-deriving "which regions are counters" from this file's own
output. 02 keeps its OWN internal colour sampling for merge decisions
(background_stats), which is a different question (do two pockets belong to
one field?) answered before this file has anything to identify.

Algorithm
---------
1. PARSE each region's raw strings. A separator makes the structure explicit:
   "1:53" is a two-part value, "50/100" is a two-part value, "245" is not.
   Nothing here guesses which part is which yet.

2. MATCH WINDOW, from the clock and nothing else. Within a period the clock
   satisfies k = value + t_sec = constant, so grouping a region's reads by
   constant k recovers the periods directly. The region with the most
   countdown structure IS the timer; the match ends at the largest k over its
   runs and starts at the first sample of its earliest run.

   This is measured, not assumed, and it works where audio does not: match3
   is 2025 REEFSCAPE and 01_audio.py has no cue profile for it, so it has no
   phases file at all, yet its clock puts match_end at 159.8s from a
   143-sample run. Where 01 does have marks the two agree to within -0.9 to
   +1.9s. `timing_delta` reports that difference as a free cross-modal check.

   Largest k rather than longest run: where 02 truncated a clock box to the
   seconds (match7, match9) teleop is chopped into minute-length segments,
   and it is the LAST of them that ends at the buzzer.

3. CLASSIFY each region from features measured over the window: does it
   count down, does it change at all, is it integer-valued, does it carry a
   separator, where does it sit. Junk regions -- live camera content that
   happened to decode -- fail on stability, not on any geometric test.

4. SAMPLE COLOUR, but only for regions that already look like scorebug
   candidates (numeric kind, sitting in the clock's band) -- there is no
   reason to open the video for a region already excluded on cheaper, purely
   textual grounds. A handful of frames spread across the match window are
   read once and reused for every candidate: each candidate's chip colour is
   the MEDIAN of its background ring across those frames (guards against one
   frame catching a mid-transition or lighting flicker), and its chip
   uniformity (bg_std) is the MEAN of each frame's own ring spread -- that
   second number is what a single frame could already tell you (an
   individual reading's ring is flat or it isn't), just measured several
   times instead of once. Confirmed necessary, not theoretical: match1's
   "red_counter[1]" was a static arena banner (a shadow line across a yellow
   sign, same at three points spread across the whole match -- checked by
   pulling the frames and looking) that happened to template-match a few
   digits and land in the scorebug band. Its ring is loud -- bg_std=45.44 --
   against 0.69-6.02 for every genuinely real field across all 9 matches on
   hand. Nothing before this step looks at whether the background is
   actually a flat chip, so nothing before this step could have caught it.

5. ASSIGN roles. Colour gives the alliance, position separates a central
   score from an edge counter, and behaviour separates a timer from a score
   from a progress fraction. All three are needed and none suffices alone:
   match6 has red on the left (position alone fails), 2026's badge tracks the
   score closely and both are monotone integers on the same-coloured chip
   (behaviour alone fails), and colour says which alliance but not which
   field.

6. IDENTIFY ICONS for every counter[i] role just assigned: search a band
   beside its box (both sides -- no "icon is always on the left" assumption
   any more than "blue is always on the left") for a white square, using one
   of the frames already read in step 4. Threshold it into a binary mask and
   match by IoU against data/icon_templates/*.png (built by
   build_icon_templates.py from officially-sourced reference art, not
   harvested from broadcast video -- the icon set is small, fixed, and
   published). Below MIN_ICON_IOU or MIN_ICON_MARGIN, the icon is left
   unassigned with a reason rather than guessed -- same "decline rather than
   name wrongly" rule as every other role here. Measured on match1 (fuel,
   the single 2026 icon) and match3 (coral vs algae, both 2025 icons, both
   counters on both alliances): correct matches scored 0.565-0.789 and beat
   the runner-up by 0.23-0.38.

Score decreases are NOT errors
------------------------------
FRC scores legitimately flicker and revert live -- referees undo scoring
actions, and the overlay shows it. Confirmed directly in match3, whose
overlay reads 248/250/250/248/250 across consecutive frames, verified frame
by frame against the video. Nothing here treats a decrease as suspicious,
and an earlier version of this pipeline had a jump-rejection filter that
would have silently rewritten exactly those genuine corrections.

Output
------
data/<match>_match.json:
{
  "match":.., "window": {"start_sec":.., "end_sec":.., "source":"clock",
                          "timing_delta_vs_audio":..},
  "fields": {"clock": {"region":"r04", "events":[...]},
             "blue_score": {...}, "red_score": {...},
             "blue_counters": [{"region":.., "events":[...], "icon":"fuel",
                                 "icon_iou":.., "icon_margin":..}, ...],
             "red_counters": [...]},
  "regions": [ per-region features and the role assigned, including the
               ones deliberately left unassigned ]
}

Known limitations / unvalidated
--------------------------------
- Role assignment is rule-based over measured features, not learned, and the
  rules encode 2026/2025 FRC scorebug conventions. A materially different
  scorebug (match8 is one: a different style with a "3/6" match counter)
  will produce regions this file declines to name rather than names wrongly,
  which is the intended failure direction but is still a failure.
- The clock is required. A broadcast with no readable clock has no window,
  and this file will say so rather than guess.
- MAX_CHIP_BG_STD and the icon-search constants (SEARCH_*_MULT, ASPECT_*,
  MIN/MAX_AREA_FRAC, MIN_ICON_IOU/MARGIN) are measured on a handful of real
  matches, not swept.
- Colour/icon sampling reads N_COLOR_SAMPLES frames spread across the match
  window and assumes that's enough to catch a mid-transition or occluded
  chip; if a broadcast production animates or obscures the scorebug for a
  large fraction of the match, this could still land every sample on a bad
  frame.

Usage
-----
  python pipeline/04_identify.py --match match1 --save
"""

import argparse, collections, importlib.util, json, pathlib, sys

import cv2
import numpy as np

import build_icon_templates as icon_templates

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
TEMPLATES_DIR = DATA_DIR / "icon_templates"


def _load_detect_overlay():
    spec = importlib.util.spec_from_file_location(
        "detect_overlay", pathlib.Path(__file__).parent / "02_detect_overlay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# background_stats only -- the ring-sampling math, reused rather than
# duplicated. Nothing else of 02's is needed here.
detect_overlay = _load_detect_overlay()

# Longest displayable FRC period (2:15 plus slack). A parsed clock above this
# is not a clock reading.
MAX_CLOCK_SEC = 180

# Countdown-run grouping. Each read is compared against the run's RUNNING
# MEAN k, not the previous read: a STATIC display (the teleop clock sitting
# at 2:20 between periods) has k increasing by exactly the sample interval,
# so neighbour-to-neighbour it is indistinguishable from a countdown and
# would chain forever. Against a running mean it drifts out within a few
# samples and never reaches MIN_COUNTDOWN_RUN.
COUNTDOWN_TOL_SEC = 1.5
MIN_COUNTDOWN_RUN = 5
MIN_COUNTDOWN_SPAN_SEC = 5
# A countdown must actually descend: value lost per second of wall clock. A
# real countdown sits at ~1.0; a STATIC display that survived grouping sits at
# 0. The threshold is a rate, not an absolute second count -- an absolute one
# scales wrongly with run length and rejected match2's entire 141-second
# teleop run for being 2.06s off when the allowance was 2.0s.
MIN_COUNTDOWN_RATE = 0.5

# Chip colour -> alliance. Hue in OpenCV's 0-179 scale.
HUE_BLUE = (100, 135)
HUE_RED_LO = (0, 10)
HUE_RED_HI = (168, 179)
SAT_MIN_COLOR = 90
VAL_MIN_COLOR = 60

# A region counts as "central" (score-like) if its centre sits within this
# fraction of the frame width from the middle; "edge" if outside it. The FRC
# scorebug puts alliance totals beside the clock and per-alliance counters
# out at the margins.
CENTRAL_FRAC = 0.25

# The scorebug is one horizontal band, and the clock -- located by behaviour,
# not by position -- anchors it. A region whose vertical centre falls outside
# that band is part of some other graphic and gets no role, however
# number-like it behaves. This is what stops a lower-third team panel from
# being named a score: match2 has a blue region at y=137 reaching 868, which
# beat the real blue score (y=77-116, max 403) under a "highest value wins"
# rule. Expressed in clock glyph heights so it travels across overlay scales.
BAND_PAD_GLYPH_H = 0.75

# How many frames, spread evenly across the match window, colour/icon
# sampling reads. Shared across every candidate region -- read once, reused,
# not one read per region.
N_COLOR_SAMPLES = 5

# A real scorebug chip is a flat colour. Measured with THIS file's own
# multi-frame sampling (mean per-frame ring std, pooled over
# N_COLOR_SAMPLES frames -- see sample_chip) across every genuinely-named
# field on all 9 matches on hand: 1.38-14.54, the one outlier being
# match10's red_counter[0] at 14.54 (a real field, just a noisier ring than
# the rest -- confirmed by checking its values: a clean 1..62 count).
# match1's red_counter[1] was actually a static arena banner (a shadow line
# across a yellow sign) that happened to template-match a few digits by
# coincidence and land in the scorebug band -- bg_std=44.74 under this same
# multi-frame method, confirmed by pulling the raw frame at three points
# across the match and looking at it, not by threshold alone. This number
# is NOT the single-frame threshold 02 used to compute this from (that was
# 0.69-6.02 vs 45.44) -- averaging several frames' ring noise runs a bit
# higher than trusting one lucky frame, so the threshold moved with the
# measurement. Set well above the real-field ceiling and well below the one
# confirmed junk case; not swept beyond that.
MAX_CHIP_BG_STD = 20.0

# The clock reaching zero is the buzzer, but the scorebug keeps settling for a
# moment afterwards and starts a moment before the countdown's first sampled
# reading. Both margins are measured, not guessed:
#
#   LEAD  - the countdown's first sample lands 0.6-1.7s after the audio start
#           cue, because the clock is not sampled at the instant it appears.
#           2s recovers the pre-match zero state on every match here.
#   TAIL  - the last genuine value change sits at most 1.38s after the clock
#           hits zero (match1's blue score, at 171.80s against a clock zero of
#           170.42s). The nearest contamination is match2's post-match scene
#           wipe at +4.78s. So the viable band is [1.38, 4.78] and 3.0 sits
#           near its middle. Measuring against the CLOCK rather than the audio
#           buzzer is what makes that band comfortable -- against audio the
#           same band is only [2.90, 4.71].
#
# These live here, in the semantic layer, rather than in 03_extract.py: they describe how
# a scorebug behaves around a buzzer, which is exactly the kind of convention
# this file exists to hold.
WINDOW_LEAD_SEC = 2.0
WINDOW_TAIL_SEC = 3.0

# --- icon search (see identify_icon below) ----------------------------------
# Near-white test for locating an icon's white square in a live frame -- same
# idea as hue_class's "white" bucket, tuned separately (measured on real
# broadcast crops, not build_icon_templates.py's clean sourced art).
ICON_WHITE_VAL_MIN = 160
ICON_WHITE_SAT_MAX = 40
# How far beside the counter box to search, in multiples of the counter
# box's own height. Measured on match1 (ch=22): the true icon square (50px)
# wasn't fully visible until pad_x >= ~6.4x and pad_y >= ~2.3x; both carry
# margin above that.
SEARCH_WIDTH_MULT = 8.0
SEARCH_HEIGHT_MULT = 2.0
# Breaks thin (1-2px) bridges between the icon square and a neighbouring
# bright graphic -- measured necessary on match1, where the raw mask
# connects the icon square to an unrelated yellow arrow a few px away.
SEARCH_OPEN_KERNEL = np.ones((3, 3), np.uint8)
# Candidate filtering: roughly square, not tiny (a stray text glyph or
# antialiasing fleck), and not implausibly large -- MAX_AREA_FRAC rejects a
# failure measured on match4: its counter sits close enough to the frame
# edge that the search window clips before reaching a colour boundary, and
# a bright background (empty bleacher seats) reads as more near-white
# pixels contiguous with the true icon square, merging into one component
# that fills the whole window (~14x a real icon's area there). A real icon
# square measured 2.7-3x counter_height**2 on match1; 6x leaves margin
# above that while still rejecting a window-filling blob.
ICON_ASPECT_LO, ICON_ASPECT_HI = 0.6, 1.6
ICON_MIN_AREA_FRAC = 0.5  # of counter_height ** 2
ICON_MAX_AREA_FRAC = 6.0
# Past the antialiased blend ring at the crop's edge -- see
# build_icon_templates.py's CROP_INSET for the same issue at a larger
# scale. Smaller here because these live crops are smaller (~45px vs the
# templates' ~60-100px).
ICON_CROP_INSET = 2
ICON_MATCH_SIZE = 64
MIN_ICON_IOU = 0.4
MIN_ICON_MARGIN = 0.15


# ---------------------------------------------------------------------------
# Parsing -- structure only, no meaning
# ---------------------------------------------------------------------------

def parse_raw(raw: str):
    """-> (kind, parts, sep) where kind is 'int', 'pair' or None.

    sep is the literal separator glyph decoded ('/' or ':'), not a guess --
    it is what tells as_clock_seconds a pair is mm:ss rather than a
    numerator/denominator. Discarding it and inferring the meaning from the
    tail's magnitude instead is what made "3/6" (match8's match counter)
    indistinguishable from a clock reading "0:06": both have parts[1] < 60.
    """
    if not raw:
        return None, None, None
    for sep in ":/":
        if sep in raw:
            head, _, tail = raw.partition(sep)
            if head.isdigit() and tail.isdigit():
                return "pair", (int(head), int(tail)), sep
            if tail.isdigit() and head == "":
                return "pair", (0, int(tail)), sep
            return None, None, None
    if raw.isdigit():
        return "int", (int(raw),), None
    return None, None, None


def as_clock_seconds(kind, parts, sep):
    """A two-part value read as mm:ss, if the separator says it is one."""
    if kind == "pair" and sep == ":" and parts[1] < 60:
        return parts[0] * 60 + parts[1]
    if kind == "int" and parts[0] <= MAX_CLOCK_SEC:
        return parts[0]
    return None


# ---------------------------------------------------------------------------
# Countdown structure
# ---------------------------------------------------------------------------

def countdown_runs(samples):
    """samples: [(t_sec, seconds)] -> list of runs, each a list of samples
    that share a constant k = seconds + t_sec and actually count down."""
    runs, cur, mean_k = [], [], 0.0
    for t, v in samples:
        k = v + t
        if cur and abs(k - mean_k) <= COUNTDOWN_TOL_SEC:
            cur.append((t, v))
            mean_k += (k - mean_k) / len(cur)
        else:
            runs.append(cur)
            cur, mean_k = [(t, v)], k
    runs.append(cur)
    good = []
    for r in runs:
        if len(r) < MIN_COUNTDOWN_RUN:
            continue
        span = r[-1][0] - r[0][0]
        drop = r[0][1] - r[-1][1]
        # A real countdown loses about one second of value per second of wall
        # clock; this is what rejects a static display that survived grouping.
        if span < MIN_COUNTDOWN_SPAN_SEC or drop < MIN_COUNTDOWN_RATE * span:
            continue
        good.append(r)
    return good


def k_of(run):
    return sum(v + t for t, v in run) / len(run)


# ---------------------------------------------------------------------------
# Region features
# ---------------------------------------------------------------------------

def hue_class(bgr):
    if bgr is None:
        return "unknown"
    px = np.uint8([[bgr]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(h), int(s), int(v)
    if s < 60 and v > 150:
        return "white"
    if s < SAT_MIN_COLOR or v < VAL_MIN_COLOR:
        return "dark"
    if HUE_BLUE[0] <= h <= HUE_BLUE[1]:
        return "blue"
    if HUE_RED_LO[0] <= h <= HUE_RED_LO[1] or HUE_RED_HI[0] <= h <= HUE_RED_HI[1]:
        return "red"
    return f"other({h})"


def region_features(rid, meta, events, img_w, window=None):
    ev = sorted(events, key=lambda e: e["t_sec"])
    if window:
        ev = [e for e in ev if window[0] <= e["t_sec"] <= window[1]]
    parsed = [(e["t_sec"], *parse_raw(e["raw"])) for e in ev]
    ok = [(t, k, p, s) for t, k, p, s in parsed if k]
    ints = [(t, p[0]) for t, k, p, s in ok if k == "int"]
    pairs = [(t, p) for t, k, p, s in ok if k == "pair"]
    clock_samples = [(t, as_clock_seconds(k, p, s)) for t, k, p, s in ok
                     if as_clock_seconds(k, p, s) is not None]
    # A pair whose separator is NOT ':' (a progress fraction like "105/240")
    # is not a clock reading -- as_clock_seconds rejects it, so without this
    # branch its values list stays permanently empty. The comparable scalar
    # is the NUMERATOR, the part that actually counts up -- the denominator
    # is normally a fixed target.
    fraction_samples = [(t, p[0]) for t, k, p, s in ok
                        if k == "pair" and as_clock_seconds(k, p, s) is None]
    runs = countdown_runs(clock_samples) if clock_samples else []

    x0, y0, x1, y1 = meta["box"]
    cx = (x0 + x1) / 2
    # For a two-part value the comparable scalar is the whole reading, not
    # the first half -- otherwise a clock reports its MINUTES as its range.
    # Which stream to trust is a MAJORITY vote (mirrors the n_pair > n_int
    # split used for "kind" below), not "int wins if any exist": a single
    # OCR frame that drops a separator glyph and reads e.g. "12/20" as the
    # int "1220" must not erase 199 good pair reads and flatten n_changes to
    # 0 -- that silently misclassified real fraction/counter fields as
    # "static" in match2, match4, match6, match8 and match10's saved data.
    if len(pairs) > len(ints):
        values = ([v for _, v in clock_samples] if len(clock_samples) >= len(fraction_samples)
                  else [v for _, v in fraction_samples])
    else:
        values = [v for _, v in ints]
    changes = sum(1 for a, b in zip(values, values[1:]) if a != b)
    f = {
        "region": rid, "box": meta["box"], "glyph_h": meta["glyph_h"],
        "polarity": meta["polarity"],
        # Filled in later, only for candidates that clear the cheap textual
        # filters -- see sample_chip_for and identify()'s pass 4. Every
        # region still gets these keys so output/printing never has to
        # special-case "never sampled".
        "chip": "unknown", "bg_std": None,
        "x_center_frac": round(cx / img_w, 3),
        "central": abs(cx / img_w - 0.5) <= CENTRAL_FRAC,
        "n_events": len(ev), "parse_rate": round(len(ok) / max(1, len(ev)), 3),
        "n_pair": len(pairs), "n_int": len(ints),
        "n_changes": changes,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "countdown_runs": len(runs),
        "countdown_span": round(sum(r[-1][0] - r[0][0] for r in runs), 1),
        "calib_margin": meta.get("calib_margin"),
    }
    f["_runs"] = runs
    f["_ints"] = ints
    f["_pairs"] = pairs
    return f


# ---------------------------------------------------------------------------
# Pixel access -- colour and icon identity, both anchored to already-
# validated regions rather than searched for blind (see module docstring).
# ---------------------------------------------------------------------------

def read_sample_frames(video_path, fps, window, n=N_COLOR_SAMPLES):
    """Read n frames evenly spread across window once, for every candidate
    region to reuse -- not one read per region."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames = []
    for t in np.linspace(window[0], window[1], n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def sample_chip(frames, box):
    """-> (median_bgr, mean_std) pooled across `frames`, or (None, None).
    median_bgr: the chip colour, robust to any single frame catching a
    transition or lighting flicker. mean_std: the chip's own flatness,
    averaged rather than trusted from one reading -- see MAX_CHIP_BG_STD."""
    meds, stds = [], []
    for frame in frames:
        med, std = detect_overlay.background_stats(frame, box)
        if med is not None:
            meds.append(med)
            stds.append(std)
    if not meds:
        return None, None
    return np.median(np.array(meds), axis=0), float(np.mean(stds))


def find_icon_square(frame, box, side):
    """-> (x, y, w, h) of the icon candidate in FRAME's own coordinates, or
    None. side is 'left' or 'right' of box."""
    x0, y0, x1, y1 = box
    ch = y1 - y0
    cy = (y0 + y1) // 2
    pad_x = int(ch * SEARCH_WIDTH_MULT)
    pad_y = int(ch * SEARCH_HEIGHT_MULT)
    sy0, sy1 = max(0, cy - pad_y), cy + pad_y
    if side == "left":
        sx0, sx1 = max(0, x0 - pad_x), x0
    else:
        sx0, sx1 = x1, x1 + pad_x
    search = frame[sy0:sy1, sx0:sx1]
    if search.size == 0:
        return None

    b, g, r = cv2.split(search.astype(np.int32))
    mx = np.maximum(np.maximum(b, g), r)
    mn = np.minimum(np.minimum(b, g), r)
    sat = mx - mn
    white = ((mx > ICON_WHITE_VAL_MIN) & (sat < ICON_WHITE_SAT_MAX)).astype(np.uint8) * 255
    opened = cv2.morphologyEx(white, cv2.MORPH_OPEN, SEARCH_OPEN_KERNEL)
    n, _, stats, _ = cv2.connectedComponentsWithStats(opened, 8)

    sh, sw = search.shape[:2]
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (ICON_MIN_AREA_FRAC * ch * ch <= area <= ICON_MAX_AREA_FRAC * ch * ch):
            continue
        if not (ICON_ASPECT_LO <= w / h <= ICON_ASPECT_HI):
            continue
        if x == 0 or y == 0 or x + w == sw or y + h == sh:
            continue  # touches the search window's own edge -- possibly clipped
        if best is None or area > best[4]:
            best = (x, y, w, h, area)
    if best is None:
        return None
    x, y, w, h, _ = best
    return int(sx0 + x), int(sy0 + y), int(w), int(h)


def load_icon_templates():
    templates = {}
    for f in sorted(TEMPLATES_DIR.glob("*.png")):
        if f.stem.startswith("_") or f.stem.endswith("_compare"):
            continue
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        templates[f.stem] = cv2.resize(m, (ICON_MATCH_SIZE, ICON_MATCH_SIZE),
                                       interpolation=cv2.INTER_NEAREST) > 0
    return templates


def match_icon(mask, templates):
    """-> (best_name, best_iou, margin), best first."""
    qm = cv2.resize(mask, (ICON_MATCH_SIZE, ICON_MATCH_SIZE), interpolation=cv2.INTER_NEAREST) > 0
    scores = []
    for name, tm in templates.items():
        inter = (qm & tm).sum()
        union = (qm | tm).sum()
        scores.append((name, inter / union if union else 0.0))
    scores.sort(key=lambda s: s[1], reverse=True)
    best_name, best_iou = scores[0]
    margin = best_iou - (scores[1][1] if len(scores) > 1 else 0.0)
    return best_name, best_iou, margin


def identify_icon(frame, box, templates):
    """-> dict with 'icon' (name or None) plus iou/margin/box/side/reason."""
    for side in ("left", "right"):
        found = find_icon_square(frame, box, side)
        if found:
            break
    else:
        return {"icon": None, "reason": "no square candidate found either side"}

    x, y, w, h = found
    crop = frame[y + ICON_CROP_INSET:y + h - ICON_CROP_INSET,
                 x + ICON_CROP_INSET:x + w - ICON_CROP_INSET]
    mask = icon_templates.ink_mask(crop)
    name, iou, margin = match_icon(mask, templates)
    entry = {"box": [x, y, x + w, y + h], "side": side,
             "iou": round(float(iou), 3), "margin": round(float(margin), 3)}
    if iou < MIN_ICON_IOU or margin < MIN_ICON_MARGIN:
        entry["icon"] = None
        entry["reason"] = f"best={name}@{iou:.2f} below MIN_ICON_IOU/MARGIN"
    else:
        entry["icon"] = name
    return entry


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------

def identify(match: str) -> dict:
    tl_path = DATA_DIR / f"{match}_regions_timeline.json"
    doc = json.loads(tl_path.read_text())
    regions, events = doc["regions"], doc["events"]
    reg_doc = json.loads((DATA_DIR / f"{match}_regions.json").read_text())
    img_w = reg_doc["img_w"]

    by_region = collections.defaultdict(list)
    for e in events:
        by_region[e["region"]].append(e)

    # --- pass 1: unwindowed, only to find the clock ------------------------
    pre = {rid: region_features(rid, meta, by_region[rid], img_w)
           for rid, meta in regions.items()}
    timer = max(pre.values(), key=lambda f: (f["countdown_span"], f["countdown_runs"]),
                default=None)
    if timer is None or not timer["_runs"]:
        return {"match": match, "error": "no region shows countdown structure -- "
                                         "cannot locate the match without a clock",
                "regions": [{k: v for k, v in f.items() if not k.startswith("_")}
                            for f in pre.values()]}
    start = min(r[0][0] for r in timer["_runs"])
    end = max(k_of(r) for r in timer["_runs"])

    phases = {}
    p = DATA_DIR / f"{match}_phases.json"
    if p.exists():
        try:
            phases = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            phases = {}
    audio_end = (phases.get("phases") or {}).get("match_end")
    audio_start = (phases.get("phases") or {}).get("auto_start")

    window = (max(0.0, start - WINDOW_LEAD_SEC), end + WINDOW_TAIL_SEC)

    # --- pass 2: features measured INSIDE the window -----------------------
    feats = {rid: region_features(rid, meta, by_region[rid], img_w, window)
             for rid, meta in regions.items()}

    # --- classify ------------------------------------------------------
    for rid, f in feats.items():
        if rid == timer["region"]:
            f["kind"] = "timer"
        elif f["parse_rate"] < 0.8 or f["n_events"] < 2:
            f["kind"] = "unstable"
        elif f["n_changes"] == 0:
            f["kind"] = "static"
        elif f["n_pair"] > f["n_int"]:
            f["kind"] = "fraction"
        else:
            f["kind"] = "counter"

    ty0, ty1 = regions[timer["region"]]["box"][1], regions[timer["region"]]["box"][3]
    pad = BAND_PAD_GLYPH_H * regions[timer["region"]]["glyph_h"]
    band = (ty0 - pad, ty1 + pad)
    for f in feats.values():
        cy = (f["box"][1] + f["box"][3]) / 2
        f["in_scorebug_band"] = band[0] <= cy <= band[1]

    # --- pass 3: colour, only for cheap-filtered candidates -----------------
    color_frames = []
    candidates = [f for f in feats.values()
                  if f["kind"] in ("counter", "fraction") and f["in_scorebug_band"]]
    video_path = ROOT.parent / f"{match}.mp4"
    if candidates and video_path.exists():
        color_frames = read_sample_frames(video_path, doc["fps"], window)
    for f in candidates:
        chip_bgr, chip_std = sample_chip(color_frames, f["box"])
        f["chip"] = hue_class(chip_bgr)
        f["bg_std"] = round(chip_std, 2) if chip_std is not None else None

    # --- assign roles --------------------------------------------------
    named = {"clock": timer["region"]}
    numeric = [f for f in feats.values()
               if f["kind"] in ("counter", "fraction") and f["chip"] in ("blue", "red")
               and f["in_scorebug_band"] and (f["bg_std"] or 0) <= MAX_CHIP_BG_STD]
    for alliance in ("blue", "red"):
        side = [f for f in numeric if f["chip"] == alliance]
        central = [f for f in side if f["central"]]
        edge = [f for f in side if not f["central"]]
        # The alliance total is the central number on that alliance's chip;
        # where several qualify, the one that reaches the highest value.
        if central:
            best = max(central, key=lambda f: (f["max"] or 0))
            named[f"{alliance}_score"] = best["region"]
            best["role"] = f"{alliance}_score"
        for i, f in enumerate(sorted(edge, key=lambda f: f["box"][0])):
            f["role"] = f"{alliance}_counter[{i}]"
        named[f"{alliance}_counters"] = [f["region"] for f in
                                         sorted(edge, key=lambda f: f["box"][0])]
    feats[timer["region"]]["role"] = "clock"

    # --- pass 4: icon identity, only for counter[i] roles just assigned -----
    icon_frame = color_frames[len(color_frames) // 2] if color_frames else None
    if icon_frame is not None:
        templates = load_icon_templates()
        for f in feats.values():
            if "_counter[" in f.get("role", ""):
                f["icon_info"] = identify_icon(icon_frame, f["box"], templates)

    fields = {}
    for role, rid in named.items():
        if isinstance(rid, list):
            fields[role] = [{"region": r,
                             "events": [e for e in by_region[r]
                                        if window[0] <= e["t_sec"] <= window[1]],
                             **(feats[r].get("icon_info") or {})}
                            for r in rid]
        else:
            fields[role] = {"region": rid,
                            "events": [e for e in by_region[rid]
                                       if window[0] <= e["t_sec"] <= window[1]]}

    return {
        "match": match,
        "window": {"start_sec": round(window[0], 2), "end_sec": round(window[1], 2),
                   "clock_start_sec": round(start, 2), "clock_zero_sec": round(end, 2),
                   "source": "clock",
                   "audio_start_sec": audio_start, "audio_end_sec": audio_end,
                   "timing_delta_vs_audio": (round(end - audio_end, 2)
                                             if audio_end is not None else None)},
        "fields": fields,
        "regions": [{k: v for k, v in f.items() if not k.startswith("_") and k != "icon_info"}
                    for f in sorted(feats.values(), key=lambda f: f["region"])],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", required=True)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    result = identify(args.match)
    if "error" in result:
        print(f"[error] {args.match}: {result['error']}", file=sys.stderr)
    else:
        w = result["window"]
        print(f"[window] {args.match}: {w['start_sec']}-{w['end_sec']}s from clock"
              + (f" (audio match_end differs by {w['timing_delta_vs_audio']:+}s)"
                 if w["timing_delta_vs_audio"] is not None else " (no audio phases)"),
              file=sys.stderr)
        for f in result["regions"]:
            role = f.get("role", "-")
            print(f"    {f['region']} {str(f['box']):<26} {f['chip']:<8} "
                  f"{f['kind']:<9} changes={f['n_changes']:<4} "
                  f"range={f['min']}..{f['max']} -> {role}", file=sys.stderr)
        for role, v in result["fields"].items():
            if isinstance(v, list):
                for entry in v:
                    last = entry["events"][-1]["raw"] if entry["events"] else "-"
                    icon = f" icon={entry['icon']}" if entry.get("icon") else (
                           f" icon=? ({entry['reason']})" if "reason" in entry else "")
                    print(f"[field] {role:<16} {entry['region']}  final={last}{icon}", file=sys.stderr)
            else:
                last = v["events"][-1]["raw"] if v["events"] else "-"
                print(f"[field] {role:<16} {v['region']}  final={last}", file=sys.stderr)

    if args.save:
        out = DATA_DIR / f"{args.match}_match.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"[save] -> {out}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
