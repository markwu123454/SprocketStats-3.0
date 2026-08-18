#!/usr/bin/env python3
"""
Step 3 -- Calibration, part 1: harvest real digit-glyph crops from broadcast
video, keyed by which digit they actually render, so a later step
(04_build_templates.py) can fuse many low-res instances of the same digit
into one higher-quality template. This file only COLLECTS; it does not
build templates itself -- kept as a separate step because collection is
video/OCR-bound (slow, one pass per match) while template-building is pure
image processing over already-saved crops (fast, cheaply re-run as more
instances accumulate). Same "split by cadence, not by CV technique"
rationale as the rest of this pipeline (see README).

Why "every instance", not just "the first clean read of each digit"
---------------------------------------------------------------------
Requested directly in chat: the overlay's font is not monospaced, so a
digit's exact sub-pixel horizontal position shifts slightly whenever ANY
digit in the same field changes -- not just when that digit's own value
changes (e.g. going from "19" to "20": the "2" occupies a different exact
pixel phase than "1" did, even though neither digit is new to the bank,
because the glyph widths differ and the field's rendering re-flows).
Different sub-pixel phases alias differently when the source is only
double-digit-pixels tall, so many phase-diverse low-res instances of the
same glyph contain MORE combined information than any single instance --
classic multi-frame super-resolution setup. This is why capture triggers on
"the whole field's rendered pixels changed at all" (see _looks_changed),
not "this specific digit's OCR'd value changed": every glyph visible at a
settled frame after ANY change gets grabbed, whether or not its own digit
value differs from last time.

Algorithm
---------
1. Walk each match's video sequentially (decode only every STRIDE-th frame
   via cap.grab() on the rest -- full per-frame decode isn't needed since
   clock/score changes settle over multiple frames at broadcast framerates,
   not within one).
2. For each numeric field in that match's already-detected boxes (from
   02_detect_overlay.py's saved *_overlay_boxes.json -- this file does not
   redo detection, it consumes it), track the field's crop pixels frame to
   frame. Two consecutive SAMPLED frames reading the same crop (within
   noise) means the field has settled into a new (or unchanged) stable
   state -- this is what "instance" means, and it's also what protects
   against capturing a mid-transition animation frame if the overlay
   animates digit swaps instead of snapping instantly.
3. On a newly-settled frame that differs from the last captured one for
   that field, segment the crop into individual glyph boxes (connected
   components of the FOREGROUND color -- see polarity below) and separately
   OCR the whole crop for a digit-string read. If the component count
   doesn't match the OCR digit count, the frame is skipped for that field
   (no confident per-glyph attribution) rather than guessed at -- this can
   happen mid-animation even after the 2-consecutive-frame settle check, or
   on an OCR misread.
4. Each matched (glyph box, digit char) pair is saved as one instance, to
   data/digit_instances/<kind>/<digit>/<match>_<field>_f<frame>_s<slot>.png,
   plus one manifest.jsonl row.

Foreground/background polarity (empirically confirmed, not assumed)
---------------------------------------------------------------------
blue_score/red_score/*_badges: white digits on a solid alliance-color chip
(confirmed by 02_detect_overlay.py's own color classification, which is
exactly how these boxes got found in the first place). clock: BLACK digits
on a solid WHITE chip -- the OPPOSITE polarity, confirmed by looking
directly at a real clock crop (see broadcast/data/font_check/match1_clock_
real.png from the font-identification check earlier this session) -- "0:42"
renders as dark strokes on white, not white-on-dark like the scores. This
is why segmentation needs a `kind`-specific foreground mask, not one
color rule reused everywhere.

Grouped into 3 `kind`s for template-building, NOT one shared bank, because
badge digits are visibly smaller (~20px tall in a 1920x1080 frame) than the
main score digits (~40px tall) -- different rendering scale entirely, even
if it's nominally "the same font". blue_score and red_score DO get pooled
together under "score" -- same size, same white-on-color rendering, only
the background hue differs, which the glyph shape doesn't depend on.

Known limitations / unvalidated
--------------------------------
- STRIDE, the change-detection thresholds, and the noise-floor component
  area are starting points, not swept -- check actual instance counts and
  a few sampled crops before trusting this blindly on a new match.
- No attempt yet to reject a field that's showing unrelated bookend content
  (a title card, post-match b-roll) the way 02's initial-value check does --
  bounded by --end-frac same as 02's SAMPLE_END_FRAC default for the same
  reason (dodge the tail), but a contaminated MIDDLE frame isn't caught.
  Relies on the connected-component-count-vs-OCR-count agreement check to
  reject most garbage implicitly (unrelated content rarely happens to
  segment into exactly N glyphs matching an N-digit OCR read), not a
  guarantee.
- match8 has NO usable boxes at all (confirmed separately, not a bug in
  this file or 02_detect_overlay.py): the video is trimmed to start already
  mid-auto (RED 1 / BLUE 8 / clock 0:17 visible at frame 0), so there is no
  pre-match zero anywhere in the file for 02's disambiguation to confirm
  against. Skipped by user decision rather than building a fallback
  confirmation strategy for it right now.

Usage
-----
  python pipeline/03_calibrate.py --matches match1 match2 match3 match4 match6 match7 match9 match10 --save

Install: pip install opencv-python numpy easyocr torch
"""

import argparse, json, pathlib, sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
INSTANCES_DIR = DATA_DIR / "digit_instances"

DEFAULT_MATCHES = ["match1", "match2", "match3", "match4", "match6", "match7", "match9", "match10"]

# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

# Every STRIDE-th frame is decoded, the rest are cap.grab()'d (cheap skip,
# no decode) -- clock/score transitions settle over multiple frames at
# broadcast framerates (30-60fps), so per-frame decode is wasted work for
# this purpose specifically, unlike 02's temporal-activity sampling which
# needs a spread-out but sparse set of frames for a different statistic.
STRIDE = 4

# Same reasoning as 02_detect_overlay.py's SAMPLE_END_FRAC: dodges known
# post-match bookend content (title cards, celebration b-roll -- confirmed
# present on match2). Unlike 02, START defaults to the true beginning: this
# script doesn't need a pre-match zero to anchor against (02 already solved
# that disambiguation problem; this file trusts 02's saved boxes instead of
# re-deriving them), so there's no reason to skip early frames here.
START_FRAC = 0.0
END_FRAC = 0.85

# ---------------------------------------------------------------------------
# Change / settle detection
# ---------------------------------------------------------------------------

# Fraction of crop pixels that must differ by more than CHANGE_PIXEL_DELTA
# for two crops to count as "different" -- deliberately NOT a raw mean-diff
# threshold (a whole-crop mean washes out a single narrow digit's change in
# a wide multi-digit crop) and NOT "any pixel differs" (compression noise
# flickers a handful of pixels even on a genuinely static crop). Starting
# point, unswept -- see module docstring.
CHANGE_PIXEL_DELTA = 25
CHANGE_FRAC_THRESH = 0.01

# ---------------------------------------------------------------------------
# Glyph segmentation (per-kind foreground polarity)
# ---------------------------------------------------------------------------

# Originally reused verbatim from 02_detect_overlay.py's own confirmed-by-eye
# ranges (that file still uses 60/170 for its own, different job of finding
# the chip BOXES, not per-glyph masks -- untouched there). LOOSENED slightly
# here (60->70, 170->160) after a real fragmentation bug traced to its root
# cause directly in chat: badge digits render only ~20px tall, and at that
# scale a ring digit's ("0"/"6"/"8"/"9") antialiased blend toward the
# alliance-color background sometimes lands in a band that's bright enough
# (val>200) but still too saturated (sat~90-105) to pass the old sat<=60
# cutoff. Confirmed on match1 blue_badges[0] frame 760 (true value "20",
# read as "743"): per-column HSV sampling showed mask column 27 -- dead
# center of the "0"'s ring -- was FULLY EMPTY across all 22 rows under the
# old thresholds, splitting one digit into two connected components.
#
# NOT loosened further, on purpose, despite that gap needing sat~90-105 to
# fully close: tried it (up to sat<=100/val>=130), and at that strength it
# ALSO bridges the genuinely narrow (2px) gap between two ADJACENT digits in
# a different real frame (blue_badges[0] f2856, "100" -- the gap between its
# two "0"s is tighter than the gap this fix targets), merging them into one
# oversized blob. 70/160 is the loosest setting that closes the ring-gap
# (confirmed on f760 and a second real fragmentation frame, f3022) without
# reproducing that merge on f2856 -- checked by sweeping sat 60->100 against
# both frames together, not picked from one side alone.
#
# This does NOT fully fix classification on the frames it segments
# correctly, though -- known limitation, not solved here: f760/f3022/f2856
# all now produce the right GLYPH COUNT, but per-glyph match confidence on
# these ~20px badge digits is still weak (worst scores 0.45-0.65, several
# below MIN_GLYPH_CONFIDENCE), so raw classification can still land on the
# wrong digit even once segmentation finds the right box. That's a separate
# problem from fragmentation (mask threshold vs. template-match quality) --
# see 05_extract.py's low_confidence flag and jump-rejection, which are
# production's existing defense against exactly this residual noise.
SAT_MAX_WHITE = 70
VAL_MIN_WHITE = 160
# NEW for this file: clock digits are dark strokes on the white chip, the
# opposite polarity from score/badge digits -- confirmed against a real
# clock crop (see module docstring). Starting point, not measured as
# precisely as VAL_MIN_WHITE was, and NOT part of the SAT/VAL investigation
# above -- clock's own fragmentation risk is unverified either way.
VAL_MAX_DARK = 110

# Noise floor for a glyph connected component, in raw pixels (not
# frame-area-fraction like 02's pocket search -- these crops are already
# small, a handful of pixels is definitely noise regardless of frame size).
MIN_GLYPH_AREA_PX = 6

# Geometry validation applied to every component that survives the area
# filter above -- added directly from chat after two real garbage badge
# reads were traced to their source frames (match1 blue_badges[0] f760 "20"
# read as "743", f3022 "120" read as "1761"): in both, at least one
# component was a genuine SUB-FRAGMENT of a single digit (part of a ring or
# stroke), not a full glyph -- exactly what the SAT/VAL loosening above
# targets, but kept as a second, independent line of defense for whatever
# fragmentation that fix doesn't catch (a different digit, a different
# match's rendering scale, motion blur).
#
# The height check compares each component to the FIELD'S OWN BOX height,
# not to other components found in the same frame (that was the old
# SEPARATOR_HEIGHT_RATIO approach, and its exact failure mode: if EVERY
# component in a corrupted frame is undersized, comparing them only to each
# other never catches that -- box height is a stable per-field calibration
# value a bad frame can't drag down). It also still does the old filter's
# job of dropping the clock's ":" separator dots, which sit far below any
# reasonable fraction of box height. Values are margined below/above the
# real range measured across ~18k harvested digit_instances (score/badge/
# clock all cluster in the same band): true single-digit height sits at
# roughly 0.65-1.0x box height depending on kind/match rendering scale, and
# width/height ratio runs ~0.41 ("1", the narrowest digit) to ~0.75 ("4",
# the widest) fairly consistently across all three kinds. A ratio above the
# max is assumed to be two touching/merged digits (no digit-splitting logic
# exists yet, so these are just rejected, not recovered) rather than one
# wide glyph -- not observed in this investigation, follows from no digit
# instance measuring anywhere close to it. Starting points from one
# investigation, not swept broadly -- worth rechecking if a match with a
# very different rendering scale starts producing rejections.
MIN_GLYPH_HEIGHT_FRAC = 0.5
MIN_GLYPH_WH_RATIO = 0.30
MAX_GLYPH_WH_RATIO = 0.95

# ---------------------------------------------------------------------------
# OCR (same approach as 02_detect_overlay.py, duplicated rather than
# imported -- 02's filename starts with a digit so it isn't a valid Python
# module name to import directly, and the two files' OCR needs are similar
# but not identical: this one reads the FULL sampled range of a match, not
# just a handful of early confirmation frames.)
# ---------------------------------------------------------------------------

OCR_UPSCALE = 4

_READER = None


def get_reader():
    global _READER
    if _READER is None:
        import easyocr
        print("[ocr] loading EasyOCR digit reader ...", file=sys.stderr)
        _READER = easyocr.Reader(["en"], gpu=True)
    return _READER


def read_digits_str(frame: np.ndarray, box: list[int], allow_colon: bool = False) -> str | None:
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    reader = get_reader()
    allowlist = "0123456789:" if allow_colon else "0123456789"
    results = reader.readtext(thresh, allowlist=allowlist, detail=1)
    if not results:
        return None
    results.sort(key=lambda r: r[0][0][0])
    text = "".join(c for c in "".join(r[1] for r in results) if c.isdigit())
    return text or None


# ---------------------------------------------------------------------------
# Glyph segmentation
# ---------------------------------------------------------------------------

def segment_glyphs(frame_bgr: np.ndarray, box: list[int], polarity: str) -> list[list[int]]:
    """Individual glyph boxes (absolute frame coords, left-to-right,
    separators dropped) inside `box`. `polarity` is 'light' (white digits
    on a colored chip -- score/badges) or 'dark' (black digits on a white
    chip -- clock)."""
    x0, y0, x1, y1 = box
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    if polarity == "light":
        mask = ((sat <= SAT_MAX_WHITE) & (val >= VAL_MIN_WHITE)).astype(np.uint8)
    else:
        mask = (val <= VAL_MAX_DARK).astype(np.uint8)

    box_h = y1 - y0
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    digits = []
    for lbl in range(1, n_labels):
        bx, by, bw, bh, barea = stats[lbl]
        if barea < MIN_GLYPH_AREA_PX:
            continue
        # See MIN_GLYPH_HEIGHT_FRAC/MIN_GLYPH_WH_RATIO/MAX_GLYPH_WH_RATIO's
        # comment above for why these compare to the box, not to the other
        # components found in this same frame.
        if bh < box_h * MIN_GLYPH_HEIGHT_FRAC:
            continue
        if not (MIN_GLYPH_WH_RATIO <= bw / bh <= MAX_GLYPH_WH_RATIO):
            continue
        digits.append([x0 + int(bx), y0 + int(by), x0 + int(bx) + int(bw), y0 + int(by) + int(bh)])
    digits.sort(key=lambda c: c[0])
    return digits


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def _looks_changed(a: np.ndarray | None, b: np.ndarray) -> bool:
    """True if crop `b` differs meaningfully from `a` -- see
    CHANGE_PIXEL_DELTA/CHANGE_FRAC_THRESH docstrings. `a is None` (first
    frame seen) counts as changed."""
    if a is None:
        return True
    if a.shape != b.shape:
        return True
    diff = cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    return float((diff > CHANGE_PIXEL_DELTA).mean()) > CHANGE_FRAC_THRESH


# ---------------------------------------------------------------------------
# Field tracker
# ---------------------------------------------------------------------------

class FieldTracker:
    """One per (match, field). Feeds sampled frames in; internally decides
    when the field has settled into a new state worth capturing (two
    consecutive sampled frames agreeing, and differing from what was last
    captured -- see module docstring for why "differing" is judged on the
    whole crop's pixels, not the OCR'd value alone)."""

    def __init__(self, name: str, box: list[int], polarity: str, kind: str, allow_colon: bool):
        self.name = name
        self.box = box
        self.polarity = polarity
        self.kind = kind
        self.allow_colon = allow_colon
        self.prev_crop = None
        self.last_captured_crop = None
        self.n_captured = 0
        self.n_skipped_mismatch = 0

    def _crop(self, frame_bgr):
        x0, y0, x1, y1 = self.box
        return frame_bgr[y0:y1, x0:x1]

    def offer(self, frame_bgr: np.ndarray, frame_idx: int, sink) -> None:
        crop = self._crop(frame_bgr)
        if crop.size == 0:
            return
        if not _looks_changed(self.last_captured_crop, crop):
            self.prev_crop = crop
            return
        if not _looks_changed(self.prev_crop, crop):
            # Two consecutive sampled frames agree -> settled. Capture.
            self._capture(frame_bgr, frame_idx, sink)
            self.last_captured_crop = crop
        self.prev_crop = crop

    def _capture(self, frame_bgr, frame_idx, sink):
        glyph_boxes = segment_glyphs(frame_bgr, self.box, self.polarity)
        value = read_digits_str(frame_bgr, self.box, allow_colon=self.allow_colon)
        if value is None or len(glyph_boxes) != len(value):
            self.n_skipped_mismatch += 1
            return
        for slot, (gbox, ch) in enumerate(zip(glyph_boxes, value)):
            sink(self.kind, ch, self.name, frame_idx, slot, gbox, frame_bgr)
        self.n_captured += 1


# ---------------------------------------------------------------------------
# Per-match driver
# ---------------------------------------------------------------------------

def build_trackers(boxes: dict) -> list[FieldTracker]:
    trackers = []
    if "blue_score" in boxes:
        trackers.append(FieldTracker("blue_score", boxes["blue_score"]["box"], "light", "score", False))
    if "red_score" in boxes:
        trackers.append(FieldTracker("red_score", boxes["red_score"]["box"], "light", "score", False))
    if "clock" in boxes:
        trackers.append(FieldTracker("clock", boxes["clock"]["box"], "dark", "clock", True))
    for side in ("blue_badges", "red_badges"):
        for i, badge in enumerate(boxes.get(side, [])):
            trackers.append(FieldTracker(f"{side}[{i}]", badge["box"], "light", "badge", False))
    return trackers


def process_match(match: str, manifest_fh, save: bool) -> None:
    boxes_path = DATA_DIR / f"{match}_overlay_boxes.json"
    video_path = ROOT.parent / f"{match}.mp4"
    if not boxes_path.exists():
        print(f"[skip] {match}: no {boxes_path.name} -- run 02_detect_overlay.py --save first", file=sys.stderr)
        return
    if not video_path.exists():
        print(f"[skip] {match}: video not found at {video_path}", file=sys.stderr)
        return

    boxes = json.loads(boxes_path.read_text())["boxes"]
    trackers = build_trackers(boxes)
    if not trackers:
        print(f"[skip] {match}: no confirmed numeric fields in {boxes_path.name}", file=sys.stderr)
        return

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo, hi = int(total * START_FRAC), int(total * END_FRAC)
    print(f"[scan] {match}: frames {lo}-{hi} of {total}, stride={STRIDE}, "
          f"{len(trackers)} field(s): {[t.name for t in trackers]}", file=sys.stderr)

    def sink(kind, digit, field, frame_idx, slot, gbox, frame_bgr):
        if not save:
            return
        out_dir = INSTANCES_DIR / kind / digit
        out_dir.mkdir(parents=True, exist_ok=True)
        gx0, gy0, gx1, gy1 = gbox
        glyph_crop = frame_bgr[gy0:gy1, gx0:gx1]
        out_name = f"{match}_{field}_f{frame_idx}_s{slot}.png"
        cv2.imwrite(str(out_dir / out_name), glyph_crop)
        manifest_fh.write(json.dumps({
            "match": match, "field": field, "frame": frame_idx, "slot": slot,
            "kind": kind, "digit": digit, "box": gbox, "file": str((out_dir / out_name).relative_to(DATA_DIR)),
        }) + "\n")

    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    for frame_idx in range(lo, hi):
        if (frame_idx - lo) % STRIDE == 0:
            ok, frame = cap.read()
            if not ok:
                break
            for t in trackers:
                t.offer(frame, frame_idx, sink)
        else:
            cap.grab()
    cap.release()

    for t in trackers:
        print(f"[done] {match}/{t.name}: {t.n_captured} instance-frames captured, "
              f"{t.n_skipped_mismatch} skipped (glyph/OCR count mismatch)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", nargs="+", default=DEFAULT_MATCHES)
    ap.add_argument("--save", action="store_true", help="write crops + manifest.jsonl (omit for a dry-run count)")
    args = ap.parse_args()

    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = INSTANCES_DIR / "manifest.jsonl"
    mode = "a" if manifest_path.exists() else "w"
    with open(manifest_path, mode) as fh:
        for match in args.matches:
            process_match(match, fh, args.save)

    print(f"[save] manifest -> {manifest_path}" if args.save else "[dry-run] nothing written (pass --save)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
