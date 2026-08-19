#!/usr/bin/env python3
"""
One-off tool (not part of the numbered pipeline) -- Calibration, part 1:
harvest real digit-glyph crops from broadcast video, keyed by which digit
they actually render, so a later step (build_templates.py, alongside this
file in tools/) can fuse many low-res instances of the same digit into one
higher-quality template. This file only COLLECTS; it does not build
templates itself -- kept as a separate step because collection is
video/OCR-bound (slow, one pass per match) while template-building is pure
image processing over already-saved crops (fast, cheaply re-run as more
instances accumulate). Same "split by cadence, not by CV technique"
rationale as the numbered pipeline (see README) -- these two steps are run
by hand, occasionally, to (re)build the digit bank the pipeline consumes,
not on every match.

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
- match8 was unusable under the OLD role-based 02/this file (no pre-match
  zero in the file for role disambiguation to confirm against) -- no longer true
  post-refactor: 02 emits regions by behaviour, not roles, so it needs no
  pre-match zero, and match8_regions_timeline.json now exists. Included in
  DEFAULT_MATCHES.

Usage
-----
  python tools/calibrate.py --matches match1 match2 match3 match4 match6 match7 match8 match9 match10 --save

Install: pip install opencv-python numpy easyocr torch
"""

import argparse, json, pathlib, sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
INSTANCES_DIR = DATA_DIR / "digit_instances"

DEFAULT_MATCHES = ["match1", "match2", "match3", "match4", "match6", "match7", "match8", "match9", "match10"]

# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

# Every STRIDE-th frame is decoded, the rest are cap.grab()'d (cheap skip,
# no decode) -- clock/score transitions settle over multiple frames at
# broadcast framerates (30-60fps), so per-frame decode is wasted work for
# this purpose specifically, unlike 02's temporal-activity sampling which
# needs a spread-out but sparse set of frames for a different statistic.
STRIDE = 4

# Structural (no-OCR) harvest of ':' and '/' samples every settled frame, not
# just once per field-value change -- unlike digit capture, which triggers
# only on a NEW value (see FieldTracker), a separator's own pixels are
# identical on every frame of a run, so "every settled frame" would mean
# hundreds of byte-identical duplicates per run. Capturing every Nth settled
# frame instead still spans multiple runs (-> multiple sub-pixel phases,
# same reason this file wants many digit instances) without the redundant I/O.
SEPARATOR_CAPTURE_EVERY = 20

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
# see pipeline/03_extract.py's low_confidence flag and jump-rejection, which are
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

# A badge/counter field can render a fraction ("105 / 240"), and the '/' is a
# full-height diagonal stroke -- tall and narrow enough (h_frac~0.85-0.97,
# w/h~0.44-0.62) to pass the digit geometry filters above and get treated as
# an (uncounted-for) extra digit slot, which is exactly the bug that made '/'
# invisible to calibration: it silently desyncs segment_glyphs' component
# count from the OCR digit string's length, so this file skips the ENTIRE frame
# rather than the one glyph (see FieldTracker._capture). Distinguished from
# every real digit by slope, not size: split a component into its top and
# bottom thirds and compare each third's mean ink column. A '/' leans hard
# left-to-right across nearly its whole height; the most diagonal real digit
# (a slanted '7') only leans in its lower portion. Measured on real harvested
# glyphs (badge+score, ~80 samples/digit) plus three real '/' instances
# spanning three matches/rendering scales (box_h 23/24/40px): every digit
# fell in [-0.332, 0.218], every '/' measured -0.51 to -0.62 -- a wide, clean
# gap. -0.40 sits in the middle of that gap, not at either edge.
SLASH_METRIC_TOP_FRAC = 0.3
SLASH_METRIC_BOT_FRAC = 0.7
SLASH_METRIC_MAX = -0.40

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

def field_mask(crop_bgr: np.ndarray, polarity: str) -> np.ndarray:
    """Ink mask for a field crop. `polarity` is 'light' (white digits on a
    colored chip -- score/badges) or 'dark' (black digits on a white chip --
    clock). Factored out of segment_glyphs so segment_separators and
    pipeline/03_extract.py's search share ONE definition of "ink" -- they all depend
    on the same SAT_MAX_WHITE/VAL_MIN_WHITE/VAL_MAX_DARK investigation, and
    a second copy would silently drift from it."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    if polarity == "light":
        return ((sat <= SAT_MAX_WHITE) & (val >= VAL_MIN_WHITE)).astype(np.uint8)
    return (val <= VAL_MAX_DARK).astype(np.uint8)


def _diagonal_metric(comp_mask: np.ndarray) -> float | None:
    """Normalized left-right lean of a component's ink, comparing its top
    30% to its bottom 70% (SLASH_METRIC_TOP_FRAC/BOT_FRAC) -- see
    SLASH_METRIC_MAX's comment for what separates a real digit from '/'.
    None if there isn't enough ink in both bands to measure."""
    h, w = comp_mask.shape
    ys, xs = np.nonzero(comp_mask)
    top_xs = xs[ys < h * SLASH_METRIC_TOP_FRAC]
    bot_xs = xs[ys > h * SLASH_METRIC_BOT_FRAC]
    if top_xs.size < 3 or bot_xs.size < 3:
        return None
    return (float(bot_xs.mean()) - float(top_xs.mean())) / w


def normalize_polarity(crop_bgr: np.ndarray, polarity: str) -> np.ndarray:
    """Saved glyph crops need ONE convention regardless of the source
    region's polarity, or build_templates.py's per-pixel median merge
    blends photographic negatives of each other. The old role-based harvest
    got this for free -- 'clock' was hardcoded dark-on-light and lived in its
    own kind dir, which build_templates.py inverted by name. Post-refactor,
    `kind` is a single pooled "all" bucket fed by whatever polarity
    pipeline/03_extract.py measured per region (see build_trackers) -- light
    and dark regions land in the SAME digit directories, so kind-name-based
    inversion in build_templates.py can no longer tell them apart.
    Normalizing HERE instead, at save time, means build_templates.py doesn't
    need to know polarity at all: every new crop is already
    bright-ink-on-dark, the same convention build_templates.py's
    clock-inversion produced, so old and new instances merge into the same
    bucket without contradicting each other."""
    return 255 - crop_bgr if polarity == "dark" else crop_bgr


def segment_glyphs(frame_bgr: np.ndarray, box: list[int], polarity: str) -> list[list[int]]:
    """Individual glyph boxes (absolute frame coords, left-to-right,
    separators AND '/' dropped -- see segment_slashes for the latter,
    harvested separately) inside `box`."""
    x0, y0, x1, y1 = box
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    mask = field_mask(crop, polarity)

    box_h = y1 - y0
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
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
        comp_mask = (labels[by:by + bh, bx:bx + bw] == lbl)
        if (_diagonal_metric(comp_mask) or 0) <= SLASH_METRIC_MAX:
            continue
        digits.append([x0 + int(bx), y0 + int(by), x0 + int(bx) + int(bw), y0 + int(by) + int(bh)])
    digits.sort(key=lambda c: c[0])
    return digits


def segment_slashes(frame_bgr: np.ndarray, box: list[int], polarity: str) -> list[list[int]]:
    """The fraction separator '/' inside `box`, as ONE box, or [] if there
    isn't one -- the digit-shaped-but-diagonal component segment_glyphs
    excludes (see SLASH_METRIC_MAX). No OCR, same reason as
    segment_separators: identified structurally, not read."""
    x0, y0, x1, y1 = box
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    mask = field_mask(crop, polarity)
    box_h = y1 - y0
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for lbl in range(1, n_labels):
        bx, by, bw, bh, barea = stats[lbl]
        if barea < MIN_GLYPH_AREA_PX or bh < box_h * MIN_GLYPH_HEIGHT_FRAC:
            continue
        if not (MIN_GLYPH_WH_RATIO <= bw / bh <= MAX_GLYPH_WH_RATIO):
            continue
        comp_mask = (labels[by:by + bh, bx:bx + bw] == lbl)
        if (_diagonal_metric(comp_mask) or 0) <= SLASH_METRIC_MAX:
            return [[x0 + int(bx), y0 + int(by), x0 + int(bx) + int(bw), y0 + int(by) + int(bh)]]
    return []


def segment_separators(frame_bgr: np.ndarray, box: list[int], polarity: str) -> list[list[int]]:
    """The non-digit glyph inside `box` -- in practice the clock's ':' -- as
    ONE box, or [] if there isn't one.

    Two decisions worth recording:

    * The returned box spans the DIGITS' row band, not the separator's own
      tight bounds. A ':' cropped tightly to its two dots carries no fixed
      relationship to the digit baseline, so a template built from it could
      not be placed by the same vertical anchor a digit is placed by --
      pipeline/03_extract.py pins every template to the field's ink top row. Padding
      the crop out to full digit height (background above and below the
      dots) makes the separator just another glyph as far as matching is
      concerned, which is the whole point of harvesting it.
    * No OCR is involved or needed. A separator is identified structurally:
      too short to be a digit, and horizontally BETWEEN two components that
      are tall enough to be digits. That's why --separators mode can run
      without importing easyocr at all.

    Components are grouped by column overlap (the two dots share columns, so
    they merge into one glyph) and the highest-area group wins, which is what
    keeps a stray speck elsewhere in the chip from being harvested instead.
    """
    x0, y0, x1, y1 = box
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    mask = field_mask(crop, polarity)
    box_h = y1 - y0
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    tall, short = [], []
    for lbl in range(1, n_labels):
        bx, by, bw, bh, barea = stats[lbl]
        if barea < MIN_GLYPH_AREA_PX:
            continue
        (tall if bh >= box_h * MIN_GLYPH_HEIGHT_FRAC else short).append(
            (int(bx), int(by), int(bw), int(bh), int(barea)))
    if not tall or not short:
        return []

    digits_left = min(c[0] for c in tall)
    digits_right = max(c[0] + c[2] for c in tall)
    row_top = min(c[1] for c in tall)
    row_bot = max(c[1] + c[3] for c in tall)

    groups = []
    for c in sorted(short, key=lambda c: c[0]):
        if not (digits_left < c[0] and c[0] + c[2] < digits_right):
            continue  # not between digits -- chip edge speck, not a separator
        if groups and c[0] <= groups[-1][1]:
            g = groups[-1]
            g[0], g[1], g[2] = min(g[0], c[0]), max(g[1], c[0] + c[2]), g[2] + c[4]
        else:
            groups.append([c[0], c[0] + c[2], c[4]])
    if not groups:
        return []
    best = max(groups, key=lambda g: g[2])
    return [[x0 + best[0], y0 + row_top, x0 + best[1], y0 + row_bot]]


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
            sink(self.kind, ch, self.name, frame_idx, slot, gbox, frame_bgr, self.polarity)
        self.n_captured += 1


# ---------------------------------------------------------------------------
# Per-match driver
# ---------------------------------------------------------------------------

def build_trackers(doc: dict) -> list:
    """Trackers for harvesting, from pipeline/03_extract.py's regions timeline.

    Reads <match>_regions_timeline.json rather than a role-keyed box file
    because 02_detect_overlay.py no longer assigns roles -- it emits regions
    and nothing else (see its module docstring for the nine wrong boxes that
    change removed). Everything this function used to hardcode is now taken
    from what 03_extract.py MEASURED per region: polarity, and the glyph height that
    tells segment_glyphs what scale it is working at.

    `kind` is now a single pooled bucket. Keeping score/badge/clock apart was
    never carrying information -- they share a typeface (cross-kind same-digit
    NCC 0.93-0.98, aspect ratios agreeing to 0.02) -- and pooling means one
    template per character instead of three near-identical ones.

    NOT RE-RUN SINCE THE REFACTOR: harvesting needs easyocr on a GPU, and the
    existing template bank under data/digit_templates/ was built before it and
    remains valid. Treat this path as untested until it is next exercised.
    """
    trackers = []
    for rid, meta in sorted((doc.get("regions") or {}).items()):
        trackers.append(FieldTracker(rid, meta["box"], meta["polarity"], "all",
                                     allow_colon=True))
    return trackers


def process_match_separators(match: str, manifest_fh, save: bool) -> None:
    """Harvest ':' (segment_separators) and '/' (segment_slashes) -- no OCR,
    no GPU for either. Kept as its own driver rather than a branch inside
    process_match because it shares nothing with the digit path except the
    video walk: there is no value to read, nothing to pair a glyph count
    against, and no reason to load easyocr. Both structural detectors run on
    every field regardless of which glyph they're actually built for --
    a clock field simply never produces a slash candidate and a badge field
    never produces a colon candidate, so there is no need to know in advance
    which fields carry which separator."""
    boxes_path = DATA_DIR / f"{match}_regions_timeline.json"
    video_path = ROOT.parent / f"{match}.mp4"
    if not boxes_path.exists() or not video_path.exists():
        print(f"[skip] {match}: missing boxes or video", file=sys.stderr)
        return

    trackers = build_trackers(json.loads(boxes_path.read_text()))
    if not trackers:
        print(f"[skip] {match}: no confirmed numeric fields", file=sys.stderr)
        return

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo, hi = int(total * START_FRAC), int(total * END_FRAC)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)

    prev = {t.name: None for t in trackers}
    settled = {t.name: 0 for t in trackers}
    n_saved = 0
    for frame_idx in range(lo, hi):
        if (frame_idx - lo) % STRIDE != 0:
            cap.grab()
            continue
        ok, frame = cap.read()
        if not ok:
            break
        for t in trackers:
            crop = frame[t.box[1]:t.box[3], t.box[0]:t.box[2]]
            if crop.size == 0:
                continue
            if prev[t.name] is not None and not _looks_changed(prev[t.name], crop):
                settled[t.name] += 1
                if settled[t.name] % SEPARATOR_CAPTURE_EVERY == 0:
                    candidates = ([("sep", sbox) for sbox in segment_separators(frame, t.box, t.polarity)] +
                                  [("slash", sbox) for sbox in segment_slashes(frame, t.box, t.polarity)])
                    for digit_name, sbox in candidates:
                        if not save:
                            n_saved += 1
                            continue
                        out_dir = INSTANCES_DIR / t.kind / digit_name
                        out_dir.mkdir(parents=True, exist_ok=True)
                        gx0, gy0, gx1, gy1 = sbox
                        crop_norm = normalize_polarity(frame[gy0:gy1, gx0:gx1], t.polarity)
                        name = f"{match}_{t.name}_f{frame_idx}_s0.png"
                        cv2.imwrite(str(out_dir / name), crop_norm)
                        manifest_fh.write(json.dumps({
                            "match": match, "field": t.name, "frame": frame_idx, "slot": 0,
                            "kind": t.kind, "digit": digit_name, "box": sbox,
                            "file": str((out_dir / name).relative_to(DATA_DIR)),
                        }) + "\n")
                        n_saved += 1
            prev[t.name] = crop
    cap.release()
    print(f"[done] {match}: {n_saved} separator instance(s)", file=sys.stderr)


def process_match(match: str, manifest_fh, save: bool) -> None:
    boxes_path = DATA_DIR / f"{match}_regions_timeline.json"
    video_path = ROOT.parent / f"{match}.mp4"
    if not boxes_path.exists():
        print(f"[skip] {match}: no {boxes_path.name} -- run pipeline/03_extract.py --save first", file=sys.stderr)
        return
    if not video_path.exists():
        print(f"[skip] {match}: video not found at {video_path}", file=sys.stderr)
        return

    trackers = build_trackers(json.loads(boxes_path.read_text()))
    if not trackers:
        print(f"[skip] {match}: no confirmed numeric fields in {boxes_path.name}", file=sys.stderr)
        return

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo, hi = int(total * START_FRAC), int(total * END_FRAC)
    print(f"[scan] {match}: frames {lo}-{hi} of {total}, stride={STRIDE}, "
          f"{len(trackers)} field(s): {[t.name for t in trackers]}", file=sys.stderr)

    def sink(kind, digit, field, frame_idx, slot, gbox, frame_bgr, polarity):
        if not save:
            return
        out_dir = INSTANCES_DIR / kind / digit
        out_dir.mkdir(parents=True, exist_ok=True)
        gx0, gy0, gx1, gy1 = gbox
        glyph_crop = normalize_polarity(frame_bgr[gy0:gy1, gx0:gx1], polarity)
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
    ap.add_argument("--separators", action="store_true",
                    help="harvest ':' and '/' only -- no OCR/GPU (see segment_separators/segment_slashes)")
    args = ap.parse_args()

    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = INSTANCES_DIR / "manifest.jsonl"
    mode = "a" if manifest_path.exists() else "w"
    driver = process_match_separators if args.separators else process_match
    with open(manifest_path, mode) as fh:
        for match in args.matches:
            driver(match, fh, args.save)

    print(f"[save] manifest -> {manifest_path}" if args.save else "[dry-run] nothing written (pass --save)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
