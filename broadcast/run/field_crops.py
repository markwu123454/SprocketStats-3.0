"""
Pixel tests on overlay crops: which pixels are ink (digit fields and game-piece
icons), and whether a crop changed since another one.

Shared by run/ (read_overlay.py, identify_fields.py -- every match) and train/
(harvest_digits.py, build_icon_templates.py -- when rebuilding the templates). It lives in run/ because run code must not import
from train/, and both sides have to agree on these definitions: the digit
templates are fused from glyphs segmented with these thresholds, so reading
with a drifted copy would compare glyphs cut a different way.
"""
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Change / settle detection
# ---------------------------------------------------------------------------

# Fraction of crop pixels that must differ by more than CHANGE_PIXEL_DELTA
# for two crops to count as "different" -- deliberately NOT a raw mean-diff
# threshold (a whole-crop mean washes out a single narrow digit's change in
# a wide multi-digit crop) and NOT "any pixel differs" (compression noise
# flickers a handful of pixels even on a genuinely static crop). Starting
# point, unswept -- see train/harvest_digits.py's module docstring.
CHANGE_PIXEL_DELTA = 25
CHANGE_FRAC_THRESH = 0.01

# ---------------------------------------------------------------------------
# Ink thresholds (per-kind foreground polarity)
# ---------------------------------------------------------------------------

# Originally reused verbatim from detect_overlay.py's own confirmed-by-eye
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
# see run/read_overlay.py's low_confidence flag and jump-rejection, which are
# production's existing defense against exactly this residual noise.
SAT_MAX_WHITE = 70
VAL_MIN_WHITE = 160
# Not in detect_overlay.py's ranges: clock digits are dark strokes on the white chip, the
# opposite polarity from score/badge digits -- confirmed against a real
# clock crop (see module docstring). Starting point, not measured as
# precisely as VAL_MIN_WHITE was, and NOT part of the SAT/VAL investigation
# above -- clock's own fragmentation risk is unverified either way.
VAL_MAX_DARK = 110


def field_mask(crop_bgr: np.ndarray, polarity: str) -> np.ndarray:
    """Ink mask for a field crop. `polarity` is 'light' (white digits on a
    colored chip -- score/badges) or 'dark' (black digits on a white chip --
    clock). Shared by train/harvest_digits.py's segmenters and
    run/read_overlay.py's search so they use ONE definition of "ink" -- they all depend
    on the same SAT_MAX_WHITE/VAL_MIN_WHITE/VAL_MAX_DARK investigation, and
    a second copy would silently drift from it."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    if polarity == "light":
        return ((sat <= SAT_MAX_WHITE) & (val >= VAL_MIN_WHITE)).astype(np.uint8)
    return (val <= VAL_MAX_DARK).astype(np.uint8)


def looks_changed(a: np.ndarray | None, b: np.ndarray) -> bool:
    """True if crop `b` differs meaningfully from `a` -- see
    CHANGE_PIXEL_DELTA/CHANGE_FRAC_THRESH docstrings. `a is None` (first
    frame seen) counts as changed."""
    if a is None:
        return True
    if a.shape != b.shape:
        return True
    diff = cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    return float((diff > CHANGE_PIXEL_DELTA).mean()) > CHANGE_FRAC_THRESH


def icon_ink_mask(crop_bgr):
    """-> binary mask (0/255), icon ink = 255. Per-image Otsu, not a fixed
    cutoff, because ink color varies by icon (blue/red/whatever) but is
    always darker/more saturated than the white square it sits on.
    Used on the sourced art when building icon templates and on broadcast
    crops when matching them, so both masks come out of the same threshold."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask
