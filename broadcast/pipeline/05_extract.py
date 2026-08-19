#!/usr/bin/env python3
"""
Step 4 -- Extraction: walk a full match and read the live score-change
timeline (blue/red score, clock, badge/game-piece counts) by TEMPLATE
SEARCH against the digit bank 04_build_templates.py built, instead of OCR.
This is the payoff for calibration existing as its own stage at all -- see
02_detect_overlay.py's module docstring: "Calibration ... so a later
extraction step doesn't need OCR/GPU per frame." No easyocr/torch import
anywhere in this file.

Why this file was rewritten
---------------------------
The version before this one made irreversible decisions early and tried to
repair them late: segmentation committed to "these are the glyphs" using
four hand-tuned geometry thresholds BEFORE any template evidence was
consulted, classification then committed to a digit per glyph with no way
to weigh one glyph's placement against its neighbours', and a jump-magnitude
filter tried to infer from the resulting NUMBER alone whether the whole
chain had been right. Measured failure mass across 8 matches: sub-glyph
fragments winning a geometry test they should never have been asked (~63%
of misclassifications), whole-match font drift the pooled bank could not
represent (~26%), close-pair confusion on clean glyphs (~11%). Separately
and invisibly, MIN_GLYPH_WH_RATIO=0.30 DELETED the digit "1" at badge
rendering scale (5x17px -> 0.29), so those frames yielded zero glyphs and
never appeared in any error tally at all.

Algorithm
---------
1. Per-match, per-field CALIBRATION of one quantity: glyph height H, in
   pixels. Measured as the median over ~40 frames spread across the scan
   range of the field mask's ink ROW SPAN. All ten digits are lining figures
   of equal height with no ascender/descender, and the clock's ':' sits
   strictly inside that span, so the ink row span IS the glyph height --
   there is nothing to threshold. Measured stable to the pixel, and it
   adapts across rendering scales on its own: 38px score / 32px clock / 18px
   badge on the 1080p-native broadcasts, 60/65/34px on match7 and match9,
   with no configuration. This single measured number replaces
   MIN_GLYPH_AREA_PX, MIN_GLYPH_HEIGHT_FRAC, MIN_GLYPH_WH_RATIO and
   MAX_GLYPH_WH_RATIO.

2. Every template is scaled to height H preserving ITS OWN aspect ratio (so
   "1" stays narrow and "4" stays wide -- template width is evidence, not
   something a ratio test destroys before use) and slid horizontally across
   the crop. Each template is additionally tried at a few WIDTHS around its
   nominal one (WIDTH_JITTER_PX): 04's per-bucket median aspect is 1-2px off
   the true width at any given rendering scale, and that residual is not
   cosmetic -- see the objective below for what it used to cost.

3. The string is DECODED by a 1-D dynamic program over x: at each column the
   decoder either places a glyph or moves on. Digit count, identity and
   boundaries all fall out of one optimization.

Objective: explain the ink (and don't paint any that isn't there)
-----------------------------------------------------------------
Recorded because the first attempt got this wrong in a way that was not
obvious and cost a full evaluation round.

That attempt maximized a plain SUM of correlation scores over a tiling
constrained to explain EVERY ink column. The argument for why a sum needs no
per-glyph penalty was: the narrowest digit is 0.41H and the widest 0.75H, so
an n-digit span cannot host n+1 glyphs without a 0.41H gap, and real gaps are
2-4px. That argument is WRONG. It assumes a placement lies inside the ink,
and nothing forced that -- a template may extend past the ink into
background, so the width bound never binds. Measured consequence: badge "36"
decoded as "361" (per-glyph scores 0.92, 0.93, -0.45), because the exact-
coverage rule could not decline to explain the 1-2px sliver left over when a
template is slightly narrower than the real glyph, and the sum did not charge
enough for the phantom that explained it. Decode length exceeded an
independent lower bound on glyph count (count of separated ink runs) on 31.8%
of score reads and 20.6% of badge reads; the clock oracle read 4.2% correct.

Both failures are one failure: the decoder was FORCED to account for ink it
could not explain, and was not CHARGED for glyphs that explained nothing.
The objective now scores a placement by how much ink it accounts for:

    s(d, x) = NCC(d, x) * ink_mass(d)  -  GAMMA * false_ink(d, x)

    ink_mass(d)     = sum of the template's own normalized intensity
    false_ink(d, x) = template ink landing where the field mask has none
                      (= ink_mass - the template/mask overlap at x)

and the DP may SKIP any column at no cost and no reward. That change removes
both failure modes structurally rather than by tuning:

* A phantom on background earns NCC*ink_mass ~ 0 and pays GAMMA * (nearly its
  whole ink mass), so it is never worth placing. Leftover slivers are simply
  skipped.
* Length bias disappears without normalizing by glyph count. A placement is
  worth adding exactly when the ink it explains outweighs the ink it invents,
  which is the question that should have been asked in the first place.

GAMMA is the one free parameter -- the price of inventing an ink pixel
relative to the reward for explaining one. It was swept against the clock
oracle rather than guessed; see the sweep note next to its definition.

The clock separator
-------------------
The ':' is ink, and an objective built on explaining ink has to be able to
explain it. There was no separator in the bank because 03_calibrate.py's
height gate dropped those components before harvesting and its OCR allowlist
stripped the character -- so the previous rewrite, having removed that gate,
was left forcing a DIGIT onto the colon on every clock frame of every match
whose box contains one ("0:20" decoding as "9120", unanimously, all match).
03_calibrate.py now harvests it (segment_separators, no OCR required) at FULL
DIGIT HEIGHT so it shares the digits' vertical anchor, and it is matched here
as an ordinary glyph that happens to render as ':'.

That also makes the mm/ss split structural: parse_clock_seconds splits on the
separator the decoder actually found, instead of assuming the last two digits
are seconds.

Interval voting
---------------
A score sits unchanged for hundreds of frames; the original version
classified it exactly once, from whichever frame happened to trip the settle
gate, discarding ~30 independent samples per second. The unit of work is a
RUN -- a maximal stretch of sampled frames whose crop is stable and matches
the run's reference crop -- and every frame in it is decoded, with the
majority vote reported. Run boundaries use 03_calibrate.py's settle detection
unchanged; only its consumer moved.

Read the `agreement` field with care. Voting suppresses TRANSIENT error and
does nothing to systematic error, and when the decoder is systematically
wrong it is unanimously wrong: in the failed evaluation above, 97.5% of
incorrect clock runs reported agreement = 1.0. Agreement is a measure of
stability, not of correctness, and it is only informative once systematic
error has been removed by other means.

Score and badge are read as completely independent fields. In 2026 the first
badge happens to track the alliance score closely, but that is a quirk of one
season -- match3 (2025 REEFSCAPE) shows unrelated numbers in those boxes --
so nothing here cross-checks one against the other.

Confidence
----------
`min_margin` is the smallest gap between the winning glyph and the runner-up
AT THE SAME PLACEMENT, over the decoded string. It is reported, not enforced:
nothing in this file drops, rejects, rewrites or second-guesses a reading.
MIN_GLYPH_CONFIDENCE (an absolute-score floor) was removed -- measured to
catch 4.8% of score errors and 0% of clock errors, so it was not doing the
job its name implied.

Also removed: JUMP_REJECT_THRESHOLD / JUMP_CONFIRM_TOLERANCE /
JUMP_CONFIRM_STREAK (repaired a per-frame error rate at the wrong layer, and
were blind to where the errors actually were -- a dropped leading digit is a
600-point jump at high score and a 10-point jump at low score, and the
misreads clustered at low scores); and END_FRAC (a fraction of video length
guessing at where post-match content starts -- superseded by the scan window
below).

Scan window
-----------
Extraction runs from match start minus LEAD_MARGIN_SEC to match end plus
TAIL_MARGIN_SEC, where those two marks come from the CLOCK ITSELF when it is
readable (probe_clock_timing) and from 01_audio.py's phases otherwise. This
is not a tidiness measure. Every
confirmed decoder failure in the current 8-match set is a frame where the
overlay is NOT DRAWN -- CGI intro, chip mid-animation, or a post-match scene
wipe -- and on those frames the HSV mask saturates: the whole box comes back
as ink, so no template can invent any, the GAMMA * false_ink term goes to
zero, and the objective silently degenerates to plain NCC * ink_mass, which
is the broken behaviour this rewrite exists to remove. The correcting term
switches itself off precisely where it is needed. Bounding the scan to the
match is what keeps those frames from ever reaching the decoder.

Why the clock leads and audio follows:

* It works where audio has nothing. match3 is 2025 REEFSCAPE and 01 has no
  cue profile for it, so it has no phases file and was the one confirmed
  failure the audio bound could not remove. Its clock reads cleanly and puts
  match_end at 159.78s from a 143-sample run.
* It measures the right thing. The score overlay is synchronised to the
  displayed clock, not to the buzzer. On match1 -- the tightest tail in the
  set -- audio match_end is 168.90 and the clock reaches zero at 170.42, and
  the final score lands at 171.80. Against the clock that is +1.38s; against
  audio it is +2.90s, i.e. 0.10s inside a 3s tail. Clock-derived timing is
  what turns TAIL_MARGIN_SEC from a knife-edge into a real margin.
* Where both exist they agree: clock-vs-audio match_end differs by -0.17 to
  +1.53s across the 7 matches 01 marked, median ~0.0s. That difference is
  recorded as `timing_delta` and is a free cross-modal QC signal.

Audio remains the fallback for a broadcast with no readable clock. Nothing in
the current 8-match set exercises that path, so it is untested in anger.

This is a bound, not a presence check. It cannot help a field that goes
missing DURING a match.

Output
------
data/<match>_score_timeline.json -- {"match", "fps", "bound_source",
"match_start_sec", "match_end_sec", "scan_start_sec", "scan_end_sec",
"glyph_heights", "gamma", "events": [...], "finals": {...}}. Each
event is one RUN: {"frame", "end_frame", "t_sec", "t_end_sec", "field",
"kind", "raw", "value", "n_frames", "agreement", "runner_up", "min_margin",
"min_score", "mean_score"}. `raw` is the majority-voted glyph string (clock
includes the ':' as decoded); `value` is it parsed. `finals`: the last event
per field.

Known limitations / unvalidated
--------------------------------
- The pooled per-kind bank still spans two game years. Nothing here
  conditions the bank on the broadcast, so match3's heavier 2025 stroke
  weight is still matched against templates built almost entirely from 2026
  renderings. Interval voting cannot help -- that error is systematic. Note
  the old OCR-paired pipeline was also only 76.2% correct on match3's clock,
  so this is a bank problem, not a decoder problem.
- The classifier is still nearest-centroid NCC, which weights every pixel
  equally -- exactly wrong when what separates 3 from 8 is a handful of
  pixels on one edge. A discriminative model over the same glyph vector is
  NOT implemented here.
- 02_detect_overlay.py's clock box for match7 and match9 is truncated to the
  SECONDS only -- the real overlay reads "1:53" and the box covers "53".
  Nothing in this file can recover the minutes digit, and any clock metric on
  those two matches is measuring a cropped field. Independent upstream bug;
  it affected the old pipeline identically. It shows up here as those two
  matches yielding zero separator instances during harvesting.
- GAMMA, WIDTH_JITTER_PX, VSEARCH_PX and N_HEIGHT_SAMPLES are swept or
  reasoned starting points, not jointly optimized.
- Still no presence check -- the scan window bounds WHEN, not WHETHER, the
  overlay is drawn. A content-based check is the real fix; the numbers to
  build one are already computed inside decode_field (on failing frames every
  placement reports ~100% of its ink explained, against 70-90% on correct
  reads), and min_margin separates cleanly on the current set (8063 in-match
  reads on real numeric fields have min_margin >= 0.080, while match6's
  blue_badges[0] -- a box 04 already flagged as possibly not a badge -- sits
  at a median of -0.0001).
- Score DECREASES are not errors. FRC scores legitimately flicker and revert
  live, confirmed by the user and visible directly in match3's overlay
  (248/250/250/248/250 across consecutive frames). Nothing here treats a drop
  as suspicious, and any downstream QC must not either.

Usage
-----
  python pipeline/05_extract.py --match match1 --save
"""

import argparse, collections, importlib.util, json, pathlib, sys, time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
TEMPLATES_DIR = DATA_DIR / "digit_templates"


def _load_calibrate():
    spec = importlib.util.spec_from_file_location("calibrate", pathlib.Path(__file__).parent / "03_calibrate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# field_mask, _looks_changed and the box->field enumeration are reused;
# segment_glyphs and its geometry gates are not. easyocr is imported lazily
# inside 03, so this costs nothing here.
calib = _load_calibrate()

STRIDE = 2

# Scan window, relative to 01_audio.py's own phase marks: from auto_start
# minus LEAD, to match_end plus TAIL. Everything outside is CGI intro/outro
# or a broadcast transition -- see the "Scan window" section above.
#
# TAIL is the tighter of the two and the measurement behind it is worth
# keeping: the overlay carries on updating for a moment after the buzzer, and
# the largest lag measured across the 8 matches is match1's final score
# landing at match_end + 2.90s. The nearest contaminating event on the other
# side is match2's scene wipe at match_end + 4.71s. So the whole viable window
# is [2.90, 4.71] and TAIL=3 sits 0.10s inside its lower edge -- it captures
# every final score in the current set, but with almost no room. 4s would sit
# nearer the middle (1.10s / 0.71s). Raising it is the safer error: losing a
# final score is a silent wrong number, whereas admitting the wipe produces
# obvious garbage.
LEAD_MARGIN_SEC = 1
TAIL_MARGIN_SEC = 3

# --- clock-derived timing (see "Scan window" in the module docstring) -------
# Probe sample rate, in Hz, for the pass that locates the match from the clock
# itself. 2 Hz is far denser than needed to establish a 1 Hz countdown and
# keeps the probe to a fraction of the main scan.
CLOCK_PROBE_HZ = 2
MAX_CLOCK_SEC = 180          # longest displayable FRC period (2:15 plus slack)

# A candidate countdown run is grown by comparing each sample's k = value + t
# against the run's RUNNING MEAN k, not against the previous sample. That
# distinction matters: a STATIC display (the teleop clock sitting at 2:20
# between periods) has k increasing by exactly the sample interval, so
# neighbour-to-neighbour it looks identical to a countdown and would chain
# into one indefinitely. Against a running mean it drifts out within a few
# samples and never reaches MIN_COUNTDOWN_RUN.
COUNTDOWN_TOL_SEC = 1.0
MIN_COUNTDOWN_RUN = 8        # samples, i.e. 4s at CLOCK_PROBE_HZ
MIN_COUNTDOWN_SPAN_SEC = 5   # a run must also cover real wall-clock time

# Frames sampled across the scan range to measure each field's glyph height.
N_HEIGHT_SAMPLES = 40

# Vertical slack, in pixels, around the measured ink top row when placing a
# template. Absorbs antialiasing/threshold jitter in where the mask decides
# the glyph starts; not a plausibility test.
VSEARCH_PX = 2

# Width hypotheses tried per template, in pixels around its nominal scaled
# width. 04_build_templates.py fixes each digit's aspect from its bucket's
# MEDIAN height, which varies 360-380px (~5%) across digits even though real
# digits are exactly equal height -- so nominal widths are systematically
# 1-2px off, and that residual is what used to strand slivers of unexplained
# ink. Searching it costs ~5x the matchTemplate calls, which the search-based
# read path has the budget for.
WIDTH_JITTER_PX = (-2, -1, 0, 1, 2)

# Price of one invented ink pixel relative to the reward for one explained
# ink pixel -- see the module docstring's objective section.
#
# Swept 0.25/0.5/1.0/1.5/2.0 against the clock countdown oracle across the 6
# matches whose clock box is intact; see the sweep log in chat. The result is
# a broad plateau rather than a peak, which is the behaviour to expect if the
# parameter is separating two well-separated populations (a leftover sliver
# carries ~20x less ink than a real glyph) rather than trading off two
# comparable ones.
GAMMA = 1.0


# ---------------------------------------------------------------------------
# Template bank
# ---------------------------------------------------------------------------

SEPARATOR_CHAR = ":"


def load_templates() -> dict:
    """{kind: {char: template}}. `sep.png` (04's name for the clock ':'
    bucket) is loaded under SEPARATOR_CHAR and from there on is just another
    glyph -- the decoder has no special case for it."""
    bank = {}
    for kind_dir in sorted(p for p in TEMPLATES_DIR.iterdir() if p.is_dir()):
        glyphs = {}
        for f in sorted(kind_dir.glob("*.png")):
            if f.stem.endswith("_compare"):
                continue
            ch = SEPARATOR_CHAR if f.stem == "sep" else f.stem
            if ch != SEPARATOR_CHAR and not (len(ch) == 1 and ch.isdigit()):
                continue
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                glyphs[ch] = img
        if glyphs:
            bank[kind_dir.name] = glyphs
    return bank


def scale_bank(templates: dict, glyph_h: int) -> dict:
    """{char: [(width, template_0_1, ink_mass), ...]} -- every template at
    height `glyph_h`, at each width hypothesis. Templates are normalized to
    [0,1] so ink_mass and the template/mask overlap are in the same units
    (see the objective). INTER_AREA because 04 leaves templates 10x
    upscaled, so this is always a heavy downsample."""
    out = {}
    for ch, tmpl in templates.items():
        th, tw = tmpl.shape
        nominal = max(1, int(round(tw * glyph_h / th)))
        variants, seen = [], set()
        for d in WIDTH_JITTER_PX:
            w = nominal + d
            if w < 2 or w in seen:
                continue
            seen.add(w)
            t = cv2.resize(tmpl, (w, glyph_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            variants.append((w, t, float(t.sum())))
        if variants:
            out[ch] = variants
    return out


def ink_extent(mask: np.ndarray):
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


# ---------------------------------------------------------------------------
# Search + DP decode
# ---------------------------------------------------------------------------

class Decode:
    __slots__ = ("raw", "placements", "total", "min_margin", "min_score", "mean_score")

    def __init__(self, raw, placements, total, min_margin, min_score, mean_score):
        self.raw, self.placements, self.total = raw, placements, total
        self.min_margin, self.min_score, self.mean_score = min_margin, min_score, mean_score


def decode_field(crop_bgr: np.ndarray, polarity: str, scaled: dict, glyph_h: int):
    mask = calib.field_mask(crop_bgr, polarity)
    ext = ink_extent(mask)
    if ext is None:
        return None
    y_top = ext[0]

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if polarity == "dark":
        gray = 255.0 - gray
    crop_h, crop_w = gray.shape

    # Correlate only over the rows a glyph can occupy. matchTemplate would
    # otherwise compute a response for every row of the crop and we would
    # throw all but 2*VSEARCH_PX+1 of them away -- with ~55 template
    # hypotheses per read that is the difference between a usable scan and an
    # unusable one.
    band_y0 = max(0, y_top - VSEARCH_PX)
    band_y1 = min(crop_h, band_y0 + glyph_h + 2 * VSEARCH_PX)
    if band_y1 - band_y0 < glyph_h:
        band_y0 = max(0, band_y1 - glyph_h)
    if band_y1 - band_y0 < glyph_h:
        return None
    gband = np.ascontiguousarray(gray[band_y0:band_y1])
    mband = np.ascontiguousarray(mask[band_y0:band_y1].astype(np.float32))

    placements_by_char = {}    # ch -> list of (width, score_per_x, ncc_per_x)
    best_ncc = {}              # ch -> per-x best NCC over width hypotheses
    for ch, variants in scaled.items():
        for w, tmpl, ink_mass in variants:
            if w > crop_w:
                continue
            ncc_map = cv2.matchTemplate(gband, tmpl, cv2.TM_CCOEFF_NORMED)
            tp_map = cv2.matchTemplate(mband, tmpl, cv2.TM_CCORR)
            # Pick the row by correlation, then read the overlap at that SAME
            # row -- taking each maximum independently would score a placement
            # that never existed.
            rows = ncc_map.argmax(axis=0)
            cols = np.arange(ncc_map.shape[1])
            ncc = ncc_map[rows, cols]
            tp = tp_map[rows, cols]
            score = ncc * ink_mass - GAMMA * (ink_mass - tp)
            placements_by_char.setdefault(ch, []).append((w, score, ncc))
            prev = best_ncc.get(ch)
            padded = np.full(crop_w, -2.0, dtype=np.float32)
            padded[:ncc.shape[0]] = ncc
            best_ncc[ch] = padded if prev is None else np.maximum(prev, padded)
    if not placements_by_char:
        return None

    NEG = -1e18
    g = np.full(crop_w + 1, NEG, dtype=np.float64)
    g[crop_w] = 0.0
    pick = [None] * (crop_w + 1)
    for x in range(crop_w - 1, -1, -1):
        best, bp = g[x + 1], None          # skip this column: no cost, no reward
        for ch, variants in placements_by_char.items():
            for w, score, ncc in variants:
                if x + w > crop_w or x >= score.shape[0]:
                    continue
                v = float(score[x]) + g[x + w]
                if v > best:
                    best, bp = v, (ch, w, float(ncc[x]))
        g[x], pick[x] = best, bp

    raw, placements, scores, margins = [], [], [], []
    x = 0
    while x < crop_w:
        p = pick[x]
        if p is None:
            x += 1
            continue
        ch, w, ncc_here = p
        # Only glyphs that actually FIT at this x are rivals. best_ncc is
        # padded with -2.0 where a template would run off the right edge;
        # counting those as rivals inflates the margin without limit (a
        # 2.82 "NCC difference" was how this was found), which quietly
        # overstates confidence exactly on the narrow boxes where it is
        # least deserved.
        rivals = [float(best_ncc[c][x]) for c in best_ncc if c != ch and best_ncc[c][x] > -1.5]
        margins.append(ncc_here - max(rivals) if rivals else ncc_here)
        scores.append(ncc_here)
        raw.append(ch)
        placements.append([x, w])
        x += w

    if not raw:
        return None
    return Decode("".join(raw), placements, float(g[0]),
                  min(margins), min(scores), float(np.mean(scores)))


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

def parse_value(kind: str, raw: str):
    if not raw:
        return None
    if kind == "clock":
        return parse_clock_seconds(raw)
    if not raw.isdigit():
        return None
    return int(raw)


def parse_clock_seconds(raw: str):
    """Split on the separator the decoder actually found. Falls back to
    02_detect_overlay.py's positional rule (last two digits are seconds) when
    the field has no separator at all -- which is the real situation on
    match7/match9, whose clock box 02 truncated to the seconds."""
    if SEPARATOR_CHAR in raw:
        head, _, tail = raw.partition(SEPARATOR_CHAR)
        if not tail.isdigit() or (head and not head.isdigit()):
            return None
        minutes, seconds = (int(head) if head else 0), int(tail)
    elif not raw.isdigit():
        return None
    elif len(raw) <= 2:
        minutes, seconds = 0, int(raw)
    else:
        minutes, seconds = int(raw[:-2]), int(raw[-2:])
    if seconds >= 60:
        return None
    return minutes * 60 + seconds


def load_phases(match: str):
    p = DATA_DIR / f"{match}_phases.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Glyph-height calibration
# ---------------------------------------------------------------------------

def probe_clock_timing(cap, tracker, bank: dict, total: int, fps: float):
    """Locate the match from the CLOCK, with no audio and no prior bound.

    The clock is a 1 Hz countdown, so within a period k = value + t_sec is
    constant and the period ends (value 0) at t = k. Sampling the clock field
    across the whole video and grouping reads by constant k therefore recovers
    the period structure directly -- match_end is the largest k over all runs,
    and the match starts at the first sample of the earliest run.

    Two things make this safe to run BEFORE any scan bound exists, which is
    what would otherwise be circular:

    * Intro/outro garbage does not form countdown runs. It decodes to
      arbitrary values that share no k, so it fails MIN_COUNTDOWN_RUN and
      excludes itself. Measured on the unbounded extraction of all 8 matches:
      the teleop run is 140-172 consecutive samples and nothing outside the
      match contributes to it.
    * Glyph height is bootstrapped over the whole video. It is a MEDIAN over
      ~40 samples, and in these broadcasts the match covers ~77% of the
      runtime, so out-of-match frames cannot move it.

    Largest k rather than longest run: on a clock box 02 has truncated to the
    seconds (match7, match9) teleop is chopped into minute-length segments,
    and it is the LAST of them that ends at the buzzer. Longest-run happens to
    pick correctly on match7 and is luck, not logic.

    Returns None when no countdown structure is found at all -- a broadcast
    with no clock, or a box mis-detected badly enough to be unreadable -- in
    which case the caller falls back to 01_audio.py's marks.
    """
    heights = calibrate_glyph_heights(cap, [tracker], 0, total)
    glyph_h = heights.get(tracker.name)
    templates = bank.get(tracker.kind)
    if glyph_h is None or not templates:
        return None
    scaled = scale_bank(templates, glyph_h)

    step = max(1, int(round(fps / CLOCK_PROBE_HZ)))
    samples = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for idx in range(total):
        if idx % step:
            cap.grab()
            continue
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[tracker.box[1]:tracker.box[3], tracker.box[0]:tracker.box[2]]
        if crop.size == 0:
            continue
        dec = decode_field(crop, tracker.polarity, scaled, glyph_h)
        if dec is None:
            continue
        v = parse_value("clock", dec.raw)
        if v is not None and 0 <= v <= MAX_CLOCK_SEC:
            samples.append((idx / fps, v))

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

    valid = []
    for r in runs:
        if len(r) < MIN_COUNTDOWN_RUN:
            continue
        span = r[-1][0] - r[0][0]
        drop = r[0][1] - r[-1][1]
        # A real countdown loses one second of value per second of wall clock;
        # this rejects any static stretch that survived the grouping.
        if span < MIN_COUNTDOWN_SPAN_SEC or abs(drop - span) > 2.0:
            continue
        valid.append(r)
    if not valid:
        return None

    k_of = lambda r: sum(v + t for t, v in r) / len(r)
    return {"start": min(r[0][0] for r in valid), "end": max(k_of(r) for r in valid),
            "n_runs": len(valid), "n_samples": len(samples),
            "runs": [{"k": round(k_of(r), 2), "n": len(r),
                      "from": round(r[0][0], 2), "to": round(r[-1][0], 2)} for r in valid]}


def calibrate_glyph_heights(cap, trackers, lo: int, hi: int) -> dict:
    spans = collections.defaultdict(list)
    for f in np.linspace(lo, max(lo, hi - 1), N_HEIGHT_SAMPLES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok:
            continue
        for t in trackers:
            crop = frame[t.box[1]:t.box[3], t.box[0]:t.box[2]]
            if crop.size == 0:
                continue
            ext = ink_extent(calib.field_mask(crop, t.polarity))
            if ext is not None:
                spans[t.name].append(ext[1] - ext[0] + 1)
    return {name: int(np.median(v)) for name, v in spans.items() if v}


# ---------------------------------------------------------------------------
# Per-match extraction
# ---------------------------------------------------------------------------

def _close_run(field_name, kind, run, events):
    """One run -> one event. Majority vote over every frame decoded inside
    the run; `agreement` is the winning fraction. See the module docstring on
    why agreement measures stability, not correctness."""
    if not run or not run["votes"]:
        return
    ranked = collections.Counter(run["votes"]).most_common()
    raw, n_win = ranked[0]
    stats = run["stats"][raw]
    events.append({
        "frame": run["start"], "end_frame": run["end"],
        "t_sec": round(run["start"] / run["fps"], 2), "t_end_sec": round(run["end"] / run["fps"], 2),
        "field": field_name, "kind": kind, "raw": raw, "value": parse_value(kind, raw),
        "n_frames": len(run["votes"]), "agreement": round(n_win / len(run["votes"]), 3),
        "runner_up": (ranked[1][0] if len(ranked) > 1 else None),
        "min_margin": round(min(s[0] for s in stats), 4),
        "min_score": round(min(s[1] for s in stats), 4),
        "mean_score": round(float(np.mean([s[2] for s in stats])), 4),
    })


def extract_match(match: str, bank: dict) -> dict:
    boxes_path = DATA_DIR / f"{match}_overlay_boxes.json"
    video_path = ROOT.parent / f"{match}.mp4"
    boxes = json.loads(boxes_path.read_text())["boxes"]

    trackers = calib.build_trackers(boxes)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Phases are used whenever 01_audio.py found both marks, REGARDLESS of the
    # confidence it recorded. The previous "high only" rule left match7
    # (8/9 cues, "low") scanning its whole video, and that is the wrong call:
    # the phases carry their own consistency check. auto_start -> match_end
    # spans 163.1-165.9s across every match 01 has marked, and match7's
    # 163.52s sits inside that range, so its marks are sound even though one
    # cue was missed. A bound that might be slightly off still beats no bound.
    phases = load_phases(match) or {}
    ph = phases.get("phases") or {}
    audio_start, audio_end = ph.get("auto_start"), ph.get("match_end")

    clock_tracker = next((t for t in trackers if t.kind == "clock"), None)
    clock_timing = probe_clock_timing(cap, clock_tracker, bank, total, fps) if clock_tracker else None
    if clock_timing:
        print(f"[timing] {match}: clock probe found {clock_timing['n_runs']} countdown run(s) "
              f"in {clock_timing['n_samples']} samples -> {clock_timing['runs']}", file=sys.stderr)

    # Clock first, audio second. The clock measures the thing the score
    # overlay is actually synchronised to, it is available on matches
    # 01_audio.py cannot mark at all (match3: no 2025 cue profile), and it
    # agrees with a high-confidence audio match_end to within ~0.1-0.2s where
    # both exist. Audio remains the fallback for a broadcast with no readable
    # clock, which nothing in the current 8-match set exercises.
    if clock_timing:
        match_start_sec, match_end_sec = clock_timing["start"], clock_timing["end"]
        timing_source = "clock"
    else:
        match_start_sec, match_end_sec = audio_start, audio_end
        timing_source = f"audio[{phases.get('confidence')}]" if audio_start is not None else None

    lo, hi = 0, total
    bound_source = "full_video (no clock countdown and no 01_audio.py phases)"
    if match_start_sec is not None and match_end_sec is not None:
        lo = max(0, int((match_start_sec - LEAD_MARGIN_SEC) * fps))
        hi = min(total, int((match_end_sec + TAIL_MARGIN_SEC) * fps))
        bound_source = (f"{timing_source}({match_start_sec:.2f}-{LEAD_MARGIN_SEC}s .. "
                        f"{match_end_sec:.2f}+{TAIL_MARGIN_SEC}s)")

    # Disagreement between two independent measurements of the same events is
    # a free QC signal -- reported, never acted on here.
    timing_delta = None
    if clock_timing and audio_end is not None:
        timing_delta = {"end": round(clock_timing["end"] - audio_end, 2),
                        "start": round(clock_timing["start"] - audio_start, 2) if audio_start is not None else None}

    heights = calibrate_glyph_heights(cap, trackers, lo, hi)
    print(f"[calib] {match}: glyph heights " +
          ", ".join(f"{k}={v}px" for k, v in sorted(heights.items())), file=sys.stderr)

    scaled = {}
    for t in trackers:
        h = heights.get(t.name)
        tmpl = bank.get(t.kind)
        scaled[t.name] = (scale_bank(tmpl, h), h) if (h and tmpl) else (None, None)

    state = {t.name: {"prev": None, "ref": None, "run": None} for t in trackers}
    events = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    t_start = time.time()
    n_decoded = 0
    for frame_idx in range(lo, hi):
        if (frame_idx - lo) % STRIDE != 0:
            cap.grab()
            continue
        ok, frame = cap.read()
        if not ok:
            break
        for t in trackers:
            tmpl, gh = scaled[t.name]
            if tmpl is None:
                continue
            st = state[t.name]
            crop = frame[t.box[1]:t.box[3], t.box[0]:t.box[2]]
            if crop.size == 0:
                continue
            # 03's settle detection, unchanged -- only its consumer moved: a
            # settled frame now casts a VOTE inside a run rather than being
            # the single classification for that state.
            if st["prev"] is not None and not calib._looks_changed(st["prev"], crop):
                if st["ref"] is None or calib._looks_changed(st["ref"], crop):
                    _close_run(t.name, t.kind, st["run"], events)
                    st["ref"] = crop
                    st["run"] = {"start": frame_idx, "end": frame_idx, "fps": fps,
                                 "votes": [], "stats": collections.defaultdict(list)}
                dec = decode_field(crop, t.polarity, tmpl, gh)
                n_decoded += 1
                if dec is not None:
                    st["run"]["votes"].append(dec.raw)
                    st["run"]["stats"][dec.raw].append((dec.min_margin, dec.min_score, dec.mean_score))
                    st["run"]["end"] = frame_idx
            st["prev"] = crop
        if (frame_idx - lo) % 2000 == 0:
            pct = 100 * (frame_idx - lo) / max(1, hi - lo)
            print(f"[scan] {match}: frame {frame_idx}/{hi} ({pct:.0f}%), {len(events)} runs, "
                  f"{n_decoded} decodes, {time.time() - t_start:.0f}s", file=sys.stderr)
    for t in trackers:
        _close_run(t.name, t.kind, state[t.name]["run"], events)
    cap.release()

    events.sort(key=lambda e: e["frame"])
    finals = {e["field"]: e for e in events}
    return {"match": match, "fps": fps, "bound_source": bound_source,
            "timing_source": timing_source, "clock_timing": clock_timing,
            "audio_start_sec": audio_start, "audio_end_sec": audio_end,
            "timing_delta": timing_delta,
            "match_start_sec": match_start_sec, "match_end_sec": match_end_sec,
            "scan_start_sec": round(lo / fps, 2), "scan_end_sec": round(hi / fps, 2),
            "glyph_heights": heights, "gamma": GAMMA, "n_decodes": n_decoded,
            "events": events, "finals": finals}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", required=True)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    bank = load_templates()
    if not bank:
        sys.exit(f"[error] no templates under {TEMPLATES_DIR} -- run 04_build_templates.py first")
    print(f"[bank] " + ", ".join(f"{k}:{len(v)} glyphs" for k, v in bank.items()), file=sys.stderr)

    result = extract_match(args.match, bank)
    events = result["events"]
    n_unparsed = sum(1 for e in events if e["value"] is None)
    n_split = sum(1 for e in events if e["agreement"] < 1.0)
    print(f"[extract] {args.match}: {len(events)} runs from {result['n_decodes']} frame decodes "
          f"({n_unparsed} unparsable, {n_split} with split votes), scan bound: {result['bound_source']}",
          file=sys.stderr)

    for field in sorted({e["field"] for e in events}):
        e = result["finals"][field]
        print(f"[final] {field}: raw={e['raw']!r} value={e['value']} (t={e['t_sec']}-{e['t_end_sec']}s, "
              f"n={e['n_frames']}, agree={e['agreement']}, margin={e['min_margin']})", file=sys.stderr)

    if args.save:
        out_path = DATA_DIR / f"{args.match}_score_timeline.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[save] -> {out_path}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
