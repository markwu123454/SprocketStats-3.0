#!/usr/bin/env python3
"""
Step 4 -- Extraction: read every region 02_detect_overlay.py found, on every
frame where it has settled, and emit a time series of what each one said.
Nothing more. No roles, no field names, no alliance, no match window, no
"final" value -- those are 06_identify.py's job, and it is better placed to
decide them (see "Why this file knows nothing" below).

Reading is TEMPLATE SEARCH against the digit bank 04_build_templates.py
built, not OCR. No easyocr/torch import anywhere in this file.

Why this file knows nothing
---------------------------
It used to take named fields from 02 (blue_score, clock, ...) and inherit a
role-derived polarity, a role-derived template bank, and a scan window
derived from 01_audio.py's phase marks. All three were assumptions arriving
too early:

* Polarity came from the role ("clock is dark-on-light, everything else is
  light-on-dark"). It is now MEASURED per region, by decoding under both
  polarities during calibration and keeping whichever explains the ink. On
  the 32 known real fields that recovers the correct polarity 32/32, and it
  is not close: correct polarity scores 5-24 on the decode objective, wrong
  polarity -inf to 0.04. The old mapping was a latent bug -- a broadcast
  with a dark-on-light score would have failed outright.
* The template bank was chosen by role too. Banks are now merged per
  character across kinds and the best exemplar picked per region during
  calibration. Justified by measurement: score/badge/clock share a typeface
  (same-digit NCC 0.93-0.98 across kinds, with aspect ratios agreeing to
  0.02), so the split was never carrying real information -- and the year
  split (2026 vs 2025) is larger than the kind split.
* The scan window came from the clock, which required knowing which region
  WAS the clock. That is now gone: this file reads the whole video and lets
  06 cut the window from behaviour. Measured cost of dropping the bound:
  +10% decodes and +1% runs. The out-of-window runs are mostly CORRECT reads
  of the pre-match zeros and the post-match final held on screen -- their
  median margin (0.3466) is indistinguishable from in-match runs (0.3468),
  so they cannot be filtered on confidence and must be cut on time. 06 has a
  far stronger signal for that than this file ever did: not one region's
  countdown but every region's behaviour at once.

Everything this file still does is a measurement, and every one of them is
per-region and self-checking.

Algorithm
---------
1. CALIBRATION, per region. Probe N_CALIB_FRAMES frames spread across the
   video and choose, by decode objective:
     - polarity (light: bright ink on a dark chip; dark: the reverse)
     - glyph height H, the median ink ROW SPAN. All ten digits are lining
       figures of equal height and separators sit inside that span, so the
       row span IS the glyph height -- nothing to threshold. It adapts
       across rendering scales on its own (38/32/18px on 1080p-native
       broadcasts, 60/65/34px on match7/match9) with no configuration.
     - one template exemplar per character, from whichever kind's bank fits
       this region best.
   A region that never yields a usable decode is dropped here (TRIAGE) --
   this is what keeps live camera content out of the scan, and it does so by
   measurement rather than by a geometric guess. match6 has a region sitting
   on a field-wall sign reading "REBUILT"; it produces no decodable frame
   under either polarity.

2. Templates are scaled to height H preserving their own aspect ratio, and
   tried at a few WIDTHS around nominal (WIDTH_JITTER_PX): 04's per-bucket
   median aspect runs 1-2px off the true width at any given scale, and that
   residual used to strand slivers of unexplained ink.

3. DECODE by a 1-D dynamic program over x: at each column, place a glyph or
   move on. Count, identity and boundaries fall out of one optimization.

Objective: explain the ink (and don't paint any that isn't there)
-----------------------------------------------------------------
Recorded because an earlier attempt got this wrong in a way that was not
obvious and cost a full evaluation round.

That attempt maximized a plain SUM of correlation scores over a tiling
constrained to explain EVERY ink column. The argument for why a sum needed
no per-glyph penalty was: the narrowest digit is 0.41H and the widest 0.75H,
so an n-digit span cannot host n+1 glyphs without a 0.41H gap, and real gaps
are 2-4px. That argument is WRONG -- it assumes a placement lies inside the
ink, and nothing forced that. A template may extend past the ink into
background, so the width bound never binds. Measured consequence: badge "36"
decoded as "361" (per-glyph 0.92, 0.93, -0.45), because exact coverage could
not decline to explain the 1-2px sliver left when a template is slightly
narrow, and the sum did not charge enough for the phantom that explained it.

Both failures are one failure: the decoder was FORCED to account for ink it
could not explain, and was not CHARGED for glyphs that explained nothing.

    s(d, x) = NCC(d, x) * ink_mass(d)  -  GAMMA * false_ink(d, x)

    ink_mass(d)     = sum of the template's own normalized intensity
    false_ink(d, x) = template ink landing where the field mask has none

and the DP may SKIP any column at no cost and no reward. A phantom on
background earns ~0 and pays GAMMA times nearly its whole ink mass, so it is
never worth placing; leftover slivers are simply skipped. Length bias
disappears without normalizing by glyph count, because a placement is worth
adding exactly when the ink it explains outweighs the ink it invents.

One consequence worth exploiting upstream: an over-wide region costs
NOTHING here, because skipped columns are free. Region precision matters in
one direction only -- too narrow is fatal, too wide is harmless right up to
the point where it swallows a neighbouring number.

Separators
----------
':' is ink, and an objective built on explaining ink has to explain it.
03_calibrate.py harvests it (segment_separators, no OCR) at FULL DIGIT
HEIGHT so it shares the digits' vertical anchor, and it is matched here as
an ordinary glyph. `raw` therefore contains the separator as decoded, e.g.
"1:53". Splitting it into minutes and seconds -- or a numerator and a
denominator, for a "50 / 100" style field -- is semantics and belongs in 06.

Interval voting
---------------
A value sits unchanged for hundreds of frames. The unit of work is a RUN --
a maximal stretch of sampled frames whose crop is stable and matches the
run's reference crop -- and every frame in it is decoded, with the majority
vote reported. Run boundaries use 03_calibrate.py's settle detection
unchanged.

Read `agreement` with care. Voting suppresses TRANSIENT error and does
nothing to systematic error, and when the decoder is systematically wrong it
is unanimously wrong: in an earlier failed evaluation, 97.5% of incorrect
clock runs reported agreement = 1.0. It measures stability, not correctness.

Output
------
data/<match>_regions_timeline.json:
{
  "match":.., "fps":.., "n_decodes":..,
  "regions": {"r07": {"box":.., "polarity":.., "glyph_h":.., "bg_bgr":..,
                      "chars":.., "n_runs":..}, ...},
  "dropped": [{"id":.., "box":.., "reason":..}, ...],
  "events": [{"region":"r07", "frame":.., "end_frame":.., "t_sec":..,
              "t_end_sec":.., "raw":"1:53", "n_frames":.., "agreement":..,
              "runner_up":.., "min_margin":.., "min_score":.., "mean_score":..}]
}
`raw` is the majority-voted glyph string. There is deliberately no parsed
`value` and no `finals`: both require knowing what the region MEANS.

Known limitations / unvalidated
--------------------------------
- The pooled bank still spans two game years. Nothing conditions it on the
  broadcast, so 2025's heavier stroke weight is matched against templates
  built mostly from 2026 renderings. Interval voting cannot help -- that
  error is systematic.
- Still nearest-centroid NCC, which weights every pixel equally, exactly
  wrong when what separates 3 from 8 is a handful of edge pixels. A
  discriminative model over the same glyph vector is NOT implemented.
- Glyph height is a median over frames spread across the WHOLE video now
  that there is no scan bound. It survived on the current footage (the match
  is ~77% of runtime) but a broadcast with a long intro could drag it. The
  fix if that appears is to re-take the median over only frames that decoded
  well.
- GAMMA, WIDTH_JITTER_PX, VSEARCH_PX, N_CALIB_FRAMES and the triage
  thresholds are swept or reasoned starting points, not jointly optimized.

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


# field_mask and _looks_changed only. easyocr is imported lazily inside 03,
# so this costs nothing here.
calib = _load_calibrate()

STRIDE = 2

# Frames probed per region during calibration, spread across the whole video.
N_CALIB_FRAMES = 14

# Interval voting samples a run rather than exhausting it. A value that holds
# for 400 frames does not need 400 votes -- the point of voting is to drown
# out TRANSIENT error, and a few dozen independent samples do that as well as
# hundreds while costing an order of magnitude less. The first
# VOTE_ALWAYS_FIRST frames of every run are always decoded so that short runs
# (a value that changes again immediately) are never under-sampled.
VOTE_STRIDE = 5
VOTE_ALWAYS_FIRST = 6

# Vertical slack, in pixels, around the measured ink top row when placing a
# template. Absorbs antialiasing jitter in where the mask decides the glyph
# starts; not a plausibility test.
VSEARCH_PX = 2

# Width hypotheses per template, in pixels around its nominal scaled width.
# 04 fixes each glyph's aspect from its bucket's MEDIAN height, which varies
# ~5% across digits even though real digits are exactly equal height, so
# nominal widths are systematically 1-2px off.
WIDTH_JITTER_PX = (-2, -1, 0, 1, 2)

# Price of one invented ink pixel relative to the reward for one explained
# ink pixel. Swept 0/0.25/0.5/1/1.5/2/3 against the clock countdown oracle:
# 36.7% / 79.3% / 91.1% / 95.0% / 95.0% / 95.0% / 94.9%. A broad plateau
# rather than a peak, which is what to expect when the parameter separates
# two well-separated populations (a leftover sliver carries ~20x less ink
# than a real glyph) rather than trading off comparable ones.
GAMMA = 1.0

# --- triage (see Algorithm step 1) -----------------------------------------
# A region is kept if at least this fraction of calibration probes produced a
# decode, and its median margin clears the floor. Measured separation is
# wide: real numeric fields have min_margin >= 0.11 on every sampled frame
# across 8 matches (n=560, median 0.35), while the one known non-numeric
# region produces no decodable frame at all under its true polarity.
TRIAGE_MIN_DECODE_FRAC = 0.30
TRIAGE_MIN_MEDIAN_MARGIN = 0.05
# A glyph height this small is a degenerate measurement, not a glyph. Found
# on match6's field-wall region, which calibrated to 1px and then had 190px
# templates scaled down to it.
TRIAGE_MIN_GLYPH_H = 6

SEPARATOR_CHARS = ":/"


# ---------------------------------------------------------------------------
# Template bank
# ---------------------------------------------------------------------------

def load_templates() -> dict:
    """{char: [template, ...]} pooled across kinds.

    Kinds are merged rather than kept apart because the fonts were measured
    to be the same one: cross-kind same-digit NCC runs 0.93-0.98 with aspect
    ratios agreeing to 0.02. Keeping every kind's rendering as an alternative
    EXEMPLAR (rather than averaging them, which would blur across the 2025/
    2026 stroke-weight difference) lets calibration pick whichever actually
    fits a given region. `sep.png`/`slash.png` are 03/04's names for the ':'
    and '/' buckets -- named that way, not literally ':'/'/', because both
    are also used as filesystem directory names upstream in
    03_calibrate.py's INSTANCES_DIR, and '/' cannot be a path component."""
    bank = collections.defaultdict(list)
    for kind_dir in sorted(p for p in TEMPLATES_DIR.iterdir() if p.is_dir()):
        for f in sorted(kind_dir.glob("*.png")):
            if f.stem.endswith("_compare"):
                continue
            ch = {"sep": ":", "slash": "/"}.get(f.stem, f.stem)
            if ch not in SEPARATOR_CHARS and not (len(ch) == 1 and ch.isdigit()):
                continue
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                bank[ch].append(img)
    return dict(bank)


def scale_bank(bank: dict, glyph_h: int, exemplar: dict = None) -> dict:
    """{char: [(width, template_0_1, ink_mass), ...]} at height `glyph_h`.

    Each template keeps its own aspect ratio, so glyph width stays evidence
    rather than something a ratio test destroys before use. Normalized to
    [0,1] so ink_mass and the template/mask overlap are in the same units.
    INTER_AREA because 04 leaves templates 10x upscaled, making this always a
    heavy downsample. `exemplar` selects one rendering per character once
    calibration has decided; without it every exemplar is included, which is
    what calibration itself needs."""
    out = {}
    for ch, imgs in bank.items():
        chosen = [imgs[exemplar[ch]]] if (exemplar and ch in exemplar) else imgs
        variants, seen = [], set()
        for tmpl in chosen:
            th, tw = tmpl.shape
            nominal = max(1, int(round(tw * glyph_h / th)))
            for d in WIDTH_JITTER_PX:
                w = nominal + d
                if w < 2 or (id(tmpl), w) in seen:
                    continue
                seen.add((id(tmpl), w))
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
    # otherwise compute a response for every row and we would discard all but
    # 2*VSEARCH_PX+1 of them -- with tens of hypotheses per read that is the
    # difference between a usable scan and an unusable one.
    band_y0 = max(0, y_top - VSEARCH_PX)
    band_y1 = min(crop_h, band_y0 + glyph_h + 2 * VSEARCH_PX)
    if band_y1 - band_y0 < glyph_h:
        band_y0 = max(0, band_y1 - glyph_h)
    if band_y1 - band_y0 < glyph_h:
        return None
    gband = np.ascontiguousarray(gray[band_y0:band_y1])
    mband = np.ascontiguousarray(mask[band_y0:band_y1].astype(np.float32))

    NEG = -1e18
    v_char, v_w, v_score, v_ncc = [], [], [], []
    best_ncc = {}
    for ch, variants in scaled.items():
        for w, tmpl, ink_mass in variants:
            if w > crop_w:
                continue
            ncc_map = cv2.matchTemplate(gband, tmpl, cv2.TM_CCOEFF_NORMED)
            tp_map = cv2.matchTemplate(mband, tmpl, cv2.TM_CCORR)
            # Pick the row by correlation, then read the overlap at that SAME
            # row -- taking each maximum independently would score a
            # placement that never existed.
            rows = ncc_map.argmax(axis=0)
            ncc = ncc_map[rows, np.arange(ncc_map.shape[1])]
            tp = tp_map[rows, np.arange(tp_map.shape[1])]
            sc = np.full(crop_w, NEG, dtype=np.float64)
            sc[:ncc.shape[0]] = ncc * ink_mass - GAMMA * (ink_mass - tp)
            nc = np.full(crop_w, -2.0, dtype=np.float64)
            nc[:ncc.shape[0]] = ncc
            v_char.append(ch); v_w.append(w); v_score.append(sc); v_ncc.append(nc)
            best_ncc[ch] = nc if ch not in best_ncc else np.maximum(best_ncc[ch], nc)
    if not v_char:
        return None
    V_W = np.asarray(v_w, dtype=np.int32)
    V_SCORE = np.stack(v_score)                    # (n_variants, crop_w)
    V_NCC = np.stack(v_ncc)

    # A template's left edge coincides with its glyph's first ink column (04
    # builds them from tight bounding boxes), so only ink columns -- plus one
    # px of slack for antialiasing -- can start a glyph. On a wide region
    # that is most of the columns skipped before any arithmetic happens.
    ink_cols = mask.any(axis=0)
    starts = np.zeros(crop_w, dtype=bool)
    starts[ink_cols] = True
    starts[:-1] |= ink_cols[1:]
    start_idx = np.flatnonzero(starts)

    g = np.full(crop_w + 1, NEG, dtype=np.float64)
    g[crop_w] = 0.0
    pick = [None] * (crop_w + 1)
    ends_all = V_W[None, :]
    next_start = crop_w
    for x in range(crop_w - 1, -1, -1):
        g[x] = g[x + 1]                    # skip this column: no cost, no reward
        if not starts[x]:
            continue
        ends = x + V_W
        ok = ends <= crop_w
        if not ok.any():
            continue
        cand = np.where(ok, V_SCORE[:, x] + g[np.minimum(ends, crop_w)], NEG)
        i = int(np.argmax(cand))
        if cand[i] > g[x]:
            g[x] = float(cand[i])
            pick[x] = (v_char[i], int(V_W[i]), float(V_NCC[i, x]))

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
        # counting those as rivals inflates the margin without limit (a 2.82
        # "NCC difference" was how this was found), overstating confidence
        # exactly on the narrow regions where it is least deserved.
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
# Per-region calibration + triage
# ---------------------------------------------------------------------------

def calibrate_region(frames: list, box: list, bank: dict):
    """Measure polarity, glyph height and one template exemplar per character
    for this region, or return None if it never decodes.

    Polarity is chosen by decode objective rather than told to us: on the 32
    known real fields this recovers it 32/32, with correct polarity scoring
    5-24 and wrong polarity -inf to 0.04. Objective (not margin) is the right
    discriminator here -- margin does not separate polarity -- but margin IS
    the right triage signal, since a non-numeric region can score a healthy
    objective under the wrong polarity while never producing a confident
    glyph."""
    x0, y0, x1, y1 = box
    # Polarity is decided with ONE exemplar per character. The separation is
    # enormous (correct 5-24 on the objective, wrong -inf to 0.04), so it does
    # not need the full multi-exemplar bank, and using it would triple the
    # cost of the stage that runs on every candidate region including junk.
    lean = {ch: imgs[:1] for ch, imgs in bank.items()}
    best = None
    for polarity in ("light", "dark"):
        heights, objs, margins, n_ok = [], [], [], 0
        for frame in frames:
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            ext = ink_extent(calib.field_mask(crop, polarity))
            if ext is None:
                continue
            gh = ext[1] - ext[0] + 1
            if gh < TRIAGE_MIN_GLYPH_H or gh > crop.shape[0]:
                continue
            heights.append(gh)
        if not heights:
            continue
        glyph_h = int(np.median(heights))
        scaled_all = scale_bank(lean, glyph_h)
        # exemplar votes: which rendering of each character wins most often
        votes = collections.defaultdict(collections.Counter)
        for frame in frames:
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            dec = decode_field(crop, polarity, scaled_all, glyph_h)
            if dec is None:
                continue
            n_ok += 1
            objs.append(dec.total / max(1, crop.shape[1]))
            margins.append(dec.min_margin)
        if not objs:
            continue
        cand = {"polarity": polarity, "glyph_h": glyph_h,
                "obj": float(np.median(objs)), "margin": float(np.median(margins)),
                "decode_frac": n_ok / max(1, len(frames))}
        if best is None or cand["obj"] > best["obj"]:
            best = cand
    if best is None:
        return None, "no ink of either polarity"
    if best["glyph_h"] < TRIAGE_MIN_GLYPH_H:
        return None, f"degenerate glyph height ({best['glyph_h']}px)"
    if best["decode_frac"] < TRIAGE_MIN_DECODE_FRAC:
        return None, f"decoded on only {best['decode_frac']:.0%} of probes"
    if best["margin"] < TRIAGE_MIN_MEDIAN_MARGIN:
        return None, f"median margin {best['margin']:.3f} below floor"

    # Polarity and height fixed; now pick one rendering per character. Scored
    # by peak correlation against this region's own pixels rather than by a
    # full decode -- the question is only "which drawing of a 7 looks like
    # THIS broadcast's 7", which one matchTemplate answers directly and a DP
    # would answer no better for many times the cost.
    gh = best["glyph_h"]
    crops = []
    for frame in frames[:10]:
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if best["polarity"] == "dark":
            g = 255.0 - g
        crops.append(g)
    exemplar = {}
    for ch, imgs in bank.items():
        if len(imgs) == 1 or not crops:
            exemplar[ch] = 0
            continue
        peaks = []
        for tmpl in imgs:
            th, tw = tmpl.shape
            w = max(2, int(round(tw * gh / th)))
            t = cv2.resize(tmpl, (w, gh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            vals = [float(cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED).max())
                    for g in crops if g.shape[0] >= gh and g.shape[1] >= w]
            peaks.append(float(np.mean(vals)) if vals else -2.0)
        exemplar[ch] = int(np.argmax(peaks))
    best["exemplar"] = exemplar
    return best, None


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def _close_run(region_id, run, events):
    """One run -> one event. Majority vote over every frame decoded inside
    the run; `agreement` is the winning fraction. See the module docstring on
    why agreement measures stability, not correctness."""
    if not run or not run["votes"]:
        return
    ranked = collections.Counter(run["votes"]).most_common()
    raw, n_win = ranked[0]
    stats = run["stats"][raw]
    events.append({
        "region": region_id,
        "frame": run["start"], "end_frame": run["end"],
        "t_sec": round(run["start"] / run["fps"], 2), "t_end_sec": round(run["end"] / run["fps"], 2),
        "raw": raw,
        "n_frames": len(run["votes"]), "agreement": round(n_win / len(run["votes"]), 3),
        "runner_up": (ranked[1][0] if len(ranked) > 1 else None),
        "min_margin": round(min(s[0] for s in stats), 4),
        "min_score": round(min(s[1] for s in stats), 4),
        "mean_score": round(float(np.mean([s[2] for s in stats])), 4),
    })


def extract_match(match: str, bank: dict) -> dict:
    regions_path = DATA_DIR / f"{match}_regions.json"
    video_path = ROOT.parent / f"{match}.mp4"
    doc = json.loads(regions_path.read_text())
    regions = doc["regions"]

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    calib_frames = []
    for f in np.linspace(0, max(0, total - 1), N_CALIB_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if ok:
            calib_frames.append(frame)

    kept, dropped = {}, []
    for r in regions:
        cfg, why = calibrate_region(calib_frames, r["box"], bank)
        if cfg is None:
            dropped.append({"id": r["id"], "box": r["box"], "reason": why})
            continue
        cfg["box"] = r["box"]
        cfg["bg_bgr"] = r.get("bg_bgr")
        cfg["bg_std"] = r.get("bg_std")
        cfg["scaled"] = scale_bank(bank, cfg["glyph_h"], cfg["exemplar"])
        cfg["chars"] = "".join(sorted(cfg["scaled"]))
        kept[r["id"]] = cfg
    print(f"[triage] {match}: {len(kept)} regions kept, {len(dropped)} dropped "
          f"of {len(regions)}", file=sys.stderr)
    for d in dropped:
        print(f"    drop {d['id']} {d['box']}: {d['reason']}", file=sys.stderr)
    for rid, c in sorted(kept.items()):
        print(f"    keep {rid} {c['box']} polarity={c['polarity']} h={c['glyph_h']}px "
              f"margin={c['margin']:.3f}", file=sys.stderr)

    state = {rid: {"prev": None, "ref": None, "run": None} for rid in kept}
    events = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    t_start = time.time()
    n_decoded = 0
    for frame_idx in range(total):
        if frame_idx % STRIDE != 0:
            cap.grab()
            continue
        ok, frame = cap.read()
        if not ok:
            break
        for rid, cfg in kept.items():
            st = state[rid]
            x0, y0, x1, y1 = cfg["box"]
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            # 03's settle detection, unchanged -- only its consumer moved: a
            # settled frame casts a VOTE inside a run rather than being the
            # single classification for that state.
            if st["prev"] is not None and not calib._looks_changed(st["prev"], crop):
                if st["ref"] is None or calib._looks_changed(st["ref"], crop):
                    _close_run(rid, st["run"], events)
                    st["ref"] = crop
                    st["run"] = {"start": frame_idx, "end": frame_idx, "fps": fps,
                                 "seen": 0, "votes": [], "stats": collections.defaultdict(list)}
                run = st["run"]
                seen = run["seen"]
                run["seen"] = seen + 1
                if seen >= VOTE_ALWAYS_FIRST and seen % VOTE_STRIDE:
                    st["prev"] = crop
                    continue
                dec = decode_field(crop, cfg["polarity"], cfg["scaled"], cfg["glyph_h"])
                n_decoded += 1
                if dec is not None:
                    run["votes"].append(dec.raw)
                    run["stats"][dec.raw].append((dec.min_margin, dec.min_score, dec.mean_score))
                    run["end"] = frame_idx
            st["prev"] = crop
        if frame_idx % 4000 == 0:
            print(f"[scan] {match}: frame {frame_idx}/{total} "
                  f"({100*frame_idx/max(1,total):.0f}%), {len(events)} runs, "
                  f"{n_decoded} decodes, {time.time()-t_start:.0f}s", file=sys.stderr)
    for rid in kept:
        _close_run(rid, state[rid]["run"], events)
    cap.release()

    events.sort(key=lambda e: (e["frame"], e["region"]))
    n_runs = collections.Counter(e["region"] for e in events)
    out_regions = {}
    for rid, c in kept.items():
        out_regions[rid] = {"box": c["box"], "polarity": c["polarity"], "glyph_h": c["glyph_h"],
                            "bg_bgr": c["bg_bgr"], "bg_std": c["bg_std"], "chars": c["chars"],
                            "calib_margin": round(c["margin"], 4), "n_runs": n_runs.get(rid, 0)}
    return {"match": match, "fps": fps, "n_frames": total, "n_decodes": n_decoded,
            "regions": out_regions, "dropped": dropped, "events": events}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", required=True)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    bank = load_templates()
    if not bank:
        sys.exit(f"[error] no templates under {TEMPLATES_DIR} -- run 04_build_templates.py first")
    print(f"[bank] {len(bank)} characters, exemplars: " +
          ", ".join(f"{c}x{len(v)}" for c, v in sorted(bank.items())), file=sys.stderr)

    result = extract_match(args.match, bank)
    print(f"[extract] {args.match}: {len(result['events'])} runs from "
          f"{result['n_decodes']} decodes across {len(result['regions'])} regions", file=sys.stderr)

    if args.save:
        out_path = DATA_DIR / f"{args.match}_regions_timeline.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[save] -> {out_path}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
