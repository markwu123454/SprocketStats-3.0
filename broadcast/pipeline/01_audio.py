#!/usr/bin/env python3
"""
Step 1 -- Detect match phase transitions by matching official FMS audio cues
(data/field_audio/*.wav) against a broadcast video's audio track.

Anchor-and-localize assignment
-------------------------------------------------------------------------
Single-cue matched-filter scores are individually weak and noisy on real
broadcast audio (compression/EQ/crowd noise all hurt raw-waveform
correlation) -- confirmed by hand against ground-truth timestamps: a lone
top-scoring candidate is sometimes a spurious peak (match2: an intro-CG
artifact narrowly outscored the real start cue) and sometimes the real cue
just isn't in a short top-K list at all (match7: real signal ranked 6th of
8 under faint commentary). Fix: stop trusting any single cue's own score in
isolation and use the fact that this is a known, ordered SEQUENCE with
FMS-automatic (hence near-exact) timing between most of its cues.

Take the single highest-confidence candidate across EVERY reference file
(one global pool, not per-file top-K), assume it's real, and use the cue
profile's gap means (data/cue_profiles/<year>.json) to PREDICT where every
other slot should be. Each prediction is checked with a LOCAL statistical
presence test (local_presence(): is there a sample that's both a
significant outlier against the LOCAL background noise AND close to the
predicted center -- see its docstring) rather than "whichever peak scores
highest in the window", which is what let a wrong hypothesis pick up an
unrelated peak 4s after auto_start as "auto_end" before that distance
penalty existed (see PRESENCE_DISTANCE_PENALTY). If the top anchor doesn't
produce a complete chain, falls through to the next-best global candidate
(see anchor_search()). A runner-up hypothesis (same search with the
winner's timestamps masked out) gives a margin to flag low confidence.
Strict chronological order is enforced across the whole chain, since a
wrong anchor can otherwise pull in other real cues near its own
(systematically shifted) predictions and produce a chain that looks
complete/well-scored per-slot but isn't actually in sequence order.

This covers front half (auto_start/auto_end) and back half
(teleop_start..match_end) in one unified pass over the profile's full
sequence -- no special-cased split between them.

abort.wav is checked separately (single best peak, whole track) as an
anomaly signal, not part of the chain.

KNOWN GAME-YEAR DEPENDENCE: cue identities/order/whether shift_change
happens at all/every gap's timing varies by year -- ALL of that now comes
from data/cue_profiles/<year>.json (--year, default 2026), not from
anything hardcoded in this script. See that file's own "notes" for how the
current 2026 profile was built and its caveats.

TRIED AND REJECTED: a joint per-slot DP assignment, and two alternative
correlation methods (GCC-PHAT, ROTH-weighted GCC) in place of the plain
energy-normalized matched filter used here. All three lost a head-to-head
sweep across matches 1/2/4/6/7/8/9/10 and were removed -- see the
bottom-of-file "Tried and rejected" docs note for the numbers and
data/algo_assign_sweep/ for the archived sweep script + full results.

Output (JSON to stdout)
------------------------
{
  "match": "match1",
  "phases": {"auto_start":.., "auto_end":.., "teleop_start":..,
             "endgame_start":.., "match_end":..},
  "shift_changes": [.., .., .., ..],
  "confidence": "high"|"low",
  "margin": ..,
  "runner_up_total": ..,
  "found": "9/9",
  "candidates": {"<file>.wav": [{"t_sec":.., "score":..}, ...], ...},
  "aborted": bool,
  "abort_score": ..
}

Usage
-----
  python pipeline/01_audio.py --video match1.mp4 --save --viz

Install: pip install numpy scipy matplotlib
Requires: ffmpeg on PATH
"""

import argparse, json, pathlib, shutil, subprocess, sys
from fractions import Fraction

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, find_peaks, resample_poly

ROOT             = pathlib.Path(__file__).parent.parent
DATA_DIR         = ROOT / "data"
AUDIO_DIR        = DATA_DIR / "audio"
FIELD_AUDIO_DIR  = DATA_DIR / "field_audio"
CUE_PROFILE_DIR  = DATA_DIR / "cue_profiles"   # per-year profiles, see load_cue_profile()

SR = 22050
DEFAULT_YEAR = 2026

# Minimum separation between two candidate peaks of the SAME reference
# file's score curve, so one physical cue occurrence (which spans a few
# hundred ms and smears the correlation peak a bit) isn't picked twice as
# two different occurrences. Real repeats of the same file in this
# sequence (end.wav x2, shift_change.wav x4) are always tens of seconds
# apart at minimum, so this only needs to be bigger than one cue's own
# smear, not anywhere near the real inter-occurrence spacing.
MIN_PEAK_GAP_SEC = 2.0

# A chain's confidence is "high" only if its total score beats its
# runner-up (same slots, winning candidates excluded) by at least this
# fraction of the winner's own total. UNVALIDATED heuristic threshold --
# exists so a thin margin gets flagged for human review instead of quietly
# asserting a possibly-wrong answer, per explicit design requirement.
CONFIDENCE_MARGIN_FRAC = 0.15

ABORT_SCORE_THRESHOLD = 0.3   # see prior discussion; still unvalidated


def gap_tolerance(stdev: float, floor: float = 0.3, k: float = 3.0) -> float:
    """k standard deviations, floored so a still-small profile sample
    doesn't produce an unrealistically brittle tolerance for an
    already-tight gap (e.g. teleop_start->shift1's stdev=0.022s at n=6
    alone would give an absurdly narrow window)."""
    return max(k * stdev, floor)


# ---------------------------------------------------------------------------
# Cue profile (data/cue_profiles/<year>.json)
# ---------------------------------------------------------------------------
#
# Sequence/timing is year-specific (cue identities/order, whether
# shift_change happens at all, and every gap's mean+stdev) -- see that
# file's own "notes" for how it was built and its known caveats (most
# importantly: auto_end->teleop_start is the one MANUAL transition and its
# mean/stdev must never be used as a tight prior; and the auto_start/
# auto_end-adjacent gaps carry extra measurement noise from those two
# clips' softer correlation peaks, not necessarily real timing variance).
#
# profile["sequence"] is consumed directly, in order, by the anchor path
# below (predict_slot_time/evaluate_anchor) -- no front/back split needed.

def load_cue_profile(year: int) -> dict:
    path = CUE_PROFILE_DIR / f"{year}.json"
    if not path.exists():
        sys.exit(f"[error] no cue profile for year {year}: {path}\n"
                 f"  Build one from --discover output across several confirmed-{year} matches.")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.exit("[error] ffmpeg not found on PATH")


def extract_match_audio(video_path: pathlib.Path, no_cache: bool = False) -> pathlib.Path:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AUDIO_DIR / f"{video_path.stem}.wav"
    if out_path.exists() and not no_cache:
        print(f"[audio] using cached {out_path}", file=sys.stderr)
        return out_path
    check_ffmpeg()
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(video_path), "-ac", "1", "-ar", str(SR), "-vn", str(out_path)]
    print(f"[audio] extracting -> {out_path}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        sys.exit(f"[error] ffmpeg failed: {result.stderr.decode()[:500]}")
    return out_path


def load_wav_mono(path: pathlib.Path, target_sr: int = SR) -> np.ndarray:
    sr, data = wavfile.read(path)
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    elif data.dtype.kind == "u":
        data = (data.astype(np.float32) - 128) / 128
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        frac = Fraction(target_sr, sr).limit_denominator(1000)
        data = resample_poly(data, frac.numerator, frac.denominator).astype(np.float32)
    return data


# ---------------------------------------------------------------------------
# Matched filtering
# ---------------------------------------------------------------------------

def matched_filter_scores(match: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Energy-normalized matched filter -- bounded [-1, 1] cosine similarity
    between match[i:i+len(ref)] and ref at every alignment i. See module
    docstring: individually noisy on real broadcast audio, which is why
    this script never trusts a single score in isolation. Won a head-to-
    head sweep against GCC-PHAT and ROTH-weighted GCC -- see "Tried and
    rejected" docs note at the bottom of this file."""
    L = len(ref)
    if len(match) < L:
        return np.array([])
    raw = correlate(match, ref, mode="valid", method="fft")
    sq = match.astype(np.float64) ** 2
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    window_energy = csum[L:] - csum[:-L]
    window_rms = np.sqrt(np.maximum(window_energy, 1e-12))
    ref_norm = float(np.sqrt(np.sum(ref.astype(np.float64) ** 2))) + 1e-12
    return raw / (window_rms * ref_norm)


def get_all_candidates(scores: np.ndarray, sr: int,
                       min_gap_sec: float = MIN_PEAK_GAP_SEC) -> list[dict]:
    """Every local-maximum peak in `scores` at least min_gap_sec apart --
    no score/prominence cutoff. Deliberately unfiltered: a weak true cue
    (match7) must survive into this list for the chain search to have any
    chance of finding it."""
    if len(scores) == 0:
        return []
    min_gap_samples = max(1, int(min_gap_sec * sr))
    idx, _ = find_peaks(scores, distance=min_gap_samples)
    if len(idx) == 0:
        idx = np.array([int(np.argmax(scores))])
    return [{"t_sec": float(i / sr), "score": float(scores[i])} for i in sorted(idx)]


# ---------------------------------------------------------------------------
# Anchor-and-localize assignment
# ---------------------------------------------------------------------------
#
# An earlier design scored the WHOLE chain additively via a per-slot DP,
# which let a distant, unrelated, louder candidate outscore a correctly-
# positioned-but-weaker one -- this is what actually broke match7: the true
# shift1 (score 0.157, correctly ~9.9s after the true teleop_start) lost to
# a candidate 44s later (score 0.35) that fit ITS OWN neighbors well but
# wasn't really part of the true chain -- the DP had no way to make a
# 44s-away candidate ineligible, only less rewarded. Confirmed inferior in
# a head-to-head sweep against the approach below (see bottom-of-file docs
# note) and removed.
#
# This flips the strategy: take the single highest-confidence candidate
# across EVERY file (one global pool, not per-file top-K), assume it's
# real, and use the profile's gap means to PREDICT where every other slot
# should be. Search each prediction only within a narrow window in the
# already-computed candidate list -- a false candidate 44s from its
# predicted slot isn't competing on score anymore, it's just outside the
# window entirely. If the top anchor doesn't produce a complete chain, fall
# through to the next-best global candidate.
#
# Costs essentially nothing extra: the expensive step (correlating each
# reference file across the whole track) has to run once regardless of
# assignment strategy, to know what candidates exist anywhere. Everything
# below is filtering already-computed candidate lists. Also covers
# front+back half in one pass over the profile's full sequence, rather than
# special-casing that split.

# How many gap-tolerance-widths (combined in quadrature across every gap
# crossed) to search around a predicted slot time. Wider than a soft
# Gaussian falloff would need, since this is a hard cutoff -- missing the
# true candidate here means the slot goes unfound, so err generous.
ANCHOR_SEARCH_K = 4.0

# Tolerance floor used only when a prediction's path crosses the one manual
# gap (auto_end -> teleop_start). Wider than the ordinary 0.3s floor because
# that gap is known to vary sub-second to 5+ seconds across events -- a
# tight window here would systematically miss the true candidate on exactly
# the matches where this matters most.
MANUAL_GAP_TOL_FLOOR = 3.0

# How many local-background MAD-units a predicted window's peak must clear
# to count as "found" -- see local_presence(). UNVALIDATED starting point;
# expect to tune once run against the ground-truth matches.
LOCAL_Z_THRESHOLD = 4.0

# How far beyond a predicted search window to sample for the LOCAL
# background estimate used by local_presence(). Wide enough to get a stable
# median/MAD (needs >=100 samples' worth), narrow enough to still reflect
# "typical background right around here" rather than the whole match.
LOCAL_BACKGROUND_MARGIN_SEC = 15.0

# How many z-units a sample loses per (distance-from-predicted-center /
# base_tol)^2. The search window itself is deliberately wider than the raw
# gap tolerance (ANCHOR_SEARCH_K) so an imprecisely-localized real cue isn't
# clipped out -- but without this penalty, "significant anywhere in that
# wide window" is too permissive: on match7 it picked up an unrelated peak
# 4s after auto_start as "auto_end" instead of the (weaker, but correctly
# ~21s later) real one. At exactly 1 base_tol away a sample needs
# PRESENCE_DISTANCE_PENALTY extra z-units of raw significance just to break
# even with a sample sitting right at the predicted center.
PRESENCE_DISTANCE_PENALTY = 3.0


def local_presence(scores: np.ndarray, sr: int, predicted_t: float, half_width: float,
                   center_tol: float, background_margin: float = LOCAL_BACKGROUND_MARGIN_SEC,
                   z_threshold: float = LOCAL_Z_THRESHOLD) -> dict | None:
    """
    Statistical presence test within a narrow predicted window: is there a
    sample that's BOTH a significant outlier against the LOCAL background
    (median + MAD of the surrounding margin region, excluding the window
    itself) AND reasonably close to the predicted center -- not just "the
    highest value inside the window", which is still "found" even when
    that's an unrelated peak near the window's edge (see
    PRESENCE_DISTANCE_PENALTY docstring for why the plain window-max
    version was too permissive).

    Compares against a LOCAL background (not the whole track's noise floor)
    because correlation noise level genuinely varies over a multi-minute
    match -- crowd-noise stretches vs. quiet ones -- so a single global
    threshold would be too strict in quiet sections and too loose in loud
    ones. MAD (median absolute deviation, scaled by 1.4826 to be a
    consistent std estimator for roughly-normal data) is used instead of
    plain std so a stray strong peak inside the background region itself
    doesn't inflate the estimate and mask a real detection.

    `center_tol` is the pre-widening base tolerance from predict_slot_time
    (i.e. half_width / ANCHOR_SEARCH_K) -- the "how precise do we actually
    expect this gap to be" scale, used only for the distance penalty, not
    for the window bounds themselves.

    Returns None ("not found at all", not just "found weakly") if nothing
    in the window clears z_threshold after the distance penalty.
    """
    n = len(scores)
    lo_idx = max(0, int((predicted_t - half_width) * sr))
    hi_idx = min(n, int((predicted_t + half_width) * sr))
    if hi_idx <= lo_idx:
        return None
    window = scores[lo_idx:hi_idx]
    if not np.any(np.isfinite(window)):
        return None

    bg_lo = max(0, int((predicted_t - half_width - background_margin) * sr))
    bg_hi = min(n, int((predicted_t + half_width + background_margin) * sr))
    background = np.concatenate([scores[bg_lo:lo_idx], scores[hi_idx:bg_hi]])
    background = background[np.isfinite(background)]
    if len(background) < 100:
        return None

    bg_median = float(np.median(background))
    bg_mad = float(np.median(np.abs(background - bg_median))) * 1.4826 + 1e-9

    z_window = np.where(np.isfinite(window), (window - bg_median) / bg_mad, -np.inf)
    center_idx = (predicted_t - lo_idx / sr) * sr
    dist_sec = np.abs(np.arange(len(window)) - center_idx) / sr
    ct = max(center_tol, 1e-6)
    penalty = PRESENCE_DISTANCE_PENALTY * (dist_sec / ct) ** 2
    combined = z_window - penalty

    peak_offset = int(np.argmax(combined))
    peak_z = z_window[peak_offset]
    if not np.isfinite(peak_z) or peak_z < z_threshold:
        return None
    peak_score = float(window[peak_offset])
    return {"t_sec": (lo_idx + peak_offset) / sr, "score": peak_score, "z": round(float(peak_z), 2)}


def mask_scores_near(scores: np.ndarray, sr: int, t_sec: float,
                     radius: float = MIN_PEAK_GAP_SEC) -> np.ndarray:
    """Copy of `scores` with a small region around t_sec set to -inf, so a
    runner-up search (see anchor_with_runner_up) can't rediscover the exact
    same peak local_presence already used."""
    masked = scores.copy()
    lo = max(0, int((t_sec - radius) * sr))
    hi = min(len(scores), int((t_sec + radius) * sr))
    masked[lo:hi] = -np.inf
    return masked


def build_candidate_pool(candidates_by_file: dict[str, list[dict]]) -> list[dict]:
    """Flatten every file's candidates into one list, each tagged with its
    source file, sorted by score descending -- the highest-confidence
    detection across the WHOLE match, regardless of which cue it is."""
    pool = [{**c, "file": fname} for fname, cs in candidates_by_file.items() for c in cs]
    pool.sort(key=lambda c: -c["score"])
    return pool


def predict_slot_time(sequence: list[dict], anchor_idx: int, anchor_t: float,
                      target_idx: int, search_k: float = ANCHOR_SEARCH_K
                      ) -> tuple[float, float, float]:
    """Predicted absolute time for `target_idx`, propagated from
    `anchor_idx` at `anchor_t` by walking the profile's gap means between
    them (forward or backward). Returns (predicted_time, search_half_width,
    base_tol): base_tol is each crossed gap's tolerance combined in
    quadrature at its own scale (used by local_presence to penalize
    distance from the predicted center); search_half_width = search_k *
    base_tol is how far out the search window itself extends (wider, so a
    real-but-imprecisely-localized cue isn't clipped out of consideration
    entirely). Uses the wider manual-gap floor if that link is crossed."""
    if target_idx == anchor_idx:
        return anchor_t, 0.0, 0.0
    lo, hi = (anchor_idx, target_idx) if target_idx > anchor_idx else (target_idx, anchor_idx)
    offset, var = 0.0, 0.0
    for k in range(lo + 1, hi + 1):
        g = sequence[k]["gap_from_prev"]
        floor = MANUAL_GAP_TOL_FLOOR if g.get("manual") else 0.3
        tol = gap_tolerance(g["stdev"], floor=floor)
        offset += g["mean"]
        var += tol ** 2
    base_tol = var ** 0.5
    half_width = search_k * base_tol
    predicted = anchor_t + offset if target_idx > anchor_idx else anchor_t - offset
    return predicted, half_width, base_tol


def evaluate_anchor(sequence: list[dict], scores_by_file: dict[str, np.ndarray],
                    anchor_idx: int, anchor_c: dict, sr: int = SR,
                    z_threshold: float = LOCAL_Z_THRESHOLD) -> dict:
    """Assume `anchor_c` is slot `anchor_idx`; predict + run a LOCAL
    presence test (local_presence, not "highest score in the window") for
    every other slot. Returns the resulting chain, total score, and count
    of slots found -- a chain missing slots is still returned, since the
    caller ranks hypotheses on (found, score) together, not either alone."""
    n = len(sequence)
    chain = [None] * n
    chain[anchor_idx] = anchor_c
    for j in range(n):
        if j == anchor_idx:
            continue
        fname = sequence[j]["file"]
        scores = scores_by_file.get(fname)
        if scores is None:
            continue
        predicted_t, half_width, base_tol = predict_slot_time(sequence, anchor_idx, anchor_c["t_sec"], j)
        hit = local_presence(scores, sr, predicted_t, half_width, base_tol, z_threshold=z_threshold)
        if hit is not None:
            chain[j] = hit

    # Each slot above was found independently against its OWN prediction --
    # nothing yet guarantees the results are mutually consistent as a
    # whole. A wrong anchor hypothesis (e.g. off-by-one on which
    # shift_change occurrence it actually is) can still pull in OTHER real
    # cues near each of its own (systematically shifted) predictions,
    # producing a chain that looks complete/well-scored per-slot but isn't
    # actually in sequence order -- caught this happening on match7 before
    # this check existed (a resulting teleop_start that landed AFTER its
    # own auto_end). Enforce strict chronological order across the whole
    # chain by dropping any slot that doesn't advance past the last kept
    # one; the earlier DP approach got this for free from its ordering
    # constraint, this is the equivalent guardrail here.
    last_t = float("-inf")
    for j in range(n):
        c = chain[j]
        if c is None:
            continue
        if c["t_sec"] <= last_t:
            chain[j] = None
            continue
        last_t = c["t_sec"]

    total_score = sum(c["score"] for c in chain if c is not None)
    found = sum(1 for c in chain if c is not None)
    return {"chain": chain, "total_score": total_score, "found": found}


def anchor_search(sequence: list[dict], candidates_by_file: dict[str, list[dict]],
                  scores_by_file: dict[str, np.ndarray], max_anchors: int = 20,
                  sr: int = SR, z_threshold: float = LOCAL_Z_THRESHOLD) -> dict:
    """Try candidates from the global pool (candidates_by_file, the
    already-extracted whole-track peaks) in descending score order as the
    anchor -- this step is deliberately still a global search, not local,
    since we have no prior on where the FIRST slot should be. For each
    anchor, try every slot it could plausibly be (its file may be used by
    more than one slot -- shift_change.wav x4, end.wav x2), evaluating each
    via evaluate_anchor's local presence test against scores_by_file (the
    raw score curves), and keep the best (found, score) hypothesis seen.
    Stops early once a fully complete chain is reached; otherwise tries up
    to `max_anchors` candidates and returns whichever attempt found the
    most slots (score as tiebreak)."""
    pool = build_candidate_pool(candidates_by_file)
    n = len(sequence)
    best = None
    for anchor_c in pool[:max_anchors]:
        slot_ids = [i for i, e in enumerate(sequence) if e["file"] == anchor_c["file"]]
        for idx in slot_ids:
            result = evaluate_anchor(sequence, scores_by_file, idx, anchor_c, sr, z_threshold)
            if best is None or (result["found"], result["total_score"]) > (best["found"], best["total_score"]):
                best = result
        if best and best["found"] == n:
            break
    if best is None:
        return {"chain": [None] * n, "total_score": None, "found": 0}
    return best


def anchor_with_runner_up(sequence: list[dict], candidates_by_file: dict[str, list[dict]],
                          scores_by_file: dict[str, np.ndarray], max_anchors: int = 20,
                          sr: int = SR, z_threshold: float = LOCAL_Z_THRESHOLD) -> dict:
    """Runner-up-margin confidence check: re-run with every timestamp the
    winner used masked out of both the candidate pool (so a different
    anchor gets picked) and the raw score curves (so local_presence can't
    just rediscover the same peak), so the runner-up is a genuinely
    different hypothesis, not the same chain with one slot swapped."""
    best = anchor_search(sequence, candidates_by_file, scores_by_file, max_anchors, sr, z_threshold)
    n = len(sequence)
    if best["found"] == 0:
        return {**best, "runner_up_total": None, "confidence": "low", "margin": None}

    used = [(sequence[j]["file"], c["t_sec"]) for j, c in enumerate(best["chain"]) if c is not None]
    stripped_candidates = {
        fname: [c for c in cs if not any(f == fname and abs(c["t_sec"] - t) < 1e-6 for f, t in used)]
        for fname, cs in candidates_by_file.items()
    }
    stripped_scores = {fname: arr.copy() for fname, arr in scores_by_file.items()}
    for fname, t in used:
        if fname in stripped_scores:
            stripped_scores[fname] = mask_scores_near(stripped_scores[fname], sr, t)
    runner = anchor_search(sequence, stripped_candidates, stripped_scores, max_anchors, sr, z_threshold)

    total = best["total_score"]
    runner_total = runner["total_score"] if runner["found"] > 0 else None
    margin = 1.0 if runner_total is None else (total - runner_total) / max(abs(total), 1e-9)
    complete = best["found"] == n
    confidence = "high" if (complete and margin >= CONFIDENCE_MARGIN_FRAC) else "low"
    return {**best, "runner_up_total": runner_total, "confidence": confidence, "margin": round(margin, 3)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, metavar="PATH")
    ap.add_argument("--year", type=int, default=DEFAULT_YEAR,
                    help=f"cue profile to use, data/cue_profiles/<year>.json (default: {DEFAULT_YEAR})")
    ap.add_argument("--z-threshold", type=float, default=LOCAL_Z_THRESHOLD,
                    help=f"local-background MAD-units a predicted window's peak must clear to "
                         f"count as found, see local_presence() (default: {LOCAL_Z_THRESHOLD})")
    ap.add_argument("--min-gap", type=float, default=MIN_PEAK_GAP_SEC)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--save", action="store_true", help="write data/<match>_phases.json")
    ap.add_argument("--viz", action="store_true")
    args = ap.parse_args()

    video_path = pathlib.Path(args.video)
    if not video_path.exists():
        sys.exit(f"[error] video not found: {video_path}")
    match_name = video_path.stem

    if not FIELD_AUDIO_DIR.exists():
        sys.exit(f"[error] {FIELD_AUDIO_DIR} not found")

    profile = load_cue_profile(args.year)
    sequence = profile["sequence"]
    abort_file = profile.get("abort_file", "abort.wav")
    print(f"[profile] year={profile['year']} (n={profile['built_from']['sample_size']})", file=sys.stderr)

    wav_path = extract_match_audio(video_path, no_cache=args.no_cache)
    match_audio = load_wav_mono(wav_path)
    print(f"[audio] match track: {len(match_audio) / SR:.1f}s @ {SR}Hz", file=sys.stderr)

    needed_files = sorted({e["file"] for e in sequence} | {abort_file})
    candidates: dict[str, list[dict]] = {}
    scores_by_file: dict[str, np.ndarray] = {}
    for fname in needed_files:
        p = FIELD_AUDIO_DIR / fname
        if not p.exists():
            print(f"[warn] missing reference clip: {p}", file=sys.stderr)
            continue
        ref = load_wav_mono(p)
        scores = matched_filter_scores(match_audio, ref)
        scores_by_file[fname] = scores
        candidates[fname] = get_all_candidates(scores, SR, args.min_gap)
        if fname != abort_file:
            print(f"[match] {fname}: {len(candidates[fname])} candidate(s)", file=sys.stderr)

    aborted, abort_score = False, None
    if abort_file in candidates and candidates[abort_file]:
        best_abort = max(candidates[abort_file], key=lambda c: c["score"])
        abort_score = best_abort["score"]
        aborted = abort_score >= ABORT_SCORE_THRESHOLD
        print(f"[abort] best score={abort_score:.3f}  aborted={aborted}", file=sys.stderr)

    result = anchor_with_runner_up(sequence, candidates, scores_by_file, sr=SR, z_threshold=args.z_threshold)
    n_slots = len(sequence)
    print(f"[anchor] confidence={result['confidence']} margin={result['margin']} "
          f"found={result['found']}/{n_slots} total={result['total_score']}", file=sys.stderr)

    phases, shift_changes = {}, []
    for e, c in zip(sequence, result["chain"]):
        if c is None:
            continue
        if e["name"] == "shift_change":
            shift_changes.append(c["t_sec"])
        else:
            phases[e["name"]] = c["t_sec"]

    print(f"[phases] {phases}", file=sys.stderr)
    print(f"[shift_changes] {shift_changes}", file=sys.stderr)

    output = {
        "match": match_name,
        "phases": {k: round(v, 3) for k, v in phases.items()},
        "shift_changes": [round(v, 3) for v in shift_changes],
        "confidence": result["confidence"],
        "margin": result["margin"],
        "runner_up_total": result["runner_up_total"],
        "found": f"{result['found']}/{n_slots}",
        "aborted": aborted,
        "abort_score": round(abort_score, 4) if abort_score is not None else None,
        "candidates": {fname: [{"t_sec": round(c["t_sec"], 3), "score": round(c["score"], 4)}
                               for c in cs] for fname, cs in candidates.items()},
    }

    if args.save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DATA_DIR / f"{match_name}_phases.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"[save] -> {out_path}", file=sys.stderr)

    if args.viz:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(needed_files), 1, figsize=(14, 2.0 * len(needed_files)), sharex=True)
        if len(needed_files) == 1:
            axes = [axes]
        chosen = set(phases.values()) | set(shift_changes)
        for ax, fname in zip(axes, needed_files):
            for c in candidates.get(fname, []):
                is_chosen = any(abs(c["t_sec"] - t) < 1e-6 for t in chosen)
                ax.axvline(c["t_sec"], color="green" if is_chosen else "orange",
                           alpha=0.9 if is_chosen else 0.35, linewidth=2 if is_chosen else 1)
                ax.plot(c["t_sec"], c["score"], "k.", markersize=3)
            ax.set_ylabel(fname, fontsize=8)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(match_name)
        fig.tight_layout()
        viz_path = AUDIO_DIR / f"{match_name}_cues.png"
        fig.savefig(viz_path, dpi=110)
        plt.close(fig)
        print(f"[viz] saved -> {viz_path}", file=sys.stderr)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Docs note: cue order + design history
# ---------------------------------------------------------------------------
# Sequence (start -> end -> resume -> shift_change x4 -> warning -> end)
# confirmed against a 2026 reference table (event names: MATCH start / AUTO
# ends / TELEOP & TRANSITION begins / ALLIANCE SHIFT starts x4 / END GAME
# begins / MATCH end / MATCH stopped) -- FIRST's public-facing themed cue
# names ("Cavalry Charge", "3 Bells", "POWER UP - Linear Popping", "Steam
# Whistle") differ from field_audio/*.wav's generic internal filenames but
# name the same cues in the same order.
#
# Ground-truth auto_start spot-checks (match1=5.77s, match2=5.97s,
# match4=~5s, match6=~6s, match7=~17s "really faint", match8=no real cue/
# video cut before it) drove the original two-stage design (a plain top-1-
# score pick got 3/6 right and failed on match2 -- a spurious early peak
# narrowly outscored the real cue -- and match7 -- the real cue simply
# wasn't in a top-8 list at all). Those same spot-checks were reused,
# unchanged, as ground truth for the sweep below. broadcast/ has never been
# committed to git (no history to fall back on), so the full pre-cleanup
# file -- this design's own account of itself, in its own words -- is
# archived verbatim at data/algo_assign_sweep/01_audio_pre_cleanup.py
# rather than left to git history that doesn't exist.
#
# ---------------------------------------------------------------------------
# Tried and rejected: algo x assign sweep (2026-08-17)
# ---------------------------------------------------------------------------
# This script originally exposed BOTH a choice of correlation algorithm
# (--algo: ncc/phat/roth, see matched_filter_scores's docstring) and a
# choice of candidate-assignment strategy (--assign: dp/anchor, see
# "Anchor-and-localize assignment" above), defaulting to ncc/dp. Before
# locking in a single default and deleting the losers, all 6 combinations
# were run against every match with cached audio (match1,2,4,6,7,8,9,10)
# and scored against two things this repo already treats as ground truth:
# (a) the cue profile's own "FMS-automatic, confirmed exact" gap means/
# stdevs (2026.json's own notes -- every gap except the one manual
# auto_end->teleop_start link), as |observed_gap - mean| / stdev per
# adjacent pair of found slots; and (b) the hand-verified auto_start spot-
# checks above. Harness + full per-match results archived at
# data/algo_assign_sweep/ (sweep_assign_algo.py + report.json) for anyone
# who wants to re-run the comparison (e.g. after adding more matches)
# rather than trust this summary blindly.
#
# On matches 1/2/4/6/9/10 (the profile's own 6-match training set -- the
# fairest comparison, since ground truth there IS the profile) every combo
# found all 9 slots, but gap-fidelity split algorithms cleanly:
#     algo/assign    mean |z|   max |z|
#     ncc/dp           0.745     1.678
#     ncc/anchor       0.745     1.678    <- tied: no ambiguity to resolve
#                                              differently on easy matches
#     phat/anchor      2.120     5.883
#     roth/anchor      1.475     3.243
#     phat/dp          3.328    18.118
#     roth/dp          6.873    39.438
# ncc beat both phat and roth under EITHER assignment strategy, confirming
# with numbers what the removed matched_filter_scores_phat/_roth functions'
# own docstrings already concluded by anecdote (PHAT over-whitens this
# cue's narrowband/comb-shaped spectrum; ROTH's milder down-weighting still
# underperforms no reweighting at all).
#
# The assignment strategies only diverged on the two hard matches, and it
# wasn't close:
#   - match7 (faint cues, excluded from profile-building): ncc/dp locked
#     onto a WRONG chain and reported HIGH confidence doing it (mean
#     z=47.7, max z=324, auto_start off by 16s) -- confidently wrong, the
#     exact failure mode anchor-and-localize was built to fix (see design
#     note above). ncc/anchor degraded gracefully instead: correctly
#     flagged LOW confidence, still landed auto_start within 1s of the
#     ~17s spot-check, and genuinely failed to find only 1 of 9 slots
#     rather than confidently mis-locating all of them.
#   - match8 (video literally cut before the auto_start cue plays -- no
#     correct answer exists for that ONE slot, per this profile's own
#     exclusion notes above). Every combo gets auto_start wrong here,
#     which is unavoidable and not a meaningful data point against anchor.
#     What DOES matter: ncc/anchor and roth/anchor independently landed on
#     IDENTICAL timestamps (to the millisecond) for all 7 of the OTHER
#     slots despite using different score curves -- strong cross-
#     validation that those are real detections. The one elevated gap
#     among them (shift1->shift2, z~15 in both) is the SAME absolute
#     timestamps in both, i.e. a real ~0.9s site-specific timing deviation
#     at this Texas district (already flagged in 2026.json as not
#     reliably following FIRST's national webcast guidelines), not a
#     detection error.
#
# Decision: ncc + anchor-and-localize, now the only path. matched_filter_
# scores_phat/_roth, the joint-DP assignment (solve_chain/
# chain_with_runner_up/best_pair_with_runner_up/build_back_half/
# auto_gap_prior/auto_duration_range/auto_lookback), and their --algo/
# --assign/--auto-lookback CLI flags were deleted. broadcast/ has never
# been committed to git, so (unusually) there's no git history to recover
# them from -- the full working file from just before this cleanup is
# instead archived verbatim at data/algo_assign_sweep/01_audio_pre_cleanup.py
# if the removed code is ever needed again.
