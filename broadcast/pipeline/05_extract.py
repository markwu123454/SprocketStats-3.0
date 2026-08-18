#!/usr/bin/env python3
"""
Step 4 -- Extraction: walk a full match and read the live score-change
timeline (blue/red score, clock, badge/game-piece counts) via TEMPLATE
MATCHING against the digit bank 04_build_templates.py built, instead of
OCR. This is the payoff for calibration existing as its own stage at all --
see 02_detect_overlay.py's module docstring: "Calibration ... so a later
extraction step doesn't need OCR/GPU per frame." No easyocr/torch import
anywhere in this file.

Reuses 03_calibrate.py's settle-detection (FieldTracker/_looks_changed) and
glyph segmentation (segment_glyphs) verbatim via importlib -- both are
already validated (03's own run logged single-digit-percent mismatch
rates against real match footage) and the "when has a field settled into a
new state" problem is identical here, just consumed differently (classify
each settled glyph instead of harvesting it as a template instance).

Algorithm
---------
1. Same sequential stride-sampled video walk as 03. Each field's tracker
   fires once it settles into a state that differs from the last ACCEPTED
   one (see jump-rejection below for what "accepted" means).
2. On fire: segment_glyphs the crop into individual glyph boxes (same
   polarity rules as calibration). For clock, apply the SAME inversion
   04_build_templates.py applied when building clock's templates (dark-on-
   light -> bright-on-dark) so the query and the bank are in the same
   convention -- forgetting this would silently compare against the wrong
   polarity and never match well.
3. Classify each glyph independently: resize it (cubic) to each candidate
   digit template's own (h, w) and score with cv2.matchTemplate's
   TM_CCOEFF_NORMED on the same-size pair (a single-point normalized cross-
   correlation, not a sliding search -- there's nowhere to slide when both
   inputs are the same size, which is exactly what's wanted here: "how well
   does this glyph match digit N's canonical shape", not "where in this
   glyph does digit N's shape occur"). Highest score wins.
4. Concatenate the classified glyphs left-to-right into a value string,
   parse it (int for score/badge, seconds for clock -- see
   03_calibrate.py's clock docstring for the mm/ss split rule, reused
   as-is).
5. A capture whose WORST glyph score falls below MIN_GLYPH_CONFIDENCE is
   flagged low_confidence=true but NOT dropped by itself -- there's no OCR
   fallback anymore to cross-check against (that dependency is exactly what
   this file exists to remove), and it's known to have false negatives (see
   jump-rejection below), so it's kept as an informational signal rather
   than the primary defense.

Scan boundary: 01_audio.py's match_end, not a fixed fraction
--------------------------------------------------------------
Raised directly in chat: 03/02's generic END_FRAC=0.85 dodge is a blunt
guess at where post-match content starts, and it isn't tight enough --
confirmed on match2, where a broadcast scene-transition wipe (into
whatever plays after "Einstein Final Tiebreaker" ends) corrupted several
fields' crops simultaneously at t=176.56s, well inside the 0.85 window.
01_audio.py already solves "when does this specific match actually end"
per-match, from the field's own end-of-match audio cue (data/<match>_
phases.json's phases.match_end) -- far sharper than a fraction of total
video length. When that file exists and reports "confidence": "high",
extraction stops at match_end + TAIL_MARGIN_SEC (5s: enough margin to
still catch a score/clock settling into its true final value if the
overlay updates with a short lag after the buzzer, not so much that it
wanders deep into post-match graphics). Falls back to the old END_FRAC
fraction when no phases file exists (match3 -- 2025 REEFSCAPE audio cues
aren't in the 2026 cue profile 01 uses) or confidence isn't "high" (match7
-- 8/9 cues found, "low" confidence recorded by 01 itself).

Even with that tighter cutoff, match2's transition still falls inside the
5s tail margin (176.56s vs match_end=171.851s) -- so the tail can still be
garbage, just a shorter, bounded stretch of it. That's what jump-rejection
(next section) actually defends against.

Jump rejection: reject by MAGNITUDE, not direction
-----------------------------------------------------
Also settled directly in chat, through a real correction: the first draft
of this file assumed score/badge values only ever increase and used that
to silently prefer the max value seen. That assumption is wrong -- FRC
referees occasionally undo a scored point live, so a real match CAN show a
small decrease. The right filter is on how BIG a single settle-to-settle
jump is, not which direction it goes: for score/badge fields (NOT clock --
see below), if a newly classified value differs from the last ACCEPTED
value by more than JUMP_REJECT_THRESHOLD in EITHER direction, the reading
is rejected -- recorded in the timeline (rejected=true) for visibility,
but it does NOT become the new last-accepted value, and critically, the
tracker's crop baseline (last_captured_crop) is NOT advanced to the
rejected frame's crop either. That last part is what makes "discard the
current frame but keep going" actually work: the NEXT sampled frame keeps
comparing against the last GOOD crop, not the garbage one, so it's
compared as "still different from what we last trusted" and gets a fresh
classification attempt rather than the corrupted frame quietly becoming
the new normal baseline.

This is also strictly more robust than relying on low_confidence alone:
match2's blue_badges misread jumped 403 -> 13 (a swing of 390, rejected on
magnitude) while its own per-glyph match score was 0.75 -- ABOVE
MIN_GLYPH_CONFIDENCE, so low_confidence alone would have missed it
entirely. With jump-rejection in place, "final value" per field is simply
the last ACCEPTED event -- no special-casing needed, and a legitimate
small referee-undo decrease passes through exactly like any other real
change would.

Clock is explicitly EXEMPT from jump-rejection: it legitimately jumps by
more than any reasonable magnitude threshold at every period boundary
(e.g. 20 -> 0 as auto ends, then straight up to the full teleop duration
as teleop starts) -- a magnitude-only cap would reject those correct
transitions along with real corruption. Clock still gets a
low_confidence flag, just not jump-filtered.

Recovery: a rejected jump has to repeat before it's trusted
-----------------------------------------------------------
Found on the very first real run of the jump-rejection logic above,
against match2: a rejected reading NEVER updates last_accepted_value, so
if the classifier ever produces a WRONG reading that happens to fall
within JUMP_REJECT_THRESHOLD of the current (correct) baseline, that wrong
value gets silently ACCEPTED and becomes the new baseline -- and every
correct reading after it, being far from this now-wrong baseline, gets
rejected too. With no recovery, this is permanent for the rest of the
match: exactly what happened to match2's blue_badges, which got stuck near
~8-30 for the whole second half of the match after one coincidental
garbage-but-small-delta misread, while blue_score/red_score/red_badges (no
such coincidence) tracked correctly the whole way through.

The fix: a rejected value is only DISCARDED once. If the same value
(within JUMP_CONFIRM_TOLERANCE) is read again on the NEXT rejected attempt
for that field -- JUMP_CONFIRM_STREAK times in a row -- it's accepted
despite the jump, and last_accepted_value resyncs to it. This is the same
"don't trust a single sample, trust two that agree" idea 03_calibrate.py's
settle-detection already uses for crop stability, just applied one level
up: a one-off misread won't repeat itself, but a stale baseline that
needs to catch up to a real, large, legitimate value will keep reading
close to that same real value on every subsequent attempt.

Output
------
data/<match>_score_timeline.json -- {"match", "fps", "hi_bound_source",
"match_end_sec", "events": [...], "finals": {...}}. Each event: {"frame",
"t_sec", "field", "kind", "raw", "value", "low_confidence", "rejected",
"reject_reason"}. `raw` is the concatenated digit string as classified;
`value` is it parsed (int, or total seconds for clock -- null if parsing
failed, e.g. clock seconds >= 60). `finals`: the last ACCEPTED event per
field (absent if a field had zero accepted events).

Known limitations / unvalidated
--------------------------------
- JUMP_REJECT_THRESHOLD=50 and TAIL_MARGIN_SEC=5 are starting points
  agreed in chat, not swept -- a legitimate single-frame score swing larger
  than 50 (multiple simultaneous scoring actions landing in the same
  settle window) would currently be wrongly rejected. Not observed in the
  8 matches processed so far, but not structurally impossible.
- match3 and any low-audio-confidence match (match7) still use the old
  END_FRAC=0.85 fraction bound, with all of ITS original caveats (a
  contaminated frame inside that wider window has no defense at all).
- MIN_GLYPH_CONFIDENCE is a starting point, picked by looking at real
  match/template match-score distributions once (see validation run in
  chat), not swept, and known to have false negatives (see jump-rejection
  section) -- kept as an informational flag, not relied on alone anymore.
- A rejected reading's underlying cause (segmentation missing a glyph vs.
  a genuinely different but implausible classification) isn't
  distinguished -- reject_reason is currently only ever "jump>N".

Usage
-----
  python pipeline/05_extract.py --match match1 --save
"""

import argparse, importlib.util, json, pathlib, sys, time

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


calib = _load_calibrate()

STRIDE = 2  # denser than 03's -- no OCR/GPU cost per frame anymore, cheap to scan more thoroughly
START_FRAC = 0.0
END_FRAC = 0.85  # fallback only -- see load_phases()/module docstring for the preferred match_end-based bound

# See module docstring "Scan boundary" section.
TAIL_MARGIN_SEC = 5

# Worst-glyph NCC score below which a capture is flagged low_confidence
# (kept, not dropped -- see module docstring). Picked from eyeballing real
# match1 match-score distributions (correct reads clustered >0.75, the
# handful of genuinely ambiguous glyphs sat lower) -- not swept broadly.
MIN_GLYPH_CONFIDENCE = 0.6

# See module docstring "Jump rejection" section. Points, either direction,
# score/badge kinds only -- never applied to clock.
JUMP_REJECT_THRESHOLD = 50

# Recovery from a stale/wrong baseline (see module docstring): a rejected
# jump is accepted anyway once the SAME value (within this tolerance) has
# been read on JUMP_CONFIRM_STREAK consecutive rejected attempts in a row.
# Tolerance is a few points wide, not exact-match, since the real value can
# tick up slightly between the confirming reads if scoring is still active.
# Both starting points, not swept.
JUMP_CONFIRM_TOLERANCE = 5
JUMP_CONFIRM_STREAK = 2


# ---------------------------------------------------------------------------
# Template bank
# ---------------------------------------------------------------------------

def load_templates() -> dict[str, dict[str, np.ndarray]]:
    bank = {}
    for kind_dir in sorted(TEMPLATES_DIR.iterdir()):
        if not kind_dir.is_dir():
            continue
        digits = {}
        for f in kind_dir.glob("[0-9].png"):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                digits[f.stem] = img.astype(np.float32)
        if digits:
            bank[kind_dir.name] = digits
    return bank


def classify_glyph(glyph_gray: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[str, float]:
    """Best-matching digit for one glyph crop, via same-size normalized
    cross-correlation against every candidate template -- see module
    docstring step 3."""
    best_char, best_score = "?", -2.0
    for ch, tmpl in templates.items():
        h, w = tmpl.shape
        resized = cv2.resize(glyph_gray, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        score = float(cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
        if score > best_score:
            best_score, best_char = score, ch
    return best_char, best_score


def classify_field(frame_bgr: np.ndarray, glyph_boxes: list[list[int]], templates: dict[str, np.ndarray],
                   invert: bool) -> tuple[str, float]:
    chars, worst = [], 2.0
    for gx0, gy0, gx1, gy1 in glyph_boxes:
        crop = cv2.cvtColor(frame_bgr[gy0:gy1, gx0:gx1], cv2.COLOR_BGR2GRAY)
        if invert:
            crop = 255 - crop
        ch, score = classify_glyph(crop, templates)
        chars.append(ch)
        worst = min(worst, score)
    return "".join(chars), worst


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

def parse_value(field: str, raw: str) -> int | None:
    if not raw or "?" in raw:
        return None
    if field == "clock":
        return parse_clock_seconds(raw)
    try:
        return int(raw)
    except ValueError:
        return None


def parse_clock_seconds(digit_str: str) -> int | None:
    """Same mm/ss split rule as 02_detect_overlay.py's parse_clock_seconds
    -- last two digits are seconds, the rest (if any) are minutes."""
    if len(digit_str) <= 2:
        minutes, seconds = 0, int(digit_str)
    else:
        minutes, seconds = int(digit_str[:-2]), int(digit_str[-2:])
    if seconds >= 60:
        return None
    return minutes * 60 + seconds


# ---------------------------------------------------------------------------
# 01_audio.py's phase output -- see module docstring "Scan boundary"
# ---------------------------------------------------------------------------

def load_phases(match: str) -> dict | None:
    p = DATA_DIR / f"{match}_phases.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Per-match extraction
# ---------------------------------------------------------------------------

def extract_match(match: str, bank: dict[str, dict[str, np.ndarray]]) -> dict:
    boxes_path = DATA_DIR / f"{match}_overlay_boxes.json"
    video_path = ROOT.parent / f"{match}.mp4"
    boxes = json.loads(boxes_path.read_text())["boxes"]

    trackers = calib.build_trackers(boxes)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo = int(total * START_FRAC)
    frac_hi = int(total * END_FRAC)

    phases = load_phases(match)
    match_end_sec = None
    hi, hi_source = frac_hi, f"frac({END_FRAC})"
    if phases and phases.get("confidence") == "high":
        match_end_sec = phases.get("phases", {}).get("match_end")
        if match_end_sec is not None:
            phase_hi = int((match_end_sec + TAIL_MARGIN_SEC) * fps)
            hi = min(frac_hi, phase_hi)
            hi_source = f"phase(match_end={match_end_sec}+{TAIL_MARGIN_SEC}s)"

    events = []
    last_value = {}            # (raw, value) of the last ACCEPTED read per field, for dedup
    last_accepted_value = {}   # numeric value of the last ACCEPTED read per field, for jump-rejection
    pending_jump = {}          # name -> {"value", "count"}, streak-tracking for jump recovery

    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    t_start = time.time()
    # Reuses each FieldTracker's settle-detection STATE (prev_crop/
    # last_captured_crop) and STATELESS helpers (_looks_changed,
    # segment_glyphs) from 03_calibrate.py directly, but NOT its
    # offer()/_capture() methods -- those hardcode an OCR value read, which
    # is exactly the dependency this file exists to remove. Reimplementing
    # the settle check inline (rather than adding an OCR-free code path to
    # FieldTracker itself) keeps 03's file focused on collection only.
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
            if not calib._looks_changed(t.last_captured_crop, crop):
                t.prev_crop = crop
                continue
            if not calib._looks_changed(t.prev_crop, crop):
                glyph_boxes = calib.segment_glyphs(frame, t.box, t.polarity)
                templates = bank.get(t.kind, {})
                if glyph_boxes and templates:
                    raw, worst = classify_field(frame, glyph_boxes, templates, invert=(t.kind == "clock"))
                    value = parse_value(t.kind, raw)

                    rejected, reject_reason = False, None
                    if t.kind in ("score", "badge") and value is not None:
                        prev = last_accepted_value.get(t.name)
                        if prev is not None and abs(value - prev) > JUMP_REJECT_THRESHOLD:
                            rejected, reject_reason = True, f"jump>{JUMP_REJECT_THRESHOLD}"

                    if rejected:
                        # A rejected jump is confirmed (and then ACCEPTED
                        # despite its size) if the same value keeps
                        # reappearing on consecutive rejected attempts --
                        # distinguishes a one-off misread (never repeats,
                        # stays rejected forever) from a stale baseline that
                        # needs to resync to a real, large, legitimate
                        # change. Found necessary on match2's blue_badges:
                        # without this, one small-delta GARBAGE read got
                        # accepted by coincidence and froze the baseline for
                        # the rest of the match, permanently rejecting every
                        # correct large catch-up read after it (each looked
                        # like a huge jump from the now-wrong frozen
                        # baseline) -- see module docstring.
                        pend = pending_jump.get(t.name)
                        if pend is not None and abs(value - pend["value"]) <= JUMP_CONFIRM_TOLERANCE:
                            pend["count"] += 1
                            pend["value"] = value
                        else:
                            pend = {"value": value, "count": 1}
                        pending_jump[t.name] = pend
                        if pend["count"] >= JUMP_CONFIRM_STREAK:
                            rejected, reject_reason = False, None
                            del pending_jump[t.name]
                    else:
                        pending_jump.pop(t.name, None)

                    if rejected or last_value.get(t.name) != (raw, value):
                        events.append({
                            "frame": frame_idx, "t_sec": round(frame_idx / fps, 2),
                            "field": t.name, "kind": t.kind, "raw": raw, "value": value,
                            "low_confidence": worst < MIN_GLYPH_CONFIDENCE,
                            "rejected": rejected, "reject_reason": reject_reason,
                        })
                    if not rejected:
                        # Only an ACCEPTED read updates the crop baseline --
                        # a rejected frame must NOT become the new "normal"
                        # to compare future frames against (see module
                        # docstring's "discard the current frame but keep
                        # going" explanation).
                        last_value[t.name] = (raw, value)
                        if value is not None:
                            last_accepted_value[t.name] = value
                        t.last_captured_crop = crop
            t.prev_crop = crop
        if (frame_idx - lo) % 500 == 0:
            # Progress heartbeat -- with no visible output otherwise, a slow
            # stretch (e.g. a rapid-scoring flurry triggering classification
            # on nearly every sampled frame -- observed on match2's first
            # ~15s) is indistinguishable from a genuine hang from the
            # outside. Found the hard way debugging this file: burned real
            # time twice killing a process that was just slow, not stuck,
            # because there was nothing printed to tell the difference.
            pct = 100 * (frame_idx - lo) / max(1, hi - lo)
            print(f"[scan] {match}: frame {frame_idx}/{hi} ({pct:.0f}%), "
                  f"{len(events)} events so far, {time.time() - t_start:.0f}s elapsed", file=sys.stderr)
    cap.release()

    events.sort(key=lambda e: e["frame"])
    finals = {e["field"]: e for e in events if not e["rejected"]}
    return {"match": match, "fps": fps, "hi_bound_source": hi_source, "match_end_sec": match_end_sec,
            "events": events, "finals": finals}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", required=True)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    bank = load_templates()
    if not bank:
        sys.exit(f"[error] no templates under {TEMPLATES_DIR} -- run 04_build_templates.py first")
    print(f"[bank] kinds: {list(bank.keys())}", file=sys.stderr)

    result = extract_match(args.match, bank)
    events = result["events"]
    n_rejected = sum(1 for e in events if e["rejected"])
    n_low = sum(1 for e in events if e["low_confidence"])
    print(f"[extract] {args.match}: {len(events)} events ({n_rejected} rejected as implausible jumps, "
          f"{n_low} low-confidence), scan bound: {result['hi_bound_source']}", file=sys.stderr)

    all_fields = sorted({e["field"] for e in events})
    for field in all_fields:
        e = result["finals"].get(field)
        if e is None:
            print(f"[final] {field}: NO accepted reads", file=sys.stderr)
        else:
            print(f"[final] {field}: raw={e['raw']!r} value={e['value']} (frame {e['frame']}, "
                  f"t={e['t_sec']}s)", file=sys.stderr)

    if args.save:
        out_path = DATA_DIR / f"{args.match}_score_timeline.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[save] -> {out_path}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
