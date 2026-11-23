# broadcast

Extracts structured match events from the broadcast audio and CG scoreboard
overlay. Provides ground-truth timing and scoring signals that other pipelines
can align against.

The two main signals are complementary: audio gives low-latency phase transition
cues (start/end buzzers are unmistakable), and the CG overlay gives ground-truth
score values (useful for validating CV-derived scoring estimates).

## Inputs

- Match video with audio (the same source used by homography/ and mtmct/)
- Optionally: TBA match result (for post-hoc validation of extracted scores)

## What this extracts

**Phase transitions** — the precise frame/timestamp when each match phase
begins and ends: pre-match → auto → teleop → endgame → post-match.
Primary signal: audio (start buzzer, end buzzer, announcer cues).
Secondary signal: CG overlay countdown timer / phase indicator.

**Score update events** — each time either alliance's displayed score changes,
record `{t_sec, alliance, old_score, new_score, delta}`. The delta tells you
when a scoring event occurred and approximately how many points it was worth,
without knowing which robot caused it.

## run/ — per match

`run/__init__.py` lists the order. Each step's module docstring explains its
method and what was tried before it.

- `detect_phases.py` — matches the official FMS audio cues (`data/field_audio/`)
  against the broadcast audio, anchored on the strongest cue and checked against
  the year's cue timing (`data/cue_profiles/<year>.json`).
  Output: `data/<match>_phases.json`.
- `split_views.py` — finds the 2–3 camera feeds embedded in the frame (the
  regions whose pixels keep changing). Output: `data/<match>_views.json`, read
  by `homography/run/detect_tags.py`.
- `detect_overlay.py` — finds every overlay region that holds a changing number,
  by behaviour, without deciding what it is. Output: `data/<match>_regions.json`.
- `read_overlay.py` — reads every region on every settled frame by template
  search against the digit bank. Output: `data/<match>_regions_timeline.json`.
- `identify_fields.py` — decides what each region means (scores, clock,
  counters) and cuts the timeline to the match window. Output: `data/<match>_match.json`.
- `layouts.py` — hand-measured per-event layout: the field outline, overlay
  boxes holding printed digits, and boxes where spotter dots are ignored even
  inside the field. mtmct uses it to decide which
  robot dots count, ocr to mask digits before reading bumpers.
- `field_crops.py` — the ink and change tests that run/ and train/ must agree
  on (digit glyphs and game-piece icons are matched against templates built
  with the same thresholds).

## train/ — rebuilding what run/ reads

- `harvest_digits.py` → `build_digit_templates.py` — collect real digit glyphs
  from broadcast video, then fuse them into `data/digit_templates/`.
- `build_icon_templates.py` — game-piece icon templates from sourced art
  (`data/icon_templates_src/` → `data/icon_templates/`).

## debug/

- `test_layouts.py` — per-event layout tests.

`detect_phases.py` and `detect_overlay.py` still take a `--viz` flag that draws
their own intermediate results.

## Design notes

- Phases come from matching the official FMS audio cues, not from a generic
  buzzer detector. Plain normalized cross-correlation (NCC) beat GCC-PHAT and
  ROTH-weighted GCC on the broadcast audio; the sweep is in `detect_phases.py`.
- Overlay regions are found by behaviour (`detect_overlay.py`), so the
  scoreboard needs no per-event crop calibration. `layouts.py` is still
  hand-measured per event, but it covers the field outline, printed-digit boxes
  and spotter ignore boxes, not the scoreboard.
- Score reading is the fragile part — graphics animate in and out, and digits
  can be partly hidden during score updates. `read_overlay.py` reads only
  settled frames and takes a majority vote over each run of stable frames.
  Voting suppresses transient errors but not systematic ones: its `agreement`
  value measures stability, not correctness.
- Score deltas are alliance-level only (same limitation as TBA data). Attribution
  to individual robots still requires CV-based scoring detection.
