# broadcast

Extracts structured match events from the broadcast audio and CG scoreboard
overlay. Provides ground-truth timing and scoring signals that other pipelines
can align against.

The two main signals are complementary: audio gives low-latency phase transition
cues (start/end buzzers are unmistakable), and the CG overlay gives ground-truth
score values (useful for validating CV-derived scoring estimates).

## Inputs

- Match video with audio (the same source used by homography/ and tracking/)
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

**Penalty events** — score increases that don't correspond to a known scoring
action (useful as a foul signal for match prediction).

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

- The FRC game buzzer is acoustically distinctive and consistent year to year —
  a short burst of a specific frequency. Audio detection should be reliable even
  with crowd noise.
- CG scoreboard position and font vary by broadcast production team (regional
  vs. district vs. championship). The crop region will need calibration per
  event, similar to how homography needs a profile per match.
- Score OCR is the fragile part — graphics sometimes animate in/out, and digits
  can be partially occluded during score updates. Aggregate over a small window
  (±3 frames) and take the most-common reading.
- Score deltas are alliance-level only (same limitation as TBA data). Attribution
  to individual robots still requires CV-based scoring detection.
