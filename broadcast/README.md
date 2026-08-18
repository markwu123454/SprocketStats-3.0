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

## Pipeline

`pipeline/01_audio.py` — detect match phase transitions from audio. Extract
the audio track, run onset/energy analysis to find the characteristic buzzer
pattern at match start and end. Output timestamps for each phase boundary.

`pipeline/02_ocr_score.py` — sample frames at regular intervals, crop the
CG scoreboard region (position varies by broadcast; may need a per-event
calibration), run OCR to read alliance scores, and emit a score timeline.
Output: `data/<match>_scores.jsonl` — `{frame, t_sec, blue_score, red_score}`

`pipeline/03_events.py` — merge audio timestamps and score timeline into a
unified event stream. Align phase boundaries with score-change events to
produce the final output.
Output: `data/<match>_events.json` — `{phases: {auto_start, teleop_start, ...},
score_events: [{t_sec, alliance, delta, cumulative}]}`

## Viz

`viz/03_events_view.py` — feeds from 03_events output. Timeline plot of both
alliance scores over the match with phase boundaries marked. Quick visual check
that the extracted events look reasonable.

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
