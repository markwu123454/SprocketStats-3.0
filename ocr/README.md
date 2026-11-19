# OCR / team assignment

One job: given finished robot tracks (from `../mtmct/`), name each one with a
team number, leave it unknown, or drop it as spotter noise. Team numbers are
never decoded from pixels: TBA gives the six teams, and each bumper crop is
scored against those six candidates (README §6, "Score candidates, don't
decode").

## run/

- `read_bumpers.py` — crops around each tracked dot, finds bumper text, and
  keeps the best accepted read per robot per frame. `is_confident` decides
  which reads are allowed to cut a track (used by mtmct).
- `team_scorer.py` — PP-OCRv6's CTC lattice scored against the candidate teams.
- `assign_teams.py` — naming: reads summed along each track under the
  one-team-per-robot and alliance-colour constraints, then elimination,
  bridging and stitching; `mark_noise` drops tracks that can't be a robot.
- `ocr_engine.py` — PaddleOCR on the Transformers engine, with its model
  cache kept in `ocr/.cache`.

## debug/

- `probe_bumper_reads.py` — whole-frame OCR probe on a few 2026 matches.
- `check_ocr_model.py` — GPU smoke test for the OCR model.
- `test_assign_teams.py` — naming tests.

The end-to-end run is `python -m mtmct.run.track_robots`.
