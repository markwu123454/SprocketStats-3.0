# MTMCT / Tracking

All tracking lives here: linking robot detections into tracks, cutting tracks
where they may have swapped robots, and running the end-to-end pass that ends
with every track named. Naming itself is ocr's one job (`../ocr/`): it gets
finished tracks and their bumper reads, and answers with a team, unknown, or
"not a robot" per track.

Today this is single-view and in image pixels, driven by HRNet dots. The root
README's design (field-space tracking, YOLO boxes for identity, multi-view
fusion) isn't built yet.

## run/

- `tracker.py` — frame-to-frame association of robot dots into tracks that
  never jump.
- `split_tracks.py` — cuts after association: where confident bumper reads
  switch team (`split_by_reads`), and where geometry says a track could have
  swapped onto another same-alliance robot (`split_ambiguous`).
- `track_robots.py` — the end-to-end pass: spotter → tracker → bumper reads →
  cuts → naming, saved to `data/tracks/<match>.json`. `--relabel` redoes the
  cuts and naming from a saved run without OCR.

## debug/

- `render_tracks.py` — the labelled video and stills, rendered from a saved run.
- `test_tracks.py` — tracking, cutting, battery guard and label placement tests.

## Why ocr and mtmct import each other

Cutting a track needs bumper reads, and naming needs the cut tracks, so the
work alternates by step: mtmct tracks, ocr reads, mtmct cuts, ocr names.
`track_robots.py` calls into `ocr.run`, and `ocr/run/assign_teams.py` imports
the shared track geometry (`DUP_DIST`, `STITCH_GAP`, `frames_index`) from
`mtmct.run`, so both sides agree on what "the same dot" and "a hand-over" mean.
No single file imports in a circle.
