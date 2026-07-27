# MTMCT / Tracking

Multi-target multi-camera tracking, in progress. Consumes per-frame
keypoint detections from `../HRNet-W32/` and is expected to produce a
per-match track manifest (persistent per-robot track IDs + positions over
time) consumed by `../action/` for action classification.

See `../action/01_train.py` for the manifest format `../action/` expects
this pipeline to eventually emit.
