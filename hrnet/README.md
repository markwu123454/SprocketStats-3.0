# FRC Alliance-Keypoint Pipeline

Box-free heatmap keypoint detector: one dot per robot on the carpet under the
bumper center, colored by alliance (blue / red). Built for season-agnostic use
(train on 2022–2026, deploy 2027+).

## Model

**HRNet-W32 backbone + 2-channel Gaussian heatmap head (CenterNet-style).**

- Channel 0 = blue alliance, channel 1 = red. Each is a heatmap with a Gaussian
  peak at every keypoint of that alliance.
- No boxes, no anchors, no association. Inference = per-channel local-max
  peak-finding. This is what makes heavy robot overlap and airborne points
  (dot outside any robot box) tractable — none of it depends on a box.
- HRNet keeps high-resolution features throughout, which is why it localizes
  small points well across the huge scale range (near-ground to birds-eye).

Why not YOLO-pose: it requires each keypoint to be associated with a parent
robot box. Your points routinely violate that (overlap ambiguity, climbing
robots whose floor point sits outside their box), so the box paradigm is out.

## Pipeline

```
R2 (150GB images) ──┐
                    ├─► 01_pull.py ─► exports/manifest.jsonl + local image mirror
Label Studio ───────┘
                              │
                              ▼
                        02_train.py ─► checkpoints/best.pt   (season-holdout val)
                              │
                              ▼
                        03_predict.py ─► predictions pushed back to Label Studio
                                          (humans then correct → re-export → retrain)
```

### One-time setup

```bash
pip install label-studio-sdk boto3 timm scipy opencv-python-headless \
            torch torchvision pyyaml numpy
```

Set environment variables (never commit these):

```bash
export LS_API_KEY=...              # LS personal access token
export R2_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com
export R2_ACCESS_KEY_ID=...
export R2_SECRET_ACCESS_KEY=...
```

Edit `config.yaml` for bucket name, local paths, and `val_season`.

### Step 1 — pull labels + images

```bash
python scripts/01_pull.py
```

Creates an async export snapshot (required at 112k tasks — the synchronous
export times out), downloads it, parses keypoint annotations, converts LS
percent-coordinates to pixels, tags each frame with its season (parsed from the
R2 key), and mirrors the referenced images locally.

### Step 2 — train

```bash
python scripts/02_train.py
```

Holds out the entire `val_season` for validation (not random frames — 0.5fps
frames from one match are near-duplicates and would leak). Reports PCK-style
precision/recall/F1 with one-to-one same-alliance matching. Saves `best.pt`.

### Step 3 — predict back into Label Studio

```bash
python scripts/03_predict.py --only-unannotated
```

Runs the model on unlabeled tasks and writes predictions in the project's exact
KeyPointLabels format (percent coords, "Blue robot center" / "Red robot center").
Annotators then correct the pre-placed dots instead of placing from scratch.

## Active-learning loop

1. Hand-label a few thousand frames spanning seasons + hard slices.
2. Train (step 2).
3. Pre-label the rest (step 3).
4. Humans correct; re-export (step 1); retrain. Repeat.

Each round the model gets better, so corrections get faster.

## Tuning knobs (config.yaml → heatmap:)

- `sigma`: peak width. Larger = easier to learn, less precise. Start 2.0.
- `peak_threshold`: raise to cut false positives, lower to catch faint robots.
- `nms_kernel`: min separation between two same-alliance peaks. At stride 4,
  kernel 5 resolves robots ~8px apart in input space.
- `input_size`: raise (e.g. 1024) if the smallest robots are missed; costs VRAM.

## Note on the Label Studio ML backend

The prebuilt Ultralytics ML backend speaks box+associated-keypoint pose, which
this project does not use, so it does not apply. `03_predict.py` replaces it:
it pushes predictions directly through the SDK. If you want live pre-labeling
(predict-on-import rather than batch), wrap `predict_image()` in a small
label-studio-ml-backend `predict()` method returning the same result dicts —
same format, just served over HTTP instead of batched.
```
