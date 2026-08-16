# AprilTag Detection Research

Broadcast FRC footage is hard for standard detectors: tags appear at 15–50 px,
shot through long lenses, compressed with H.264, and sometimes motion-blurred.
This document records everything tried to detect them reliably.

---

## The problem

The homography pipeline needs to know where AprilTags are in each camera view
so it can solve for camera pose via PnP. The field has 32 tags; we need as many
as possible decoded per frame. Tags that fail to decode but whose corners are
correctly located are also useful (they can be matched to known field positions
geometrically).

Constraints:
- Broadcast footage only — no controlled camera, no known intrinsics up front
- Tags appear at 15–50 px apparent size in the main wide-angle view
- H.264 compression smears the bit cells inside the tag
- Sometimes 0–1 tags decode per frame; collinear decoded tags can't seed solvePnP

---

## Approaches tried

### 1. ArUco (OpenCV) — baseline

**How it works:** adaptive thresholding → find square blobs → decode bit pattern.

**Results:** 0–4 tags decoded per frame. Also exposes `rejectedImgPoints`: quads
it found but couldn't decode — ~800 per frame on broadcast footage.

**Problem with rejected candidates:** the 800 candidates are almost all false
positives (crowd, logos, robot panels, field markings). Tried two filters on top:

- **Persistence clustering + parallelogram score** — cluster rejected quads by
  pixel position across frames (real static tags recur; clutter wanders), score
  each cluster by how parallelogram-like its averaged quad is. Reduced the pool
  but still left hundreds of scored candidates with no way to assign IDs without
  a pose estimate. Chicken-and-egg: need IDs to get pose, need pose to assign IDs.

- **White-threshold ring filter** — FRC tags have a thick white quiet zone around
  the black data area. Threshold the frame to keep only bright pixels, then find
  ring-shaped (hollow) white contours using contour hierarchy. Score by aspect
  ratio, bounding box size, hollow ratio, rectangularity, and convex fill.
  Partially worked — high-confidence rings visible at correct locations — but
  too many false positives from arena structure, sponsor logos, jerseys.

  Also tried: drawing all quad-shaped contours (white quads = green, black quads
  = red) using convex hull + morphological closing to handle rounded corners and
  broken edges. Showed tag structure but missed too many and was noisy.

### 2. Direct bit decoding from candidate quads

For each high-confidence candidate quad: rectify to 64×64 using
`getPerspectiveTransform`, sample the 36h11 bit grid, try to match against
the 587-entry codebook.

**Result:** no better than ArUco. At 20–40 px tag size with H.264 compression,
each bit cell is ~3 px. Compression smears the boundaries; no amount of sampling
strategy recovers the bits reliably. Dead end at this resolution.

### 3. YOLO nano (object detection)

Labeled 3 frames with bounding boxes, fine-tuned YOLOv8-nano.

**Result:** ~50% recall after 3 labeled frames — promising start, but:
- Labeling bounding boxes is slow and tedious
- YOLO conflates localization and classification; we only need classification
  since we already have candidate regions from other methods
- Felt like overkill for the problem

Not pursued further, but worth revisiting if AT3 hits a ceiling.

### 4. AprilTag 3 — pupil-apriltags ✓ adopted

**How it works:** gradient-based quad detection instead of thresholding.
Computes image gradients, clusters pixels by gradient direction, uses union-find
to group into regions, fits quads to region boundaries. Fundamentally more
robust to compression artifacts and blur because it operates on edges rather
than thresholded pixel values.

**Installation:** `pip install pupil-apriltags`

**Parameter search (on broadcast frames where ArUco gets 0–1 decoded):**

| quad_sigma | decode_sharpening | tags decoded |
|---|---|---|
| 0.0 | 0.25 | 5 |
| 0.0 | 1.25 | **5, high margin (249–454)** |
| 0.8 | 0.25 | 3 |
| 0.8 | 1.25 | 4 |
| 1.2 | 1.25 | 2 |
| 1.8+ | any | 0 |

**Key findings:**
- `quad_sigma=0.0` (no pre-blur) is best. The gradient-based quad finder does
  not need blur; adding it smears the bit cells before decode.
- `decode_sharpening=1.25` (aggressive) consistently recovers more compressed
  tags. Higher sharpening of the tag interior before bit decoding helps.
- `quad_decimate=1.0` (no subsampling) is necessary for small/distant tags.
- `refine_edges=1` (subpixel refinement) always on.

**Results vs ArUco:**

| Frame | ArUco decoded | AT3 decoded |
|---|---|---|
| match frame 258 | 1 | 6 |
| match frame 1806 | 0 | 6 |
| match3 frame 286 | 0 | 5 |
| match4 frame 4503 | 4 | 7 |
| match5 frame 285 | 3 | 6 |

AT3 never does worse than ArUco and typically decodes 2–4× more tags.

**Limitation:** the Python wrapper (`pupil-apriltags`) only returns successfully
decoded detections. It does not expose intermediate quad candidates (the
equivalent of ArUco's `rejectedImgPoints`). The underlying C library has a
`debug=1` mode that writes quad images to `/tmp/` but this is platform-specific
and not usable on Windows.

---

## Current pipeline

`pipeline/01_detect_tags.py` — AT3 with the tuned parameters above.

**Output schema** (`data/detections/<stem>_tags.json`):

```json
{
  "video": "match.mp4",
  "params": {
    "detector": "apriltag3",
    "quad_decimate": 1.0,
    "quad_sigma": 0.0,
    "decode_sharpening": 1.25,
    "refine_edges": 1
  },
  "views": {
    "main": {
      "box": [0, 0, 1920, 712],
      "decoded_tags": {
        "5": {
          "n_frames_detected": 120,
          "mean_size_px": 36.1,
          "mean_center_px": [1409.2, 403.1],
          "mean_corners": [[x,y], [x,y], [x,y], [x,y]],
          "mean_decision_margin": 245.3,
          "observations": [...]
        }
      }
    }
  }
}
```

Note: `candidate_tags` (the ArUco rejected-quad list) is gone. AT3 doesn't
expose it.

---

## 5. Ensemble detection ✓ adopted (extends #4)

A single AT3 config is a compromise: aggressive `decode_sharpening` recovers
compressed tags but amplifies real motion blur; a tag too small for the bit
decoder needs more pixels, not different preprocessing. `pipeline/01_detect_tags.py`
now runs each sampled frame through multiple independent (image variant ×
detector config) combos and merges results per tag_id, keeping the highest
`decision_margin` on overlap — only ever *adds* detections a single pass would
have missed, never removes one.

- **Variants**: native crop / 2× bicubic upscale (more samples per bit cell
  for small tags) / CLAHE contrast enhancement (glare, shadow).
- **Configs**: `sharp` (the tuned defaults above) / `soft` (`quad_sigma=0.6,
  decode_sharpening=0.5`, for real motion blur — sharpening a blurred tag
  amplifies the blur, not the signal).
- **Combos run**: sharp×native, sharp×upscale2x, sharp×clahe, soft×native.

Measured on `match.mp4`'s `main` view, 8-frame smoke test: 65 observations vs
57 single-pass (+14%), zero regressions, one marginal tag went from decoding
in 1/8 frames to 5/8. `--no-ensemble` reverts to single-pass (~4× faster) for
quick iteration.

---

## Remaining problem

Even with ensemble AT3, decoded tags are sometimes too few or too clustered in
one part of the field to constrain a 6-DOF pose + intrinsics fit well — see
`docs/pose_calibration_research.md` for how the pose-solving side of the
pipeline handles (and doesn't fully solve) that.

**Removed, superseded by ensemble detection + `pipeline/03_solve_pose.py`'s
own corner/centroid handling** (see git history if ever needed — none of this
survived to justify keeping around):
- `viz/01_visualize_tags.py`, `viz/05_label_candidates.py` — visualized and
  hand-labeled ArUco's `rejectedImgPoints` candidate quads for a
  confidence-score ground-truth set. Dead the moment AT3 replaced ArUco (AT3's
  Python wrapper doesn't expose rejected candidates at all, see above) —
  ensemble detection covers what candidate-matching was trying to route around
  a different way (recovering tags a single detector pass would miss).
- `viz/04_confirm_tags.py` — interactive UI to manually confirm/ID a rejected
  quad or geometrically resolve it against a seed pose. Same dependency on
  ArUco's rejected candidates; also dead.
- `pipeline/03_calibrate.py`'s pose-guided correspondence recovery (project all
  32 tags from a seed pose, match nearest decoded tag geometrically) — folded
  into a different, more reliable idea: `pipeline/03_solve_pose.py`'s per-tag
  corner-vs-centroid hybrid gets more *out of the tags actually decoded*
  instead of trying to recover more tag IDs geometrically.
