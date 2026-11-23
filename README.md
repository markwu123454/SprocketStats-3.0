# SprocketStats 3.0 — FRC Automated Video Scouting

Extract per-robot quantitative data from FRC match video, automatically.

> This README holds only what the owner decided or approved, as marked in the
> 2026-11-19 review (`PLAN_REVIEW.md`). Ideas nobody has approved are in
> `IDEAS.md`, labelled unconfirmed. Design text here is not a build order: ask
> before building toward anything that isn't built yet.

---

## 1. Two programs

**Program A — Broadcast ingest.**
Input: any official FRC broadcast video for a supported year. Output: as much
per-robot data as possible, fully automatically.

**Program B — Manual multi-camera ingest.**
Input: one or more hand-filmed videos of a single match, multiple angles,
handheld or tripod, unknown cameras. Per-match manual calibration and manual
data entry are acceptable.

---

## 2. Season archetypes

Games have alternated since 2022: **shooter** (2022, 2024, 2026) and
**pick-and-place** (2023, 2025). If the pattern holds, 2027 is a place game.

For now the pipeline assumes a place game. If the whole pipeline gets built in
one season, changing it for a shooting game the next season isn't much work, so
there is no code that switches between place and shoot: it would probably be
rewritten next year anyway.

---

## 3. Components

The six pipeline modules so far. More will be added as new models are built to
do more things.

| Path | What it is |
|---|---|
| `homography/` | Camera pose and intrinsics from AprilTags. Detects tags in each camera view, fits focal length and pose jointly, and refuses views it can't solve. Produces the image↔field transform. |
| `broadcast/` | Everything specific to how a broadcast is produced: match phases from the audio, the camera feeds embedded in the frame, and the scoreboard overlay read into a timeline of scores, clock and counters. Timing and alliance-level scoring ground truth. |
| `hrnet/` | HRNet-W32 backbone + 2-channel Gaussian heatmap head (CenterNet-style). One keypoint per robot on the carpet under the bumper center, channel-separated by alliance. Label Studio + Cloudflare R2 active-learning loop. |
| `yolo/` | YOLO detectors — robot bounding boxes, and AprilTags. The box is the persistent tracking anchor (see §5). Trained; there is no `yolo/run/` yet, so nothing in the pipeline uses them. |
| `mtmct/` | Robot tracking: all of it, including the end-to-end pass that ends with named tracks. Today it tracks HRNet dots in image pixels on the main view; the YOLO box isn't wired in. |
| `ocr/` | Names each finished track with a team number, or leaves it unknown, from its bumper text scored against the six known teams (see §6, *Team number assignment*). |

Supporting code, outside that structure:

| Path | What it is |
|---|---|
| `better_sampling/` | TBA metadata and match video acquisition; frame sampling for the labeling corpus. |
| `match_prediction/` | **Separate pipeline.** Ensemble match-outcome prediction fusing Program A output, Program B output, Statbotics, TBA, and human scouting data. Downstream consumer, not part of the vision stack. |
| `scratch/` | Experiments. |
| `runs/` | Ultralytics auto-generated training output. Not a component. |

### Module layout

Each pipeline module is a Python package split the same way:

- `train/` — building datasets and fitting whatever `run/` loads: model
  checkpoints, digit and icon templates, per-year field layouts. Run
  occasionally, not per match.
- `run/` — the per-match steps, plus the code other modules are allowed to use.
- `debug/` — visualizations, probes, smoke tests and unit tests.

Imports only go one way: `train/` and `debug/` may import `run/`, `run/` never
imports either, and modules reach each other only through `<module>.run`. That
keeps the per-match path free of training and plotting dependencies, and it is
what lets a module's training code or viewers change without risk to the
pipeline. Files are named for what they do rather than numbered; where steps
have an order, `run/__init__.py` lists it.

### Environment

The environment is managed with uv, and `pyproject.toml` is the only place
dependencies are declared: one group per module, pinned to the versions the
pipeline was last checked with, plus the CUDA torch index and the OpenCV
override. `uv lock` resolves it into `uv.lock`; `uv sync` makes `.venv` match the
lock exactly and installs the six modules as editable packages, so they import
from any directory. Because the sync is exact, a new dependency goes into
`pyproject.toml` (`uv add --group <module> <pkg>`), not an ad-hoc
`uv pip install` that the next sync would remove. The comments in
`pyproject.toml` describe a one-off step the first sync needs.

Run a step by path (`python homography/run/detect_tags.py ...`), as a module
(`python -m homography.run.detect_tags ...`), or through `uv run`.

### Data

Code goes through git as usual. Each module's `data/` folder (checkpoints,
templates, layouts, per-match outputs) goes to the `SprocketStats/modelfiles`
Hugging Face repo through `hf_sync.py`, which never touches git:

```
python hf_sync.py status              # what push/pull would do; changes nothing
python hf_sync.py push                # upload files HF doesn't have yet
python hf_sync.py pull                # download files you don't have yet
```

Files already on HF are immutable by default: a push never replaces or deletes
one, and a pull never overwrites a local file that differs. Replacing takes
`push --override` / `pull --override`, deleting takes
`hf_sync.py delete <path> --override`. A checkpoint or template that others have
already pulled and built on shouldn't change under them silently. Overridden
versions stay in the HF repo's commit history. The rule lives in the script, not
on the server, so it only holds for changes made through it. The local mirror of
the R2 training images and the YouTube cache are never synced; both can be
fetched again from their source. Log in once per machine with
`.venv\Scripts\hf auth login`.

---

## 4. Registration

**Broadcasts embed 2–3 camera feeds in one frame.** `broadcast/run/split_views` recovers
them by per-pixel temporal range: broadcast CG borders and overlays are rendered
at fixed pixel positions and are bit-identical across frames, while live camera
content always varies from motion, sensor noise, and compression. Threshold the
range image, morphologically close, take bounding rects of the dynamic regions.

**Only the main camera is used.** The secondary views are too oblique: the
keypoint model struggles to find the actual spot on them, and the more top-down
the view, the easier it is to place the dot in the middle of the robot.

**The main camera does not move during the match** in recent seasons. That makes
a single aggregated static pose per view per match valid, and aggregating tag
centroids across many frames is what makes 15–50px tags usable at all — averaging
suppresses the ±2–3px per-frame corner jitter that H.264 block artifacts
introduce. Where the broadcast does cut, `solve_pose` splits at the cut, fits
each shot as its own camera, and takes the dominant shot as the view's pose.

**Pose solving is a hybrid per tag.** Corner points for tags seen close to
head-on (where foreshortening isn't amplifying pixel noise into large angular
error), centroid otherwise. Views with too few points relax to corners for every
tag. A single-tag view is planar and carries a real 2-fold ambiguity — treat it
as low confidence.

**Intrinsics and pose are fitted jointly, under absolute bounds**
(`fit_intrinsics_joint` in `homography/run/solve_pose.py`). The earlier
one-parameter ladder search stalled because focal length, `cy` and pitch all
move the projection vertically, so separate 1-D searches crawled along the
valley instead of crossing it. Extra parameters (principal point, `k1`) are
admitted on their own error bars from the covariance, not on how many points a
view has, and a fit is judged by leave-one-tag-out held-out error. A view that
can't be solved from its own correspondences is refused, with its reasons
recorded, rather than given a confident wrong pose.

`homography/docs/pose_calibration_research.md` and `homography/docs/tag_detection_research.md` record
what was tried and why it failed. Read them before proposing a "better" fit.

---

## 5. Detection: two models, two jobs

**The box tracks. The keypoint locates.**

### Keypoint (HRNet-W32)

Box-free heatmap detector. Channel 0 = blue, channel 1 = red; each is a heatmap
with a Gaussian peak at every keypoint of that alliance. Inference is per-channel
local-max peak-finding — no anchors, no association step. HRNet keeps
high-resolution features throughout, which is why it localizes small points well
across the enormous scale range between a near-ground robot and a far one.

The keypoint is the **only** thing that produces an accurate ground-plane
coordinate. A box's bottom edge is the *nearest* contact point rather than the
center, its offset grows with camera obliquity, and occlusion clips it silently
with unbounded error.

**Its weakness is persistence.** The keypoint sits low — at bumper level. When
the bumpers or the robot's lower body are occluded by another robot, by field
structure, or by the frame edge, the peak weakens or disappears entirely. So
keypoint presence is intermittent, which makes it a poor basis for temporal
identity: a dropout looks exactly like a robot leaving.

### Box (YOLO)

A box covers the whole robot, so it survives lower-body occlusion — the top half
is usually still visible when the bumpers aren't. Boxes are also what standard
tracking machinery is built around (IoU + motion association), so they chain
across frames far more reliably.

### Division of labor

- **A track is a chain of boxes.** The box owns identity and continuity.
- **Each frame, a keypoint peak falling inside or near a box supplies that
  track's field coordinate.** The keypoint owns position.
- **When no keypoint is available**, the track survives on the box alone rather
  than breaking.

---

## 6. Tracking and identity (`mtmct/`)

**Contract**: boxes + keypoints in; persistent per-robot tracks out, keyed by
team number.

**Don't guess — break the tracklet.** Where a track may have changed robots,
`mtmct/run/split_tracks.py` cuts it so each piece answers for one robot: when
confident bumper reads switch team partway, or when geometry alone says it could
have swapped onto another robot of its alliance. A needless cut costs little,
since naming can still give both pieces the same team; a missed swap names the
whole track after the wrong robot.

### Team number assignment

Team numbers are never *discovered* from pixels. TBA gives the six teams and
alliance stations for a match key. The task is to **assign six known labels to
six tracks**.

**Score candidates, don't decode.** Once colour is assigned, a track has only
three possible team numbers. Do not decode the crop to a string and then compare
— that discards the probability information. Take the recognizer's per-character
probability lattice and run each of the three candidates through the CTC scoring
function:

```
score_i = log P(candidate_i | crop)    for i in the 3 numbers on this alliance
```

Argmax wins. A crop far too blurry to read in the open 10⁴ space still gives a
decisive answer among three hypotheses.

**Appearance gallery bootstrap.** Once a team is identified at an event, cache an
appearance embedding. FRC robots are visually distinctive. Match 1 gets OCR or a
manual click; matches 40–80 are recognized by embedding against a 6-candidate
gallery — far easier than OCR.

### Time base

All events are timestamped in **match-clock seconds**, not video time. Program A
reads the match timer overlay.

---

## 7. Scoring

The pipeline assumes a place game for now (§2).

### PLACE — read state, not events

**Defining property: scored pieces persist and stay visible.** This converts
event detection into state estimation, which is far more robust.

Don't try to catch the instant of placement — it's sub-second, heavily occluded,
easily missed.

1. Define each site as a small ROI **in field coordinates**.
2. Every frame, warp those ROIs into the image and classify each: `empty` /
   `piece_type` / `occluded`.
3. Temporally median-filter. Hundreds of observations of a state that changes a
   handful of times.
4. **Locate the change window**: `[last confident empty, first confident filled]`.
   Occlusion widens it.
5. **Credit** whichever robot was dwelling adjacent to the site during the window.

> **Occlusion is evidence.** The robot blocking your view of a site is almost
> always the robot placing into it. Treat the occluder as the prime suspect, not
> as a problem.

### SHOOT — detect the projectile, fit the arc

Detect pieces in flight as their own class. Appearance differs substantially from
at-rest or held pieces: motion-blurred, elongated, isolated against background.
Then reconstruct the arc and use it for both attribution and made/miss.

**Monocular 3D reconstruction is well-conditioned here.** Unknowns are launch
point `(x₀, y₀, z₀)` and launch velocity `(vₓ, v_y, v_z)` — 6 DOF. Gravity is
known. Each 2D observation gives 2 constraints, so 3+ observations
over-determine the system. Add strong priors: `z₀` ≈ shooter height (0.5–1.5m),
`(x₀, y₀)` ≈ the field position of some tracked robot at t₀. Solve by
Levenberg-Marquardt with RANSAC over detections; RANSAC-cluster detections into
separate arcs first when shots overlap.

The fitted parabola gives:

- **Back-extrapolation to t₀ → attribution.** Which robot was at that field
  position at that instant? The cleanest attribution signal in a shooter game.
- **Forward-extrapolation → made/miss.** Does the arc pass through the goal
  aperture volume?
- **Free mechanism characterization**: shot distance, launch angle, launch speed,
  apex height, goal entry angle.

**Supporting signals:**

- **Goal-arrival monitoring.** The goal is a fixed field region; warp a small ROI
  in each frame and watch for a piece entering. Tiny ROI means you can afford a
  much heavier detector there.
- **Launch-streak detection.** If full-flight detection fails, search a small
  window anchored to the robot's box for a fast object emerging ballistically.
  Frame-difference + Hough streak detection, or a small CNN on 3-frame stacks.
  Yields launch time and departure vector even when the flight is lost.
- **Score-tick matching.** The overlay increments per made shot, giving exact
  timestamps and counts for the alliance. Bipartite-match ticks against candidate
  launches. Calibrate animation lag per event by cross-correlation.
- **Audio.** Flywheel spin-up is a narrowband tonal ramp; the shot is percussive.
  Broadcast audio is buried under commentary; Program B's near-field audio isn't.

Launches give **attempts**; score ticks and goal arrivals give **makes**.

> `accuracy = makes / attempts` **per robot is the single most valuable
> shooter-game statistic, and human scouts are notoriously bad at it.**

With launch position you get **accuracy as a function of shooting distance** —
"85% close, 40% from midfield" — which no manual scouting system produces. Robot
speed at the launch instant gives shoot-on-the-move capability.

**On heading**: chassis yaw is low-value and hard to measure — swerve robots have
no canonical front and rotate without driver intent. But *aim direction* comes
free from the projectile's departure vector. Measure the informative thing, not
the hard thing.

---

## 8. Analytics

Operates on `(t, x, y)` per robot. **No CV, no game knowledge required**, so it
works before a new game is supported and doesn't break at kickoff.

1. **Cycles as round trips.** A cycle is an A→B→A pattern. Report the
   **distribution** — median and IQR, never the mean — and fit a trend across
   match time to catch fatigue or strategy shifts.

2. **Decompose the cycle.** The most actionable output in the system. Two robots
   with identical 5-cycle matches can be entirely different: `4s transit / 6s
   scoring / 2s pickup` versus `9s / 2s / 1s`. The first has a slow mechanism;
   the second a slow drivetrain or bad route. Add **path efficiency**
   (straight-line ÷ actual path length) as a driver-skill proxy.

3. **Autonomous as a trajectory signature.** Cluster the first 15 seconds across
   a team's matches with DTW or discrete Fréchet distance. Yields: how many
   distinct autos they run, consistency of each (intra-cluster variance), and
   failure rate (trajectory truncating early relative to its prototype).

4. **Defence, measured causally.** Pairwise distance series for every
   cross-alliance pair. A defender shows high time-in-opponent-half, low dwell in
   its own scoring zones, and persistent proximity to one opponent. Measure
   impact by regressing the victim's cycle time on defender proximity across the
   event, with robot and partner random effects. Output in
   **seconds-added-per-cycle** — interpretable and unobtainable by hand.
   Collisions come from acceleration impulses coinciding with proximity.

5. **Heatmaps and alliance compatibility.** KDE over each robot's positions.
   Compare prospective partners by heatmap **overlap** — high overlap means they
   contest the same lanes and pickup stations, so their combined output is less
   than the sum of their parts. Score data cannot show this.

**Per-frame classification of what each robot is doing**, by an RNN or a similar
algorithm. Not built: the untrained `action/` scaffold for it was deleted on
2026-11-19.

---

## 9. Splitting data

**Split by season or by match, never by frame.** Frames adjacent in time are
near-duplicates; a random frame split leaks.
