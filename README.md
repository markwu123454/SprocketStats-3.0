# SprocketStats 3.0 — FRC Automated Video Scouting

Extract per-robot quantitative data from FRC match video, automatically.

> This README is the design document. It describes intended architecture and the
> reasoning behind it — not implementation status. Components are described by
> their role and contract, which stays true whether or not they are finished.

---

## 1. Two programs, one engine

**Program A — Broadcast ingest.**
Input: any official FRC broadcast video for a supported year. Output: as much
per-robot data as possible, fully automatically.
Hard constraint: **no per-match manual calibration or data entry.** Per-*year*
game knowledge and configuration authored by a human is expected and fine.

**Program B — Manual multi-camera ingest.**
Input: one or more hand-filmed videos of a single match, multiple angles,
handheld or tripod, unknown cameras. Per-match manual calibration and manual
data entry are acceptable.

Both are **offline batch**. No realtime requirement. This is a significant
freedom: it permits heavy models, bidirectional smoothing, and global
optimization over the whole match.

They converge on the same middle — *pixels → robots localized on the field plane
in match-clock time → events → metrics*. The difference is only how camera pose
and time base are obtained.

```
[Ingest]  →  [Registration]  →  [Detect / Track]  →  [Fuse]  →  [Game pack]  →  [Analytics]  →  [Export]
 A: broadcast   A: embedded views   (shared core)     A: single/multi   year plugin     shared       shared
 B: multi-cam   B: static + tags                      B: multi-view
```

---

## 2. Season archetypes

Games have alternated since 2022: **shooter** (2022, 2024, 2026) and
**pick-and-place** (2023, 2025). If the pattern holds, 2027 is a place game.

The only structural difference between the two is **the distance between where
the robot is when it scores and where the point is registered**:

- **Place**: distance ≈ 0. The robot's position *is* the evidence.
- **Shoot**: distance 2–8m. Robot position says only that it was somewhere in a
  large legal shooting region.

Everything else follows from that. Build one engine with two **evidence
backends**, selected per action type — not per game. 2024 had speaker (shoot)
and amp (place); 2025 had coral (place) and algae-into-net (shoot).

The registration accuracy bar is set by place games (~20cm, to disambiguate
adjacent grid nodes) and the transient-detection bar is set by shooter games.
Building registration to the place standard first means a shooter year is easy
afterward.

---

## 3. Components

| Path | What it is |
|---|---|
| `homography/` | Camera pose and intrinsics from AprilTags. Splits the broadcast frame into its embedded camera views, detects tags, searches focal length, solves PnP. Produces the image↔field transform everything downstream depends on. |
| `HRNet-W32/` | HRNet-W32 backbone + 2-channel Gaussian heatmap head (CenterNet-style). One keypoint per robot on the carpet under the bumper center, channel-separated by alliance. Label Studio + Cloudflare R2 active-learning loop. |
| `YOLO/` | YOLO detectors — robot bounding boxes, and AprilTags. The box is the persistent tracking anchor (see §5). |
| `mtmct/` | Multi-target multi-camera tracking. Consumes boxes and keypoints, emits persistent per-robot tracks in field coordinates. |
| `action/` | Per-robot, per-frame action classification (RNN-family) over finished track sequences. No vision dependency — consumes `(x_frac, y_frac)` only. |
| `better_sampling/` | TBA metadata and match video acquisition; frame sampling for the labeling corpus. |
| `match_prediction/` | **Separate pipeline.** Ensemble match-outcome prediction fusing Program A output, Program B output, Statbotics, TBA, and human scouting data. Downstream consumer, not part of the vision stack. |
| `scratch/` | Experiments. |
| `runs/` | Ultralytics auto-generated training output. Not a component. |

---

## 4. Registration

The foundation. Everything downstream is expressed in field coordinates, so
registration quality is the ceiling on every other number the system produces.

**Broadcasts embed 2–3 camera feeds in one frame.** `00_split_views` recovers
them by per-pixel temporal range: broadcast CG borders and overlays are rendered
at fixed pixel positions and are bit-identical across frames, while live camera
content always varies from motion, sensor noise, and compression. Threshold the
range image, morphologically close, take bounding rects of the dynamic regions.

This matters more than it sounds: **Program A gets multi-view coverage for
free**, which is otherwise Program B's main structural advantage.

**The main camera does not move during the match** in recent seasons. That makes
a single aggregated static pose per view per match valid, and aggregating tag
centroids across many frames is what makes 15–50px tags usable at all — averaging
suppresses the ±2–3px per-frame corner jitter that H.264 block artifacts
introduce. Secondary views degrade; acceptable, since the main view carries the
field.

**Pose solving is a hybrid per tag.** Corner points for tags seen close to
head-on (where foreshortening isn't amplifying pixel noise into large angular
error), centroid otherwise. Views with too few points relax to corners for every
tag. A single-tag view is planar and carries a real 2-fold ambiguity — treat it
as low confidence.

**Intrinsics are searched as a ladder, not fitted jointly.** With 2–10 tags
clustered in one region of frame, jointly solving `fx, fy, cx, cy, k1, k2, p1,
p2` plus 6 pose DOF is underconstrained — and underconstrained optimization
doesn't fail loudly, it converges on a confident-looking wrong answer. So: start
with `fx = fy = f`, principal point fixed at crop center, zero distortion, and
search the single free parameter against real reprojection RMS. Add the
principal point only once a view has enough points to support it.

`docs/pose_calibration_research.md` and `docs/tag_detection_research.md` record
what was tried and why it failed. Read them before proposing a "better" fit.

**Self-validating**: AprilTag reprojection error needs no human labels. Track it
per frame and propagate it into output confidence.

**Program B calibration**: AprilTag PnP where possible; fallback is the user
clicking 4+ known field points. Handheld footage needs per-frame propagation
(KLT + RANSAC, masking out detection boxes so moving robots don't contaminate
the feature set) with periodic tag re-anchoring to kill drift.

---

## 5. Detection: two models, two jobs

**The box tracks. The keypoint locates.**

### Keypoint (HRNet-W32)

Box-free heatmap detector. Channel 0 = blue, channel 1 = red; each is a heatmap
with a Gaussian peak at every keypoint of that alliance. Inference is per-channel
local-max peak-finding — no anchors, no association step. HRNet keeps
high-resolution features throughout, which is why it localizes small points well
across the enormous scale range between a near-ground robot and a far one.

YOLO-pose was rejected for this: it requires each keypoint to be associated with
a parent robot box, and these points routinely violate that — overlap ambiguity,
and climbing robots whose floor point sits outside their own box.

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
- **When no keypoint is available**, the track survives on the box alone;
  position is predicted from track history (and optionally the box's bottom edge
  as a weak fallback) with degraded confidence, rather than the track breaking.

Keypoint→box assignment is a small local matching problem: containment, plus
alliance-channel agreement. Where boxes overlap heavily, prefer the box whose
predicted ground point — from its own history through the homography — is
nearest to the peak.

The payoff: **keypoint dropouts become interpolated positions with lowered
confidence instead of identity breaks.**

### Alliance color

The keypoint model bakes alliance into its channels, so read *both* channel
responses at each peak and keep the ratio as a **soft** color score rather than
hard-assigning by channel. This matters because colour is properly a **global
constraint, not a per-detection classification**: exactly three robots are red
and three are blue, so it is a balanced assignment over tracks. A track scoring
60% blue gets assigned red when three others score 95% blue — a very dark navy
bumper that reads as black on camera does not need to *look* blue, it needs to
look *more blue than the three that clearly aren't*. Hard channel assignment
throws that away.

---

## 6. Tracking and identity (`mtmct/`)

**Contract**: boxes + keypoints + homography in; persistent per-robot tracks in
field coordinates out, keyed by team number, consumed by `action/`.

**Track in field space, not image space.** Run the Kalman filter on field
coordinates. Robots have bounded velocity (~4–5 m/s) and acceleration, so the
motion model is genuinely predictive. A 50-pixel gap means something completely
different at the near wall than the far wall; in metres it means the same thing
everywhere.

**Motion is not a detector.** The detectors find robots independently each frame;
the motion model only answers *which existing track this detection belongs to*.
Game pieces are a different class and never enter the robot association problem.

Association cost = Mahalanobis distance in field coordinates + box IoU + soft
colour agreement. Hungarian assignment per frame.

**Then, offline, stitch tracklets by min-cost flow.** There is a constraint no
general MOT problem has: **exactly six robots, none of which leave**. Every
tracklet must be assigned to one of six slots. This is constrained assignment,
not open-set re-ID.

**Handling ambiguity.** Two same-alliance robots in sustained contact will
produce identity swaps; nothing fully prevents this.

- Colour halves the problem — red-vs-blue contact is trivially resolvable, and
  most contact is cross-alliance (defence).
- **Don't guess — break the tracklet.** Offline, there is no obligation to commit
  at time *t*. Emit two fragments and let the global stitch resolve them using
  evidence from t+5s, when the robots separate and one produces a confident
  bumper read.
- Swaps degrade gracefully: heatmaps and alliance aggregates survive; only
  per-robot event attribution corrupts, and only for the swapped interval.

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

**Gate and accumulate.** Run OCR only where the box is large enough and Laplacian
variance indicates acceptable motion blur — maybe 3–5% of frames, still hundreds
of usable reads per robot per match. Sum log-probabilities across the whole
stitched track.

**Appearance gallery bootstrap.** Once a team is identified at an event, cache an
appearance embedding. FRC robots are visually distinctive. Match 1 gets OCR or a
manual click; matches 40–80 are recognized by embedding against a 6-candidate
gallery — far easier than OCR.

### Multi-view fusion

Applies to Program B, and to Program A's embedded views.

- **Time base**: all events are timestamped in **match-clock seconds**, not video
  time. Program A OCRs the match timer overlay. Program B syncs cameras by
  GCC-PHAT cross-correlation on the start horn and crowd noise (sub-frame
  accurate, zero user input), then anchors to match clock via a camera that sees
  the field timer, or one manual click.
- **Cross-camera association is geometric, not appearance-based.** After sync and
  calibration, two tracks are the same robot if they occupy the same field
  position at the same time.
- **Fusion**: per-robot factor graph or RTS smoother over all views, weighting
  each observation by view geometry — distance, image-edge proximity, detection
  confidence, registration residual.

---

## 7. Game pack

A per-year plugin. Everything else in the repo is year-agnostic. "Supported year"
means *someone authored a game pack and trained a piece detector*, not *we
rewrote the pipeline*.

```yaml
game_pack_2025:
  field:
    cad: reefscape_field.step
    apriltag_layout: [...]
    landmarks: [...]
  pieces: [coral, algae]
  actions:
    - name: coral_place
      archetype: PLACE
      sites: [ {id, field_xyz, level}, ... ]
      state_readable: true
    - name: algae_net
      archetype: SHOOT
      goal_volume: {...}
      max_range_m: 5
    - name: algae_processor
      archetype: PLACE
  endgame:
    archetype: CLIMB
    structures: [ {id, field_xyz} ]
  tba_breakdown_map: {...}
```

### PLACE backend — read state, not events

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

6. **Pin the terminal state.** TBA's score breakdown gives the exact
   end-of-match state. Run Viterbi over the site-state sequence with the final
   state fixed and per-period totals constrained. Mid-match errors are corrected
   by the endpoint. You are decoding the most likely event sequence consistent
   with a known outcome, not detecting from scratch.

Level/height disambiguation requires full camera pose, not just a ground
homography.

**High-value place metrics**: level *capability* (reliably, or once?), and
**alignment time** — arrival at node → piece released, which separates good
mechanisms from bad ones and is invisible to a human counting pieces.

### SHOOT backend — detect the projectile, fit the arc

Detect pieces in flight as their own class. Appearance differs substantially from
at-rest or held pieces: motion-blurred, elongated, isolated against background.
Then reconstruct the arc and use it for both attribution and made/miss.

Detection is hard — a ~35cm piece at 15m in 1080p is ~25px, motion-blurred, often
against a crowd. Mitigations: tiled high-resolution inference over the field-and-
above region only, triggered when a launch is suspected; synthetic motion-blur
augmentation (the streak is a feature, not just noise); and the ballistic fit
itself as the false-positive filter.

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

**What breaks elsewhere in shooter years**: dwell/transit segmentation (§8)
degrades, since ground intake happens while driving and shooting happens while
moving — use launch events as the cycle delimiter instead of zone occupancy. And
possession is *invisible* (the piece is inside the robot), so it must be inferred
between pickup and launch.

### ENDGAME backend — the constraint violation *is* the detector

2022 traversal, 2023 charge station, 2024 chain, 2025 cage — all break the
ground-plane assumption. When a track's plane constraint becomes geometrically
inconsistent, near a known climb structure, in the final 30 seconds: that's a
climb. Cross-check against TBA's endgame breakdown.

Outputs: climb start time (abandoning cycles at 0:30 or 0:10?), duration,
success, and for shared structures, which partners they can climb with.

---

## 8. Analytics

Operates on `(t, x, y)` per robot. **No CV, no game knowledge required** — which
is why this layer ships value before any game pack exists, and doesn't break at
kickoff.

1. **Smooth and differentiate.** Savitzky-Golay, or an RTS smoother over the
   Kalman track (bidirectional — offline). Naive finite differences on noisy
   positions produce useless velocity.

2. **Segment into dwell / transit / idle.** A 3-state HMM with Gaussian emissions
   on speed. This is the fundamental parse; everything below builds on it.

3. **Discover functional zones with no game pack.** Cluster dwell centroids
   across every robot and match at an event (DBSCAN or GMM). The clusters *are*
   the pickup stations and scoring locations — the game's spatial structure
   emerges from where robots stop.

4. **Cycles as round trips.** With dwells zone-labelled, a cycle is an A→B→A
   pattern. Report the **distribution** — median and IQR, never the mean — and
   fit a trend across match time to catch fatigue or strategy shifts.

5. **Decompose the cycle.** The most actionable output in the system. Two robots
   with identical 5-cycle matches can be entirely different: `4s transit / 6s
   scoring / 2s pickup` versus `9s / 2s / 1s`. The first has a slow mechanism;
   the second a slow drivetrain or bad route. Add **path efficiency**
   (straight-line ÷ actual path length) as a driver-skill proxy.

6. **Drivetrain type and driver quality from position alone.** Tank drives must
   decelerate to change direction; holonomic drives don't. Plot speed against
   |dθ/dt| of the *velocity vector*. Strong negative correlation ⇒ tank; near
   zero ⇒ swerve. The slope quantifies speed lost per radian of direction change.
   Requires no orientation estimate at all.

7. **Autonomous as a trajectory signature.** Cluster the first 15 seconds across
   a team's matches with DTW or discrete Fréchet distance. Yields: how many
   distinct autos they run, consistency of each (intra-cluster variance), and
   failure rate (trajectory truncating early relative to its prototype).

8. **Defence, measured causally.** Pairwise distance series for every
   cross-alliance pair. A defender shows high time-in-opponent-half, low dwell in
   its own scoring zones, and persistent proximity to one opponent. Measure
   impact by regressing the victim's cycle time on defender proximity across the
   event, with robot and partner random effects. Output in
   **seconds-added-per-cycle** — interpretable and unobtainable by hand.
   Collisions come from acceleration impulses coinciding with proximity.

9. **Reliability via change-point detection.** PELT or Bayesian online
   change-point on the speed series catches "died at t=94s" and "speed
   permanently dropped 40% after a collision."

10. **Heatmaps and alliance compatibility.** KDE over each robot's positions.
    Compare prospective partners by heatmap **overlap** — high overlap means they
    contest the same lanes and pickup stations, so their combined output is less
    than the sum of their parts. Score data cannot show this.

11. **Skip OPR.** OPR exists to deconvolve individual contribution from alliance
    score by least squares. Individual contribution is measured directly here.
    Use regression only for interaction terms.

`action/` learns a supervised version of steps 2–3 (per-frame class labels from
an RNN over ego + relational features). The unsupervised HMM and zone clustering
are its natural pre-labeler — the same correction-loop economics already used for
the keypoint model.

---

## 9. Program A stage order

1. **Ingest** — video plus match key (from TBA, or OCR of the match-ID overlay).
   Fetch teams, score, and full score breakdown.
2. **View splitting** — recover the embedded camera feeds.
3. **Overlay parsing** — match timer, live score, match ID. Per-event templates,
   since graphics change.
4. **Replay rejection** — the timer does this for free: absent or non-monotonic
   ⇒ discard the segment. Double-counting replayed scoring actions is the most
   likely silent failure mode in the system, and this check eliminates it.
5. **Registration** per view.
6. **Detection** — boxes and keypoints.
7. **Tracking and team assignment.**
8. **Game pack event extraction.**
9. **TBA-constrained reconciliation** (below).
10. **Analytics.**
11. **Export + review UI** — corrections feed back as training data.

### TBA-constrained reconciliation

The trick that makes the whole thing tractable. TBA gives exact counts of each
scoring action, per alliance, per period. So event detection becomes: generate
candidate events with confidences, then **select the subset satisfying the known
totals** — an integer program, or Viterbi with a pinned terminal state.

This converts a hard detection problem into a much easier attribution problem,
and every reconciled match becomes free labelled training data.

---

## 10. Output

Per match:

- **Metadata** — event, match key, video sources, pipeline and game pack version.
- **Tracks** — `(robot_id, team, t, x, y, confidence)` at 20–30 Hz, Parquet.
- **Events** — typed, timestamped in match-clock seconds, attributed, with
  confidence.
- **Derived metrics** — per robot, per match.

**Confidence is a first-class output, not an afterthought.** Every value carries
provenance: which view, frames observed vs occluded, registration reprojection
residual, identity assignment margin, whether the event survived TBA
reconciliation. *Scouting data nobody trusts is worse than no scouting data.*

---

## 11. Validation

- **Registration**: AprilTag reprojection error. Self-validating, no labels.
- **Events**: precision/recall against TBA score-breakdown totals across many
  matches. Free labels at scale.
- **Tracking and identity**: a handful of hand-labelled matches.
- **Credibility**: regress automated stats against human scouting data to surface
  systematic bias. Disagreements are informative in both directions.
- **B as ground truth for A**: film matches with the recommended protocol,
  process both ways, quantify A's error against B's multi-view fused output.

**Split by season or by match, never by frame.** Frames adjacent in time are
near-duplicates; a random frame split leaks.

---

## 12. Training data

- **Registration**: entirely synthetic from official field CAD in Blender with
  domain randomization. Perfect labels, unlimited quantity.
- **Robot detection / keypoint**: label a few thousand frames across events and
  lighting conditions. Cut cost by labelling one frame per shot and propagating
  with a tracker, then human-verifying.
- **Piece-in-flight**: synthetic motion blur over at-rest crops, plus real
  examples mined from launch events.
- **Self-training**: keep high-confidence TBA-reconciled output, retrain.
- **Active learning**: every review-UI correction becomes a training example.
- **Corpus**: TBA maps match keys to match videos, so a labelled-by-construction
  corpus assembles automatically. Observe source terms of service.

---

## 13. Filming protocol (Program B)

Half the CV difficulty is removable by specifying how to film. Ship this as part
of the product:

- Two static tripods at opposite field corners, wide framing, no panning.
- 4K30 or better.
- At least one AprilTag visible at all times.
- Recording started before the horn, for audio sync.
- Do not zoom.

---

## 14. Non-goals

- Per-mechanism articulation or pose estimation.
- Tracking game pieces on the ground as a primary signal — small, occluded,
  constantly moving.
- Realtime processing.
- Any per-match manual input in Program A.
- Instance segmentation of robots. Extracting `(x, y)` from a mask needs contour
  fitting plus constrained 3D box optimization; it's fragile and the gain over a
  trained keypoint is small.
