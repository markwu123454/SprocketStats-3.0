# Camera Pose & Intrinsics Calibration Research

Once `pipeline/01_detect_tags.py` has decoded AprilTags per view, the pipeline
needs each camera's pose (position + orientation) *and* intrinsics (focal
length, principal point, lens distortion) to project between pixels and field
coordinates. There's no controlled calibration target anywhere in this
problem — no checkerboard, no known lens spec, just whatever AprilTags a
broadcast camera happened to see, 2–11 of them per view, often clustered in
one part of the frame. This document records everything tried.

---

## The core difficulty

Camera calibration needs geometric *diversity* — a target seen from many
angles, or a moving camera. A single static broadcast camera looking at a
handful of tags clustered in one region and a few discrete heights is
close to the textbook example of an ill-conditioned calibration problem:
plenty of ways to make the numbers fit the tags you have while producing
nonsense everywhere else in the scene. Every early approach below hit this
same wall in a different guise before the fix (see "Adopted approach") turned
out to be about respecting *how much* the data could support, not about
finding a cleverer optimizer.

---

## Approaches tried

### 1. Manual drag-to-fit — `unused/manual_fit.py`

Render the known field (every tag + the boundary, from the official field
layout) as a wireframe over the real frame and let a human drag it into
place: position (x,y,z), heading (yaw/pitch/roll), one shared focal length
(as horizontal FOV), one distortion term (k1, deliberately range-limited).
CAD-viewport mouse navigation, keyboard-only parameter editing — deliberately
**not sliders**, because OpenCV's trackbar widget shows its own raw tick
position, not a converted physical value (it could show "k1: 0" while the
real k1 was -0.3). Produced a trustworthy seed pose, but doesn't scale: a
human in the loop per view per video.

**Superseded by**: `pipeline/03_solve_pose.py` solving automatically once
intrinsics are decent (see #6), plus a *new* manual tool this round,
`viz/09_calibrate_ui.py` — real browser `<input type="range">` sliders with
the actual value always displayed next to them, so the exact failure mode
that ruled out sliders here doesn't apply to a page you control the DOM of.

### 2. Blind correspondence-recovery fit — `unused/calibrate.py`

Project all 32 field tags through a seed pose to predict where each should
appear, search near each prediction for a matching quad even if ID decoding
failed, and jointly fit K + distortion + pose to whatever came back — no
ground truth to check the result against.

**Result**: physically implausible distortion (k1 = -0.98) on the one match
tested. Too few, too spatially-clustered tags to constrain 8 intrinsic + 6
pose parameters at once; an underconstrained fit doesn't fail loudly, it
converges on a confident-looking wrong answer.

### 3. Geometry-assisted recovery seeded from a manual pose — `pipeline/03_calibrate.py`

Same correspondence-recovery idea as #2, but seeded from #1's trusted manual
pose instead of a blind guess, on the theory that a good seed would keep the
geometric matching honest.

**Result**: `data/calibration/match_refined_intrinsics.json` — 20px RMS baked
into its own fit, before it's used for anything downstream. A better seed
didn't fix an underconstrained joint fit; it was still solving for too many
parameters at once.

### 4. Line-curvature distortion fit — `pipeline/02_find_curves.py` + `02_fit_distortion.py`

A different idea: skip tags entirely for distortion specifically. A real
straight line in the world stays straight after correct undistortion and
bows under a wrong one, so find long straight edges *anywhere* in the frame
(guardrails, catwalks, banner edges — no field-geometry knowledge needed,
via `cv2.createLineSegmentDetector`, LSD not Canny+findContours after the
latter fused real edges into unrelated clutter on this footage — see that
script's docstring for the specific failure) and search for the `(cx, cy,
k1)` that makes all of them straightest at once, weighted by length. Pose-free
and focal-length-blind by construction.

**Result**: `data/calibration/match_main_distortion.json` — the fitted
residual barely moved (0.8943px → 0.8936px), i.e. the cost surface was
"nearly flat": changing k1 meaningfully didn't meaningfully change the fit.
A richer, better-distributed geometric signal than tag corners still
couldn't pin distortion down at this resolution/compression level.

### 5. Naive FOV-based focal guess — `pipeline/02_estimate_intrinsics.py`

Assume a horizontal FOV (default 70°), zero distortion, principal point at
the exact crop center — no fitting from the video at all, just
`f = (w/2) / tan(fov/2)`.

**This was the production default** feeding `pipeline/03_solve_pose.py` until
this round of work, and its flaw is what actually caused the "doesn't line up
anywhere" symptom that kicked off this whole investigation:
`data/calibration/match_intrinsics.json` had the *identical* `focal_px:
1004.0` for `main` (87° FOV, 1920px wide) and `bot_left`/`bot_right` (50° FOV,
~940px wide) — physically impossible for independent crops, because
`--focal-px 1004` had been passed once and reused, never fit per-view.

### 6. 1-D ladder search ✓ **adopted** — now `pipeline/03_solve_pose.py`'s intrinsics search (originally a separate `pipeline/02_search_focal.py`, merged in since the two were never really separable — see that file's module docstring)

Start from the simplest model that could possibly be right — `fx=fy=f`,
`cx=W/2, cy=H/2`, `dist=0`, one free parameter — and search it with a real
objective: hand `pipeline/03_solve_pose.py`'s actual `solve_view()` a
candidate K and use its **all-correspondences RMS** (not the RANSAC
inlier-only number, which is gameable — a wrong-by-2.5× focal length was
confirmed faking a low residual by having RANSAC quietly drop half the
points as "outliers"). Coarse grid, then golden-section refine, plus a
flatness check (cost ±100px from the minimum) so a poorly-constrained "best"
f gets flagged instead of trusted.

Then — and this is the part that makes it work where 1–5 didn't — **only add
another parameter once there's enough data to support it**, gated by point
count, and only after checking the addition is real signal:

- **`+ cx, cy`** once a view clears `MIN_POINTS_FOR_PRINCIPAL_POINT` (20).
  Motivated physically, not just numerically: `main` is a Y-only crop of a
  larger broadcast frame (`pipeline/00_split_views.py`), so its true
  principal point has no reason to sit at the crop's own geometric center.
  Confirmed on `match.mp4`: `cy` moved +332px (356→688 on a 712px-tall crop,
  RMS 8.14→4.36px) while `cx` stayed at *exactly* the crop center (960.0,
  Δ0.0) — consistent with that crop being full-width/uncropped in X. A
  coincidence would not land exactly on 0.
- **`+ k1`** once a view clears `MIN_POINTS_FOR_DISTORTION` (30), and only
  after **leave-one-out cross-validation**, not a lower in-sample residual
  (which improves with *any* extra parameter, trivially, and is exactly the
  failure mode #2 and #3 fell into). Held one tag out, re-fit without it,
  checked whether allowing k1 improved the *held-out* tag's prediction.
  Confirmed on `match2.mp4` (2026 World Championship, Einstein field): mean
  held-out reprojection error fell from 5.2px (k1=0) to 2.8px (k1 free),
  improving 8 of 9 tags. Real signal, not overfitting.

Net result on `match2.mp4`'s `main` view, after also fixing the two bugs
below: RMS 105.52px (garbage — see sparse-tag filter) → 16.09px (drop the
garbage) → 1.81px (re-search f) → 1.95px with the full ladder including k1;
individual tags land within 1.1–2.1px of a fresh single-frame detection they
were never fit against.

---

## Bugs found along the way (not calibration-method choices, just bugs)

### AT3 corner order is `BL,BR,TR,TL`, not `TL,TR,BR,BL`

`tag_corners_field()`'s local corner array assumed AT3's `det.corners` (and
therefore `mean_corners` in the tags JSON) came back top-left-first. When
per-corner correspondences were added (see next section), the "most
head-on" tags got dramatically *worse* residuals (~20px) than the centroid
mode they replaced — a giveaway, since centroid mode is order-independent
and was fine. Comparing detected vs. projected pixel coordinates directly
showed a clean top/bottom flip: corners AT3 reports first are consistently
the *bottom* of the tag, not the top. Reversing the local array's order
fixed it outright (per-corner error on the affected tags: ~20px → ~1px).
Fixed in `pipeline/03_solve_pose.py`, `viz/03_overlay.py`,
`viz/09_calibrate_ui.py` — all three had their own copy of this geometry
(this codebase duplicates small helpers per-file rather than sharing
imports; see any of those files' `_quat_to_rot`/`tag_corners_field` for the
established pattern).

### Sparse-tag observations corrupting the fit

A tag decoded in only 1–2 of 150 sampled frames has no averaging benefit —
its "mean" position is just that raw detection, full pixel noise and all.
On `match2.mp4`'s `main` view, two such tags (1 and 2 observations) alone
were enough to push RMS to 105px; dropping them (same f) brought it to 16px
before any other fix. `pipeline/03_solve_pose.py` now requires
`n_frames_detected >= MIN_FRAME_FRACTION (0.1) * n_frames_sampled` before a
tag is used in pose-solving at all.

---

## Per-tag hybrid: corners for head-on tags, centroid otherwise

Not a bug fix, a genuine capability added once the corner-order bug above
was out of the way. A tag's centroid (mean of its 4 corners) suppresses
per-frame corner jitter but throws away 3 of its 4 points; a tag viewed
close to head-on has corners that are inherently well-conditioned (very
little foreshortening to amplify pixel noise into angular error), so for
those tags the 4 corners are strictly more information for little extra
noise. `_quad_squareness()` scores how close a detected quad is to a true
square (1.0) vs. an oblique skew (→0); `HEAD_ON_SQUARENESS` gates which
tags get promoted to their 4 corners. Started at a conservative 0.85 (before
the corner-order bug was found and corners looked untrustworthy generally);
lowered to 0.5 once a threshold sweep post-fix showed every view's fit
staying stable or improving all the way down to 0.0 — there was no evidence
left to justify staying conservative, it had been the bug all along, not
AT3's corners.

---

### 7. Joint bounded fit, gated on held-out error ✓ **adopted, supersedes #6**

Approach #6 above was right about the ladder and wrong about three things,
each of which produced a confidently wrong answer on real footage. What
replaced them, in the order the failures were found:

- **The field layout was assumed, not measured.** `--year` defaulted to 2026,
  and `match3` is 2025 footage whose decoded tag IDs are *all* ≤ 22 — every
  one of which also exists in the 2026 layout at a different field position.
  No ID check can catch that. It corrupts every correspondence, and the fit
  responded with `f` pinned to the search floor and 284px RMS, cached with no
  diagnostic. `detect_field_layout()` now scores every layout on disk by
  geometry: match3 goes 450px → 0.5px on the right one, and match2/match4 go
  7–8px on 2026 vs 645–652px on 2025. `match7` turns out to be 2024, which
  was not even a candidate until layout detection learned to read raw WPILib
  `<year>_layout.json` as well as the converted `<year>_tags.json`.

- **Averaging assumed a static camera.** The broadcast cuts, and in 2025 the
  bottom strip re-lays-out from two panes to three at endgame (a climb camera
  appears between them), moving every pane. Averaging across that puts each
  tag's "mean" between two framings, and `MAX_SPREAD_PX` then reads the gap
  between clusters as detection jitter — dropping exactly the tags seen in
  *both* shots and keeping the ones seen in only one, which look stable
  because they have nothing to disagree with. On `match3/bot_left` that
  turned 7 tags spanning 1822×293px into a 3-tag cluster. `split_static_
  segments()` now splits on cuts and fits each shot as its own camera, taking
  the *dominant* shot as the view's pose (head and tail shots are pre/post
  match filler). This also un-broke `match4`'s two insets, whose "geometric
  degeneracy" was really teleop and endgame observations averaged together.

- **Coordinate descent over separate 1-D searches.** `f`, `cy` and pitch all
  shift the projection vertically, so the cost surface is a diagonal valley
  that axis-aligned steps crawl along rather than cross — `match2/main` and
  `match4/main` are the same geometry but stopped at cy=352/pitch 14.8° and
  cy=734/pitch 30.1°. And the (cx, cy) grid re-centred on the *current
  iterate* each round, so its "±40% of the crop" bound compounded until
  `match4/bot_left` reached cx=1159 on a 940px-wide crop. Solving all
  parameters jointly under absolute bounds (`fit_intrinsics_joint`) fixed
  both: residuals fell on every view that fits (`match2/main` 2.25 → 1.03px)
  and focal agreement across the four `main` fits tightened from 20% to 7%.

The deeper change is what counts as evidence. #6 admitted `cx,cy` and `k1` on
correspondence-*count* thresholds, which cannot distinguish points spread
across the frame from the same number packed into a corner of it —
`match4/bot_right` cleared the 20-point bar with 5 tags in a 414×44px band
and was handed a free principal point and a free k1. Parameters are now
admitted on their own error bars from the covariance, and the *fit* is judged
by leave-one-tag-out held-out error, which is the same test that justified k1
originally, now run at runtime instead of once by hand.

Held-out error also outranks in-sample residual where they disagree:
`match6/main` sits at 15.9px in-sample yet predicts a tag it never saw to
5.3px, and the direct measurement is better evidence than the proxy.

Two things the local covariance provably cannot see, so they are measured by
refitting instead: a cost surface with a second basin (`match2/bot_right`
reports σ_f under 15% at its solution, yet dropping any one of its four tags
moves f from 589px to 1700–1900px) and a fit resting on a single tag
(`fit_depends_on_single_tag`).

Finally, the module now **refuses**. An optimizer always returns its best
point; that is not the same as having found a fit. A refused view is absent
from the poses file with its reasons recorded, rather than emitting the
cameras-below-the-floor and 59°-pitch results this pipeline used to produce
silently. Across the 7 matches in `data/detections/`, 18 views: 13 fit, 5
refused, and every refusal is a view that genuinely cannot be solved from its
own correspondences.

---

## Current pipeline

```
pipeline/00_split_views.py   -> data/profiles/<stem>_layout.json
pipeline/01_detect_tags.py   -> data/detections/<stem>_tags.json
pipeline/03_solve_pose.py    -> data/calibration/<stem>_intrinsics.json (fit)
                                 data/detections/<stem>_poses.json      (pose)
viz/03_overlay.py            -> data/overlays/<stem>_<view>_overlay.jpg  (sanity check)
viz/07_field3d.py            -> viz/field3d.html                        (the visualizer)
viz/09_calibrate_ui.py       -> manual slider fallback, when a view has too
                                 few points for the automatic ladder to trust
```

---

## Remaining known limitations

- **`00_split_views.py` cannot handle a pane layout that changes mid-video.**
  In 2025 the bottom strip is two panes during auto/teleop and three during
  endgame (left / climb camera / right). `match3` defeated the splitter
  entirely: it returned a single full-width 1920×347 bottom strip containing
  *both* PIP cameras' tags, which is why `match3/bot_right` appears to be
  "missing" and why `bot_left` can never be solved — no single camera pose
  explains tags from two cameras (best 4-tag subset 11.7px, all 7 together
  230px). `03` correctly refuses it, but the fix belongs upstream.
- **Views with only 4–5 clustered tags** solve, but their focal length does
  not survive dropping a tag (`fit_depends_on_single_tag`, focal swings of
  20–25%). The camera *position* from these is corroborated — `bot_right`
  lands at x≈16.0–16.9 across match1/2/4 independently — but the focal should
  not be quoted. More points from the same few tag locations is not the same
  as more geometric diversity; there is no parameter-search fix, only
  more/different tags.
- **Extrapolation beyond tag coverage**: a genuinely correct camera model is
  a real geometric law and must project correctly to *any* point in the
  rigid scene, not just the ones it was fit to — so divergence specifically
  on far/unfit points (the field boundary, mostly) is itself diagnostic, not
  just "expected uncertainty". On `match2.mp4`'s `main` view the left field
  edge sits past every tag that camera ever decoded (nearest used tag is
  4.27m in from that edge) — no amount of better intrinsics modeling
  substitutes for a correspondence point actually keeping that region
  honest; it would need a tag visible there, which that camera's frame may
  simply not contain.
