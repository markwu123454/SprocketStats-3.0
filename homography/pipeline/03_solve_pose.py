#!/usr/bin/env python3
"""
Step 3 -- Fit intrinsics (K, k1) and solve camera pose (position + orientation)
relative to the FRC field, from AprilTag correspondences alone.

This used to be two scripts (`02_search_focal.py` fit intrinsics,
`03_solve_pose.py` solved pose from them) glued together by `02` loading
`03` via an `importlib` hack, because `03`'s filename starts with a digit
and can't be `import`ed normally. They were never really separable -- fitting
intrinsics means optimizing a pose solve -- so they are one file now. See
docs/pose_calibration_research.md for the history of what was tried before
landing here.

Per-view flow: detect which field layout the footage was shot on, split the
sampled frames into runs of static framing, then for each run reuse cached
intrinsics if present, otherwise fit f (plus cx/cy and k1 where the data can
measure them), validate the fit against held-out tags, and solve pose from
whatever survived. --focal-px / --no-search bypass fitting entirely for the
"I already trust this calibration" case.

The single most important property of this module: it REFUSES. Every stage
below can conclude that a view is not solvable, and a refused view is left
out of the poses file rather than given a confident-looking wrong answer. An
optimizer always returns its best point; that is not the same as having
found a fit, and the difference is the whole design.


Which field layout (detect_field_layout)
----------------------------------------
Not taken from --year, because trusting it was a silent single point of
failure. match3 is 2025 footage whose decoded tag IDs are all <= 22 -- every
one of which also exists in the 2026 layout, at a completely different field
position. No ID-membership check can catch that; it just makes every
correspondence wrong, and the intrinsics fit responds by bottoming out at
284px RMS with f pinned to the search floor. Geometry can tell them apart
and does so decisively: match3 goes 450px -> 0.5px on the right layout,
while match2/match4 go 7-8px on 2026 and 645-652px on 2025. Both the
converted <year>_tags.json and raw WPILib <year>_layout.json files are
candidates, so a season nobody has converted is still detectable.


Static shots (split_static_segments)
------------------------------------
Averaging each tag over every sampled frame assumes the framing holds still.
Broadcast footage cuts, and in 2025 the bottom strip re-lays-out from two
panes to three at endgame (a climb camera appears between them), moving every
pane. Averaging across that puts each tag's "mean" position between two
unrelated framings, and the MAX_SPREAD_PX jitter gate then reads the distance
between those clusters as detection noise -- dropping precisely the tags
visible in BOTH shots and keeping the ones that appear in only one, which
look stable because they have nothing to disagree with. On match3/bot_left
that turned 7 tags spanning 1822x293px into a 3-tag cluster and reported the
discarded tags as "unstable detections".

So frames are split into runs of static framing first and each run is fit as
its own camera. The view's headline pose is its DOMINANT run (most sampled
frames), not its best-fitting one -- the short runs at the head and tail are
pre-match and post-match filler, and a pose solved from those is correct for
frames nobody wants to track. Other runs ride along in "other_segments".


Intrinsics (fit_intrinsics_joint)
---------------------------------
Every fit tried before this one in this repo solved too many parameters
(fx, fy, cx, cy, k1, k2, p1, p2, plus 6 pose DOF) from too few
correspondences, and underconstrained optimization doesn't fail loudly -- it
converges on a confident-looking wrong answer. So the model starts as small
as it can be (fx = fy = f, principal point at the crop centre, no distortion)
and grows only where the data supports growth.

What decides "supports" is each parameter's own error bar, from the
covariance at the solution -- not a correspondence-point count, which was the
previous rule and cannot see geometry. match4/bot_right cleared a 20-point
bar with 5 tags packed into a 414x44px band and was handed a free principal
point and a free k1 on that basis, returning k1=-0.118 and a principal point
below the bottom of its own crop.

All parameters are solved together, under absolute bounds, rather than by
coordinate descent over separate 1-D searches. f, cy and camera pitch all
shift the projection vertically, so the cost surface is a long diagonal
valley that axis-aligned steps crawl along instead of crossing; and the old
principal-point grid re-centred on the current iterate each round, so its
"+/-40% of the crop" bound compounded until match4/bot_left reached cx=1159
on a 940px-wide crop. Joint solving fixed both, and lowered residuals across
every view that fits (match2/main 2.25 -> 1.03px) while tightening focal
agreement between the four main-camera fits from 20% to 7%.


Trusting a fit (the acceptance gate)
------------------------------------
Held-out error is the criterion: refit without each tag in turn and ask where
the model puts that tag's known field position (cross_validate_fit). A camera
model that is a real geometric law predicts a point it never saw; anything
that merely fits well in-sample does not have to. In-sample residual is only
the cheap stand-in used when there are too few tags to hold any out --
match6/main sits at 15.9px in-sample yet predicts unseen tags to 5.3px, and
the direct measurement wins over the proxy.

A fit is refused when held-out error is bad, when the in-sample residual is
bad and there was no hold-out available, when the focal comes to rest on its
own search bound (where the range was cut, not a minimum the data chose), or
when there is not enough residual redundancy for a fit to mean anything at
all -- a single tag gives 8 residuals against 7 unknowns and will always
report a near-zero RMS, not because it is right but because it has almost no
freedom left in which to be wrong. match5/main reported 0.08px that way.

Two conditions warn rather than refuse, because the pose can be usable while
the focal is not: a focal whose 1-sigma bar exceeds MAX_FOCAL_REL_SIGMA, and
a fit whose focal swings more than MAX_FOCAL_SWING_ON_DROP when any single
tag is removed (fit_depends_on_single_tag -- the local covariance cannot see
this, since it describes only the basin the optimizer stopped in).

Solved poses are also checked for physical possibility -- above the floor,
near the building. Residual is scale-blind and an earlier revision happily
reported a camera 3.16m underground with a low one.

Rejected fits are cached with "trusted": false so they can be inspected, and
are never reused as intrinsics; a cached fit from a different field layout is
likewise re-searched rather than silently reused.


Pose solve (solve_view) -- per-tag hybrid: corners for head-on tags,
centroid otherwise
------------------------------------------------------------------------
At broadcast resolution (15-50 px tags) AT3 corner positions jitter +/-2-3 px
per frame from H.264 block-boundary artifacts. For a 30 px tag that
translates to 5-10 deg of corner-angle error -- but that error comes from
FORESHORTENING amplifying a few px of pixel noise into a large angular
one, so it's worst for tags seen at an oblique angle and much smaller for
tags seen close to head-on. The centroid of 4 corners suppresses this
jitter regardless of angle, at the cost of throwing away 3 of a tag's 4
points.

So: _collect_points() computes each tag's "squareness" (_quad_squareness --
how close its detected corner quad is to a true square, which is what a
physical AprilTag's projection is only when viewed head-on) and uses that
tag's 4 corners instead of its centroid when squareness clears
HEAD_ON_SQUARENESS -- more, cleaner points for the tags that can support
them, centroid for the rest where corner noise would hurt more than help.
The 3D point for a corner is that corner's known field position (from
tag_corners_field, the tag's known field position/orientation plus
TAG_SIZE_M); for a centroid it's just the tag's field position.

A view whose points -- even after using every eligible tag's corners --
fall short of MIN_TAGS_FOR_CENTERS relaxes further: use corners for EVERY
tag that has corner data, regardless of squareness. A single tag falls back
to exactly its 4 corners; that's a perfectly planar point set carrying a real
2-fold pose ambiguity at near-fronto-parallel angles, and it will not clear
the redundancy gate above. Output records what ran as "point_mode".

Apparent-size consistency check
--------------------------------
After solving, the expected apparent tag size (pixels) at each tag's solved
distance is compared against the observed mean_size_px:

  expected_size_px = TAG_SIZE_M * K[0,0] / distance_m

A deviation beyond 20% suggests the focal length is off.

Aggregated vs per-frame
-----------------------
Primary output uses mean_center_px (aggregated across a static shot) for each
tag, giving the most stable estimate. Pass --per-frame to also solve each
sampled frame independently and report temporal spread.

Outputs
-------
  data/calibration/<stem>_intrinsics.json
    merged into any existing file. Keyed by view for a single-shot view, and
    by "<view>@<first>-<last>" for each extra shot when the framing cuts (the
    dominant shot is also mirrored under the plain view name, so consumers
    that look up intrinsics by view keep working):
      { "<view>": { "K": [[f,0,cx],[0,f,cy],[0,0,1]], "dist": [k1,0,0,0,0],
                    "image_size": [w,h], "focal_px": f, "cx": ..., "cy": ...,
                    "k1": ..., "rms_reproj_all_px": ..., "max_reproj_px": ...,
                    "sigma": {"f":..,"cx":..,"cy":..,"k1":..},  // 1-sigma
                    "focal_rel_sigma": ...,                     // sigma_f / f
                    "n_residuals": ..., "n_free_params": ...,
                    "cross_validation": {"held_out_median_px": ...,
                                         "focal_swing_rel": ..., ...},
                    "fit": "focal_search_1d" | "focal_and_pp_search"
                           | "focal_pp_k1_search" | "focal_k1_search",
                    "field_year": "2026",
                    "trusted": true|false, "rejected_reasons": [...],
                    "diagnostics": [...] }, ... }

  data/detections/<stem>_poses.json
    {
      "<view>": {
        "camera_position_field_m": [x, y, z],
        "rvec": [r0, r1, r2],          // OpenCV Rodrigues
        "tvec_m": [t0, t1, t2],        // in camera frame
        "yaw_deg": ..., "pitch_deg": ..., "roll_deg": ...,
        "rms_reproj_px": ...,
        "point_mode": "hybrid (N corner tag(s) + M center tag(s))",
        "n_tags_used": ..., "n_points_used": ..., "n_points_inlier": ...,
        "segment": {"index":.., "n_segments":.., "frame_range":[..],
                    "n_frames_sampled":.., "label":".."},
        "other_segments": [ ... ],     // the view's non-dominant shots
        "tag_residuals": { "<tag_id>": {...} },
        "diagnostics": [{"severity": "warning" | "failure", "code": "...",
                         "detail": "..."}, ...],
          // every step of this view's solve (intrinsics included) that found
          // something worth flagging. Empty list = clean solve. A view that
          // was REFUSED has no entry in this file at all -- its reasons are
          // in the intrinsics file under "rejected_reasons".
        "per_frame": [...]             // if --per-frame
      }
    }

Usage
-----
  python pipeline/03_solve_pose.py --video match.mp4
  python pipeline/03_solve_pose.py --video match.mp4 --view main --per-frame
  python pipeline/03_solve_pose.py --video match.mp4 --refit          # re-fit intrinsics
  python pipeline/03_solve_pose.py --video match.mp4 --focal-px 1100  # skip fitting
  python pipeline/03_solve_pose.py --video match.mp4 --year 2025      # force a layout
  python pipeline/03_solve_pose.py --video match.mp4 --no-cross-validate

Install: pip install opencv-python numpy scipy
"""


import argparse, io, json, math, pathlib, sys
from contextlib import redirect_stderr
import numpy as np
import cv2
from scipy.optimize import least_squares

DATA_DIR       = pathlib.Path(__file__).parent.parent / "data"
DETECTIONS_DIR = DATA_DIR / "detections"
CALIB_DIR      = DATA_DIR / "calibration"
FIELD_DIR      = DATA_DIR / "field"

TAG_SIZE_M   = 0.1651
FIELD_YEAR   = 2026

# Nominal FRC field footprint, used only for the plausibility check on a
# solved camera position (see solve_view). Broadcast cameras sit in the
# stands, so being outside the field is normal -- being far outside is not.
FIELD_LENGTH_M = 16.541
FIELD_WIDTH_M  = 8.211
MAX_CAMERA_OFFFIELD_M = 12.0
DEFAULT_FOV  = 70.0
MAX_REPROJ   = 30.0   # RANSAC inlier threshold (px)

DEFAULT_F_MIN = 250.0
DEFAULT_F_MAX = 3000.0

# Floor on total correspondence points (not tags -- a corner-mode tag
# contributes 4) below which solvePnP doesn't have enough to work with.
MIN_TAGS_FOR_CENTERS = 4

# Minimum number of frames a tag must appear in to be used. 3 is enough to
# average out single-frame false positives; stability is checked separately
# via MAX_SPREAD_PX rather than a high frame-count floor.
MIN_FRAMES = 3

# A tag whose per-frame center positions spread more than this across the
# sampled frames is drifting (false detections at different locations, or
# a non-static object) and gets dropped regardless of frame count.
MAX_SPREAD_PX = 5.0

# _quad_squareness() >= this counts a tag as "head-on enough" to trust its
# 4 corners over its centroid -- see solve_view()'s comment for why.
# Lowered from an initial 0.85 once the real bug turned out to be
# tag_corners_field's corner-order mismatch (see git history), not AT3's
# corner accuracy itself -- with that fixed, a swept comparison across
# 0.85/0.75/0.65/0.55/0.45/0.0 on this project's own footage showed every
# view's fit stayed stable (or improved: "main" 8.87->8.38px) all the way
# down, so there was no real evidence left to justify staying conservative.
# 0.5 still excludes a genuinely degenerate quad (near-0 squareness),
# it just stops second-guessing AT3's corners on the strength of a
# perspective-skew heuristic alone.
HEAD_ON_SQUARENESS = 0.5

# NOTE: MIN_POINTS_FOR_PRINCIPAL_POINT / MIN_POINTS_FOR_DISTORTION used to
# live here, gating cx/cy and k1 on a raw correspondence-point count. They
# are gone: a count cannot distinguish points spread across the frame from
# the same number of points packed into a corner of it, and it was the
# packed case that broke. match4/bot_right cleared the 20-point bar with 5
# tags inside a 414x44px band and was handed a free principal point and a
# free k1 on that basis. Both parameters are now admitted on their own
# error bars instead -- see fit_intrinsics_joint().

# Acceptance gate. A search always returns the lowest point it found; that is
# not the same as having found a fit. If even the best f in range can't get
# all-correspondences RMS under this, the cost surface has no basin and the
# "minimum" is just the least-bad point on it -- refuse rather than cache a
# confident-looking wrong answer. Empirically this separates cleanly on this
# project's own footage with nothing in between: views that genuinely solve
# bottom out at 0.9-8.3px, views that can't (a degenerate tag cluster, or the
# wrong field layout) bottom out at 37-520px.
MAX_ACCEPT_RMS_PX = 15.0

# A view needs at least this many usable tags before its residual is worth
# scoring a candidate field layout on -- below it, a view has enough freedom
# to fit almost any layout and its low RMS means nothing.
MIN_TAGS_FOR_LAYOUT_PROBE = 4


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n < 1e-9:
        return np.eye(3)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
    ])


def rvec_to_ypr(rvec) -> tuple[float, float, float]:
    """Yaw/pitch/roll (degrees) from a Rodrigues rotation vector.

    The rotation maps from world to camera, so R.T gives camera axes in world
    coords. We decompose as: yaw (Z-up rotation), pitch (elevation), roll.
    """
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
    Rt = R.T  # camera-to-world rotation
    # Camera Z axis (optical axis) in world frame
    cz = Rt[:, 2]
    yaw   = math.degrees(math.atan2(cz[1], cz[0]))
    pitch = math.degrees(math.asin(np.clip(-cz[2], -1.0, 1.0)))
    # Roll: angle of camera X axis about optical axis
    cx = Rt[:, 0]
    roll  = math.degrees(math.atan2(cx[2], cx[0]))
    return yaw, pitch, roll


def camera_position(rvec, tvec) -> np.ndarray:
    """Convert OpenCV (rvec, tvec) to camera position in world frame."""
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
    return (-R.T @ np.array(tvec, dtype=np.float64).reshape(3)).tolist()


def tag_corners_field(tag: dict, half: float = TAG_SIZE_M / 2.0) -> np.ndarray:
    """
    (4, 3) field-frame corners of a tag, in the same order AT3 returns for
    det.corners (and that 01_detect_tags.py's mean_corners therefore
    preserves) -- empirically BL,BR,TR,TL, see the order note in the
    local-corners array below. Same WPILib-local corner geometry,
    quaternion rotation as viz/03_overlay.py::_tag_corners_field.
    """
    # NOTE: this order is BL,BR,TR,TL, not the TL/TR/BR/BL the variable
    # names below suggest -- empirically confirmed against real AT3
    # detections (see git history): projecting the "obvious" TL,TR,BR,BL
    # order against real det.corners-derived mean_corners put every
    # corner ~20px off along a consistent top/bottom flip, while this
    # (reversed) order lands within ~1-2px. AT3's actual corner winding
    # doesn't match the naive reading of "local +Y=right, +Z=up" order.
    local = np.array([[0, -half, -half], [0,  half, -half],
                      [0,  half,  half], [0, -half,  half]])
    R = quat_to_rot(tag["qw"], tag["qx"], tag["qy"], tag["qz"])
    t = np.array([tag["x"], tag["y"], tag["z"]])
    return (R @ local.T).T + t


def _quad_squareness(corners_px) -> float:
    """
    How close a detected corner quad is to a true square, in [0, 1].

    A physical AprilTag is a square, so its perspective projection is a
    square only when viewed head-on (assuming square pixels), and skews
    into a general quadrilateral -- unequal edges, unequal diagonals --
    the more oblique the viewing angle. min(shortest/longest edge,
    shortest/longest diagonal) is 1.0 for a true square and drops toward
    0 as the skew grows, giving a single number solve_view() gates
    corner-vs-centroid on per tag.
    """
    c = np.array(corners_px, dtype=np.float64)
    edges = [float(np.linalg.norm(c[(i + 1) % 4] - c[i])) for i in range(4)]
    d1 = float(np.linalg.norm(c[2] - c[0]))
    d2 = float(np.linalg.norm(c[3] - c[1]))
    edge_ratio = min(edges) / max(edges) if max(edges) > 0 else 0.0
    diag_ratio = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 0.0
    return min(edge_ratio, diag_ratio)


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

def load_or_estimate_K(intrinsics_path: pathlib.Path, view_name: str,
                       view_box: list, fov_deg: float, focal_px: float | None):
    if intrinsics_path.exists():
        intr = json.loads(intrinsics_path.read_text())
        if view_name in intr:
            K_list = intr[view_name]["K"]
            dist   = intr[view_name]["dist"]
            K  = np.array(K_list, dtype=np.float64)
            d  = np.array(dist,   dtype=np.float64)
            print(f"  [intrinsics] loaded from {intrinsics_path.name}  "
                  f"f={K[0,0]:.1f}px", file=sys.stderr)
            return K, d
    # Fall back to FOV estimate
    x0, y0, x1, y1 = view_box
    w, h = x1 - x0, y1 - y0
    if focal_px is not None:
        f = focal_px
    else:
        f = (w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)
    d = np.zeros(5, dtype=np.float64)
    print(f"  [intrinsics] estimated  f={f:.1f}px  fov={fov_deg:.1f}deg",
          file=sys.stderr)
    return K, d


# ---------------------------------------------------------------------------
# Shot segmentation
# ---------------------------------------------------------------------------
#
# Everything downstream assumes one camera pose per view, built by averaging
# each tag's position over every sampled frame. That is only valid while the
# framing holds still, and in broadcast footage it doesn't: the feed cuts.
#
# When it cuts, averaging across the cut puts every tag's "mean" position
# somewhere between two unrelated framings, and MAX_SPREAD_PX then reads the
# distance between those two clusters as detection jitter. The result is
# backwards -- it drops precisely the tags that were visible in BOTH shots
# (the well-observed ones, whose two positions differ) and keeps the tags
# that only appeared in one shot (which look perfectly stable because they
# only exist on one side of the cut). On this project's match3/bot_left that
# turned 7 tags spanning 1822x293px into a 3-tag cluster, and reported the
# discarded tags as "unstable detections".
#
# So: split the sampled frames into runs of static framing first, re-average
# within each run, and treat each run as its own camera. Cuts are obvious in
# this data -- within a shot consecutive sampled frames agree to ~0.2px,
# across a cut every co-visible tag jumps by tens to hundreds of px.

# Median inter-frame shift (over tags visible in both frames) above which two
# consecutive sampled frames are not the same shot. Far above real
# within-shot motion (<=1px here) and far below a real cut (tens of px).
SEGMENT_CUT_PX = 8.0

# A run shorter than this can't support the MIN_FRAMES tag gate anyway, and
# is usually a single stray frame at a transition.
MIN_SEGMENT_FRAMES = MIN_FRAMES


def _observations_by_frame(view_data: dict) -> dict:
    """frame_idx -> {tag_id: center_px as np.array}."""
    per_frame: dict[int, dict[int, np.ndarray]] = {}
    for tid_str, tag_data in view_data.get("decoded_tags", {}).items():
        tid = int(tid_str)
        for obs in tag_data.get("observations", []):
            c = obs.get("center_px")
            if c is None:
                corners = obs.get("corners")
                if corners and len(corners) == 4:
                    c = np.mean(np.array(corners, dtype=np.float64), axis=0)
            if c is not None:
                per_frame.setdefault(obs["frame_idx"], {})[tid] = np.asarray(
                    c, dtype=np.float64)
    return per_frame


def _cut_points(per_frame: dict) -> list[list[int]]:
    """Group sorted frame indices into runs of static framing."""
    frames = sorted(per_frame)
    if not frames:
        return []
    runs, cur = [], [frames[0]]
    for a, b in zip(frames, frames[1:]):
        common = set(per_frame[a]) & set(per_frame[b])
        if common:
            shift = float(np.median([np.linalg.norm(per_frame[b][t] - per_frame[a][t])
                                     for t in common]))
            is_cut = shift > SEGMENT_CUT_PX
        else:
            # No tag survives across the boundary, so there is no evidence
            # the framing held. Treating it as a cut is the conservative
            # reading -- merging two shots is the failure mode that matters.
            is_cut = True
        if is_cut:
            runs.append(cur)
            cur = []
        cur.append(b)
    runs.append(cur)
    return runs


def _reaggregate(view_data: dict, frames: set) -> dict:
    """
    A view-shaped dict whose decoded_tags are re-averaged over just `frames`.

    Produces exactly the fields solve_view() reads (mean_center_px,
    mean_corners, mean_size_px, n_frames_detected, observations), so the
    solver needs no knowledge that segmentation happened.
    """
    out_tags = {}
    for tid_str, tag_data in view_data.get("decoded_tags", {}).items():
        obs = [o for o in tag_data.get("observations", []) if o["frame_idx"] in frames]
        if not obs:
            continue
        centers, corners_stack, sizes, margins = [], [], [], []
        for o in obs:
            c = o.get("center_px")
            cor = o.get("corners")
            if cor and len(cor) == 4:
                corners_stack.append(np.array(cor, dtype=np.float64))
                if c is None:
                    c = np.mean(corners_stack[-1], axis=0)
            if c is not None:
                centers.append(np.asarray(c, dtype=np.float64))
            if o.get("size_px") is not None:
                sizes.append(float(o["size_px"]))
            if o.get("decision_margin") is not None:
                margins.append(float(o["decision_margin"]))
        if not centers:
            continue
        entry = {
            "n_frames_detected": len(obs),
            "mean_center_px": np.mean(centers, axis=0).tolist(),
            "mean_size_px": float(np.mean(sizes)) if sizes else 0.0,
            "observations": obs,
        }
        if corners_stack:
            entry["mean_corners"] = np.mean(np.stack(corners_stack), axis=0).tolist()
        if margins:
            entry["mean_decision_margin"] = float(np.mean(margins))
        out_tags[tid_str] = entry
    seg = dict(view_data)
    seg["decoded_tags"] = out_tags
    seg["n_frames_sampled"] = len(frames)
    return seg


def split_static_segments(view_name: str, view_data: dict) -> list[dict]:
    """
    [{index, frames (first,last), n_frames, view_data}], one per static shot.

    Returns a single whole-view segment when the framing never cuts, so the
    common case is byte-for-byte the old behaviour.
    """
    per_frame = _observations_by_frame(view_data)
    runs = [r for r in _cut_points(per_frame) if len(r) >= MIN_SEGMENT_FRAMES]
    if len(runs) <= 1:
        return [{"index": 0, "frames": None, "n_frames": len(per_frame),
                 "view_data": view_data, "is_only": True}]
    dropped = sum(1 for r in _cut_points(per_frame) if len(r) < MIN_SEGMENT_FRAMES)
    print(f"  [{view_name}] framing cuts {len(runs)} time(s) across the sampled "
          f"frames -- solving each shot separately"
          + (f" ({dropped} sub-{MIN_SEGMENT_FRAMES}-frame fragment(s) ignored)"
             if dropped else ""), file=sys.stderr)
    segs = []
    for i, run in enumerate(runs):
        segs.append({"index": i, "frames": (run[0], run[-1]), "n_frames": len(run),
                     "view_data": _reaggregate(view_data, set(run)), "is_only": False})
    return segs


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def _collect_points(view_name: str, view_data: dict, field_tags: dict,
                    diag) -> dict | None:
    """
    Usable tags -> correspondence arrays, independent of any camera model.

    Split out of solve_view() so the intrinsics optimiser can work on exactly
    the points the pose solve will use, instead of reaching them indirectly
    by re-running a whole pose solve per candidate parameter vector. Nothing
    in here depends on K or dist -- which tags survive, and whether a tag
    contributes 1 centroid or 4 corners, is a property of the detections
    alone. Returns None (having diag()'d why) when there is nothing solvable.
    """
    decoded = view_data.get("decoded_tags", {})
    if not decoded:
        diag("failure", "no_decoded_tags", "no decoded tags — skipping")
        return None

    # Collect usable tags. A tag passes if it clears two gates:
    #   1. Seen in at least MIN_FRAMES frames (filters single-frame flukes).
    #   2. Its per-frame center positions don't spread more than MAX_SPREAD_PX
    #      (filters false detections at different locations, which would make
    #      mean_center_px meaningless). A stable tag seen in only 3 frames is
    #      fine; a jittery tag seen in 100 frames is not. Note this runs on a
    #      single static shot (see split_static_segments) -- across a cut,
    #      every tag looks "jittery" and this gate would drop the good ones.
    usable, too_few, jittery = [], [], []
    for tid_str, tag_data in decoded.items():
        tid = int(tid_str)
        ft = field_tags.get(tid_str) or field_tags.get(str(tid))
        if ft is None:
            diag("warning", "tag_not_in_field", f"tag {tid} not in field layout — skipping")
            continue
        if tag_data.get("n_frames_detected", 0) < MIN_FRAMES:
            too_few.append(tid)
            continue
        centers = []
        for obs in tag_data.get("observations", []):
            c = obs.get("center_px")
            if c is not None:
                centers.append(c)
            else:
                corners = obs.get("corners")
                if corners and len(corners) == 4:
                    centers.append([sum(p[0] for p in corners) / 4,
                                    sum(p[1] for p in corners) / 4])
        if len(centers) >= 2:
            arr = np.array(centers, dtype=np.float64)
            spread = float(np.sqrt(arr.var(axis=0).sum()))
            if spread > MAX_SPREAD_PX:
                jittery.append((tid, round(spread, 1)))
                continue
        usable.append({
            "tid":        tid,
            "field_tag":  ft,
            "center_px":  tag_data["mean_center_px"],
            "size_px":    tag_data.get("mean_size_px", 0.0),
            "corners_px": tag_data.get("mean_corners"),
        })
    if too_few:
        diag("warning", "insufficient_frames",
             f"tag(s) {too_few} seen in < {MIN_FRAMES} frames — dropping")
    if jittery:
        diag("warning", "unstable_detection",
             f"tag(s) {[(t, f'{s}px') for t, s in jittery]} "
             f"position spread > {MAX_SPREAD_PX}px — dropping (unstable detections)")

    if not usable:
        diag("failure", "no_usable_tags", "no usable tags — skipping")
        return None

    # Per-tag hybrid: a tag's centroid suppresses per-frame corner jitter
    # (see module docstring) into one clean point, but throws away 3 of its
    # 4 points. A tag viewed close to head-on -- its corner quad measures
    # close to a true rectangle -- has corners that are inherently
    # well-conditioned (foreshortening is what turns a few px of corner
    # jitter into several degrees of angular error, and there's very
    # little of it here), so for those tags the 4 corners are strictly
    # more information for little extra noise: use them instead of the
    # centroid. A tag seen at an oblique angle keeps its centroid -- its
    # corners are exactly the noisy case the centroid exists to fix.
    for u in usable:
        u["squareness"] = (_quad_squareness(u["corners_px"])
                           if u["corners_px"] and len(u["corners_px"]) == 4 else 0.0)
        u["use_corners"] = u["squareness"] >= HEAD_ON_SQUARENESS

    n_points = sum(4 if u["use_corners"] else 1 for u in usable)
    if n_points < MIN_TAGS_FOR_CENTERS and any(u["corners_px"] for u in usable):
        # Not enough points even to reach centers-only's own floor -- relax
        # the head-on bar to "has corners at all", since more (noisier)
        # points beat too few points to solve at all.
        for u in usable:
            if u["corners_px"] and len(u["corners_px"]) == 4:
                u["use_corners"] = True
        n_points = sum(4 if u["use_corners"] else 1 for u in usable)
        diag("warning", "corner_relaxation",
             f"only {n_points} point(s) from head-on tags alone "
             f"— relaxing to all available corners")

    n_corner_tags = sum(1 for u in usable if u["use_corners"])
    point_mode = (f"hybrid ({n_corner_tags} corner tag(s) + "
                  f"{len(usable) - n_corner_tags} center tag(s))")

    obj_pts, img_pts, point_tag_ids = [], [], []
    for u in usable:
        if u["use_corners"]:
            for c_field, c_px in zip(tag_corners_field(u["field_tag"]), u["corners_px"]):
                obj_pts.append(c_field.tolist())
                img_pts.append(c_px)
                point_tag_ids.append(u["tid"])
        else:
            ft = u["field_tag"]
            obj_pts.append([ft["x"], ft["y"], ft["z"]])
            img_pts.append(u["center_px"])
            point_tag_ids.append(u["tid"])

    if len(obj_pts) < 4:
        diag("failure", "insufficient_points",
             f"only {len(obj_pts)} point(s) total (need >= 4 for solvePnP) — skipping")
        return None

    return {
        "usable": usable,
        "obj_pts": obj_pts, "img_pts": img_pts,
        "obj": np.array(obj_pts, dtype=np.float64).reshape(-1, 1, 3),
        "img": np.array(img_pts, dtype=np.float64).reshape(-1, 1, 2),
        "point_tag_ids": point_tag_ids,
        "point_mode": point_mode,
        "n_corner_tags": n_corner_tags,
    }


def solve_view(view_name: str, view_data: dict, field_tags: dict,
               K: np.ndarray, dist: np.ndarray,
               do_per_frame: bool) -> dict | None:
    # Every step below that finds something worth flagging appends here via
    # diag() -- {"severity": "warning"|"failure", "code": ..., "detail": ...}.
    # Folded into the returned dict as "diagnostics" on success. A step that
    # aborts the whole solve (diag("failure", ...) followed by `return None`)
    # still gets its diag() call printed to stderr same as always, it's just
    # not around afterward to attach to an output object -- the view is
    # simply absent from the caller's poses_out, same as before this existed.
    diagnostics = []

    def diag(severity: str, code: str, detail: str):
        diagnostics.append({"severity": severity, "code": code, "detail": detail})
        print(f"  [{view_name}] {detail}", file=sys.stderr)

    pts = _collect_points(view_name, view_data, field_tags, diag)
    if pts is None:
        return None
    usable        = pts["usable"]
    obj_pts       = pts["obj_pts"]
    point_tag_ids = pts["point_tag_ids"]
    point_mode    = pts["point_mode"]
    obj_arr       = pts["obj"]
    img_arr       = pts["img"]

    if pts["n_corner_tags"]:
        print(f"  [{view_name}] using 4-corner points for head-on tag(s) "
              f"{[u['tid'] for u in usable if u['use_corners']]}, "
              f"centroid for the rest ({len(obj_pts)} points total from "
              f"{len(usable)} tags)", file=sys.stderr)
    if len(usable) == 1:
        diag("warning", "single_tag_ambiguous",
             "single-tag pose — coplanar point set, real ambiguity risk, "
             "treat as low-confidence")

    # A point set drawn from a single tag spans only TAG_SIZE_M, and OpenCV's
    # SQPNP asserts (rather than returning false) when the object points have
    # near-zero coordinate variance -- an uncaught C++ assertion that takes
    # the whole run down. Shot segmentation makes single-tag point sets
    # routine, since a brief shot may only ever show one tag.
    obj_extent = float(np.sqrt(obj_arr.reshape(-1, 3).var(axis=0).sum()))
    if obj_extent < 1e-3:
        diag("failure", "degenerate_object_points",
             f"all {len(obj_pts)} object point(s) are effectively coincident "
             f"(spread {obj_extent:.2e} m) — no pose is recoverable")
        return None

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_arr, img_arr, K, dist,
        iterationsCount  = 2000,
        reprojectionError= MAX_REPROJ,
        confidence       = 0.999,
        flags            = cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        # Nearly-coplanar tag configurations (same Z) degenerate RANSAC's
        # 4-point minimal solver.  Fall back to SQPNP on all points.
        diag("warning", "ransac_degenerate",
             f"RANSAC failed — falling back to SQPNP on all {len(obj_pts)} pts")
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_arr, img_arr, K, dist,
                flags=cv2.SOLVEPNP_SQPNP,
            )
        except cv2.error as e:
            # SQPNP signals some degeneracies by assertion rather than a
            # false return; a solver refusing to solve is a normal outcome
            # here, not a crash.
            diag("failure", "solvepnp_failed",
                 f"SQPNP rejected this point configuration as degenerate "
                 f"({str(e).strip().splitlines()[-1][:120]}) — skipping")
            return None
        if not ok:
            diag("failure", "solvepnp_failed", "solvePnP also failed — skipping")
            return None
        inliers = None  # treat all as inliers for reporting

    n_inliers = len(inliers) if inliers is not None else len(obj_pts)
    inlier_set = (set(int(i) for i in inliers.flatten())
                  if inliers is not None else set(range(len(obj_pts))))

    # Refine with LM on inliers only
    if n_inliers >= 4:
        inl_obj = obj_arr[sorted(inlier_set)]
        inl_img = img_arr[sorted(inlier_set)]
        ok2, rvec, tvec = cv2.solvePnP(
            inl_obj, inl_img, K, dist,
            rvec, tvec, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    rvec = rvec.flatten().tolist()
    tvec = tvec.flatten().tolist()

    # Per-point reprojection error
    reproj, _ = cv2.projectPoints(obj_arr, np.array(rvec), np.array(tvec), K, dist)
    reproj = reproj.reshape(-1, 2)
    img_flat = img_arr.reshape(-1, 2)
    point_errs = np.linalg.norm(reproj - img_flat, axis=1)
    rms_all    = float(np.sqrt(np.mean(point_errs**2)))
    cam_pos    = camera_position(rvec, tvec)
    yaw, pitch, roll = rvec_to_ypr(rvec)

    # Aggregate per tag_id -- 1 point/tag in center mode, 4 points/tag in
    # corner mode. "inlier" means a majority of that tag's points were
    # RANSAC inliers (all 4 agree in practice; a corner tag rarely splits).
    points_by_tag: dict[int, list[int]] = {}
    for i, tid in enumerate(point_tag_ids):
        points_by_tag.setdefault(tid, []).append(i)
    usable_by_tid = {u["tid"]: u for u in usable}

    tag_residuals = {}
    for tid, idxs in points_by_tag.items():
        u          = usable_by_tid[tid]
        ft         = u["field_tag"]
        n_inl_pts  = sum(1 for i in idxs if i in inlier_set)
        dist_m     = float(np.linalg.norm(np.array([ft["x"], ft["y"], ft["z"]])
                                          - np.array(cam_pos)))
        exp_size   = TAG_SIZE_M * K[0, 0] / dist_m if dist_m > 0.1 else 0.0
        tag_residuals[str(tid)] = {
            "reproj_px":        round(float(np.mean(point_errs[idxs])), 2),
            "inlier":           n_inl_pts >= (len(idxs) + 1) // 2,
            "n_points":         len(idxs),
            "dist_m":           round(dist_m, 2),
            "expected_size_px": round(exp_size, 1),
            "observed_size_px": round(u["size_px"], 1),
        }

    # Inlier-only RMS, computed directly from per-point errors (not via the
    # per-tag aggregate above, since that's a mean and would double-count
    # unevenly in corner mode).
    if inlier_set:
        inl_errs = point_errs[sorted(inlier_set)]
        rms_inliers = float(np.sqrt(np.mean(inl_errs**2)))
    else:
        rms_inliers = rms_all

    print(f"  [{view_name}] pos=[{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]m  "
          f"yaw={yaw:.1f}°  pitch={pitch:.1f}°  "
          f"rms={rms_inliers:.2f}px ({n_inliers}/{len(obj_pts)} points, "
          f"{len(usable)} tags, mode={point_mode})",
          file=sys.stderr)

    # Physical plausibility. Reprojection residual is scale-blind and says
    # nothing about whether the answer is possible: the pre-gate pipeline
    # once reported a camera 3.16m BELOW the floor with a low residual, and
    # a 59-degree downward pitch, because nothing ever asked. A camera has
    # to be above the floor and somewhere near the building.
    if cam_pos[2] < -0.5:
        diag("warning", "implausible_camera_height",
             f"solved camera sits {-cam_pos[2]:.2f}m below the floor "
             f"(z={cam_pos[2]:.2f}m) — geometrically impossible, the fit has "
             f"converged on a mirrored or otherwise wrong solution")
    elif cam_pos[2] > 25.0:
        diag("warning", "implausible_camera_height",
             f"solved camera is {cam_pos[2]:.1f}m up — higher than any plausible "
             f"broadcast position")
    dist_from_field = max(
        0.0, -min(cam_pos[0], cam_pos[1]),
        cam_pos[0] - FIELD_LENGTH_M, cam_pos[1] - FIELD_WIDTH_M)
    if dist_from_field > MAX_CAMERA_OFFFIELD_M:
        diag("warning", "implausible_camera_position",
             f"solved camera is {dist_from_field:.1f}m outside the field footprint "
             f"(pos=[{cam_pos[0]:.2f}, {cam_pos[1]:.2f}]m, field is "
             f"{FIELD_LENGTH_M:.1f}x{FIELD_WIDTH_M:.1f}m) — further out than a "
             f"camera in the stands would be")

    # Size consistency check
    for tid_str, res in tag_residuals.items():
        exp = res["expected_size_px"]
        obs = res["observed_size_px"]
        if obs > 0 and exp > 0:
            ratio = exp / obs
            if ratio < 0.7 or ratio > 1.4:
                detail = (f"tag {tid_str} size mismatch: expected {exp:.1f}px "
                          f"observed {obs:.1f}px (ratio={ratio:.2f}) "
                          f"-- focal length may be off")
                diagnostics.append({"severity": "warning", "code": "size_mismatch",
                                    "detail": detail})
                print(f"  [warn] {detail}", file=sys.stderr)

    out = {
        "camera_position_field_m": [round(v, 4) for v in cam_pos],
        "rvec":       [round(v, 6) for v in rvec],
        "tvec_m":     [round(v, 4) for v in tvec],
        "yaw_deg":    round(yaw,   2),
        "pitch_deg":  round(pitch, 2),
        "roll_deg":   round(roll,  2),
        "rms_reproj_px":      round(rms_inliers, 3),
        "rms_reproj_all_px":  round(rms_all, 3),
        "point_mode":         point_mode,
        "n_tags_used":        len(usable),
        "n_points_used":      len(obj_pts),
        "n_points_inlier":    n_inliers,
        "tag_residuals":      tag_residuals,
        "diagnostics":        diagnostics,
    }

    if do_per_frame:
        out["per_frame"] = _per_frame(view_data, field_tags, K, dist, rvec, tvec)

    return out


# ---------------------------------------------------------------------------
# Per-frame solving
# ---------------------------------------------------------------------------

def _per_frame(view_data: dict, field_tags: dict,
               K: np.ndarray, dist: np.ndarray,
               seed_rvec, seed_tvec) -> list:
    decoded = view_data.get("decoded_tags", {})
    # Collect all frame indices that appear in at least 1 tag
    frame_map: dict[int, dict[int, list]] = {}  # frame_idx -> {tag_id -> center_px}
    for tid_str, tag_data in decoded.items():
        tid = int(tid_str)
        if tid_str not in field_tags and str(tid) not in field_tags:
            continue
        for obs in tag_data.get("observations", []):
            fi = obs["frame_idx"]
            c  = obs.get("center_px")
            if c is None:
                corners = np.array(obs.get("corners", []))
                if len(corners) == 4:
                    c = corners.mean(axis=0).tolist()
                else:
                    continue
            frame_map.setdefault(fi, {})[tid] = c

    results = []
    for fi, tag_centers in sorted(frame_map.items()):
        if len(tag_centers) < 4:
            continue
        obj_pts, img_pts = [], []
        for tid, center in tag_centers.items():
            ft = field_tags.get(str(tid))
            if ft is None:
                continue
            obj_pts.append([ft["x"], ft["y"], ft["z"]])
            img_pts.append(center)
        if len(obj_pts) < 4:
            continue
        obj_arr = np.array(obj_pts, dtype=np.float64).reshape(-1, 1, 3)
        img_arr = np.array(img_pts, dtype=np.float64).reshape(-1, 1, 2)
        ok, rvec_f, tvec_f = cv2.solvePnP(
            obj_arr, img_arr, K, dist,
            np.array(seed_rvec), np.array(seed_tvec),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        reproj, _ = cv2.projectPoints(obj_arr, rvec_f, tvec_f, K, dist)
        rms = float(np.sqrt(np.mean(np.sum(
            (reproj.reshape(-1, 2) - img_arr.reshape(-1, 2))**2, axis=1))))
        pos = camera_position(rvec_f.flatten(), tvec_f.flatten())
        results.append({
            "frame_idx":    fi,
            "n_tags":       len(obj_pts),
            "rms_reproj_px": round(rms, 3),
            "camera_position_field_m": [round(v, 3) for v in pos],
        })
    return results


# ---------------------------------------------------------------------------
# Intrinsics search (1-D focal search + coordinate-descent principal
# point / distortion ladder)
# ---------------------------------------------------------------------------

def _cost(f: float, view_name: str, view_data: dict, field_tags: dict,
         cx: float, cy: float, k1: float = 0.0) -> float:
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    with redirect_stderr(io.StringIO()):   # solve_view is chatty; silence during search
        result = solve_view(view_name, view_data, field_tags, K, dist,
                            do_per_frame=False)
    if result is None:
        return math.inf
    # Deliberately rms_reproj_all_px (every correspondence), NOT the
    # RANSAC inlier-only rms_reproj_px -- confirmed on this project's own
    # "main" view that the inlier-only number can be gamed: at f=375px
    # RANSAC quietly drops 5/10 tags as "outliers" and reports a great
    # rms=2.8 on the remaining 5, while those same 5 dropped tags land
    # off by hundreds of pixels (all-points rms=1570) -- a wrong-by-2.5x
    # focal length faking a low residual via a smaller inlier set. The
    # real, broad, physically-plausible basin (f=850-1000, matching this
    # project's earlier hand-fit and the CV605-BK spec range) only shows
    # up once every point has to agree, not just whichever subset RANSAC
    # kept.
    return result["rms_reproj_all_px"]


# ---------------------------------------------------------------------------
# Joint intrinsics + pose refinement
# ---------------------------------------------------------------------------
#
# This replaces a ladder of three separate 1-D searches (golden section on f,
# a 13x13 grid on cx/cy, golden section on k1) driven by coordinate descent.
# Two failures of that design, both observed on this project's own footage:
#
#   * f, cy and camera pitch are strongly coupled -- all three shift the
#     projection vertically -- so the cost surface is a long diagonal valley.
#     Coordinate descent may only step along the axes, so it crawls down the
#     valley instead of across it and stops wherever it started from.
#     match2/main and match4/main are the same camera geometry yet landed on
#     cy=352 with pitch 14.8deg and cy=734 with pitch 30.1deg.
#   * The (cx, cy) grid re-centred on the CURRENT iterate each round rather
#     than on the crop, so its "+/-40% of the crop" bound compounded: three
#     rounds could walk the principal point ~1100px. match4/bot_left reached
#     cx=1159 on a 940px-wide crop -- a principal point off the image.
#
# Solving the intrinsics together with the 6 pose DOF, under absolute bounds,
# in one bounded trust-region least-squares removes both: the optimiser sees
# the coupling in the Jacobian, and the bounds mean what they say. It also
# yields the thing the old flatness heuristic was approximating -- a real
# covariance, so "is this parameter actually measured" stops being a
# two-sample finite difference and becomes an error bar.

# Residual scale (px) at which the robust loss starts discounting a point.
# Good fits here sit at 1-3px, so ordinary noise stays in the quadratic
# regime while a single bad correspondence stops dragging the whole solution.
ROBUST_F_SCALE_PX = 3.0

# Identifiability requires redundancy: a parameter is only pinned down by
# observations beyond those needed to determine it. This is the minimum
# excess of residuals over free parameters for a fit to mean anything. A
# single tag supplies 8 residuals against 7 unknowns (f + 6 pose) and will
# always report a near-zero RMS -- not because it is right, but because it
# has almost no freedom left in which to be wrong. match5/main is exactly
# that case and reported rms=0.08px.
MIN_RESIDUAL_REDUNDANCY = 8

# 1-sigma bars beyond which a parameter counts as unsupported by the data.
MAX_FOCAL_REL_SIGMA = 0.15    # sigma_f / f
MAX_PP_SIGMA_FRAC   = 0.25    # sigma_cx / crop width (and cy / height)
MIN_K1_SNR          = 2.0     # |k1| / sigma_k1

# theta layout shared by everything below.
_I_F, _I_CX, _I_CY, _I_K1 = 0, 1, 2, 3
_I_POSE = [4, 5, 6, 7, 8, 9]
_PARAM_LABEL = {_I_F: "f", _I_CX: "cx", _I_CY: "cy", _I_K1: "k1"}


def _theta_to_Kd(theta):
    f, cx, cy, k1 = theta[:4]
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _reproj_residuals(theta, obj, img):
    """Flat (2N,) reprojection residual vector, in pixels."""
    K, dist = _theta_to_Kd(theta)
    proj, _ = cv2.projectPoints(obj,
                                np.ascontiguousarray(theta[4:7]),
                                np.ascontiguousarray(theta[7:10]),
                                K, dist)
    return (proj.reshape(-1, 2) - img.reshape(-1, 2)).ravel()


def _param_sigmas(jac, resid, free_idx, n_theta):
    """
    1-sigma uncertainty per free parameter from the Jacobian at the solution,
    via cov ~= sigma^2 (J'J)^-1.

    Returns NaN for a parameter whenever the data cannot support an error bar
    at all -- no residual degrees of freedom left, or a rank-deficient J'J,
    which means some direction in parameter space leaves the residuals
    untouched and that combination simply is not measured. (Under a robust
    loss this variance estimate is approximate; it is used to judge whether a
    parameter is determined to within tens of percent, not to publish.)
    """
    sig = np.full(n_theta, np.nan)
    m, n = jac.shape
    dof = m - n
    if dof <= 0:
        return sig
    s2 = float(resid @ resid) / dof
    JtJ = jac.T @ jac
    sv = np.linalg.svd(JtJ, compute_uv=False)
    if sv[0] <= 0 or sv[-1] / sv[0] < 1e-12:
        return sig
    cov = np.linalg.pinv(JtJ) * s2
    d = np.diag(cov)
    for k, i in enumerate(free_idx):
        sig[i] = math.sqrt(d[k]) if d[k] > 0 else np.nan
    return sig


def _refine(obj, img, theta0, free_idx, bounds):
    """Bounded robust least-squares over the free subset of theta."""
    theta = np.array(theta0, dtype=np.float64)
    free_idx = list(free_idx)
    lo = np.array([bounds[i][0] for i in free_idx], dtype=np.float64)
    hi = np.array([bounds[i][1] for i in free_idx], dtype=np.float64)
    x0 = np.clip(theta[free_idx], lo, hi)

    def fun(x):
        t = theta.copy()
        t[free_idx] = x
        return _reproj_residuals(t, obj, img)

    res = least_squares(fun, x0, bounds=(lo, hi), loss="soft_l1",
                        f_scale=ROBUST_F_SCALE_PX, x_scale="jac", max_nfev=500)
    t = theta.copy()
    t[free_idx] = res.x
    resid = _reproj_residuals(t, obj, img)
    per_pt = np.linalg.norm(resid.reshape(-1, 2), axis=1)
    span = np.maximum(np.abs(hi - lo), 1e-9)
    return {
        "theta": t,
        "rms": float(np.sqrt(np.mean(per_pt ** 2))),
        "max_err": float(per_pt.max()),
        "sigma": _param_sigmas(res.jac, resid, free_idx, len(theta)),
        "free": free_idx,
        "n_resid": len(resid),
        "at_bound": [i for k, i in enumerate(free_idx)
                     if (res.x[k] - lo[k]) / span[k] < 1e-4
                     or (hi[k] - res.x[k]) / span[k] < 1e-4],
    }


def seed_scan(view_name: str, view_data: dict, field_tags: dict,
              cx: float, cy: float, f_min: float, f_max: float,
              n_samples: int = 40):
    """
    Coarse sweep over f for the basin, and a pose seed for the joint solve.

    Log-spaced because f is a scale parameter: the old fixed 25px step was
    10% of f at the bottom of the range and 0.8% at the top, spending most of
    its 110 samples where they resolved the least. 40 log-spaced samples give
    a uniform ~7% relative resolution across the whole range.

    Scored on all-correspondences RMS, never the RANSAC inlier-only number:
    at f=375 on this project's main view RANSAC quietly drops 5 of 10 tags as
    "outliers" and reports a fine rms=2.8 on the rest while those dropped
    tags sit hundreds of px away. Only a cost every point must answer for
    shows the real basin.
    """
    best = None
    for f in np.geomspace(f_min, f_max, n_samples):
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        with redirect_stderr(io.StringIO()):
            r = solve_view(view_name, view_data, field_tags, K, np.zeros(5), False)
        if r is None:
            continue
        if best is None or r["rms_reproj_all_px"] < best["rms"]:
            best = {"rms": r["rms_reproj_all_px"], "f": float(f),
                    "rvec": r["rvec"], "tvec": r["tvec_m"]}
    return best


def fit_intrinsics_joint(view_name: str, view_data: dict, field_tags: dict,
                         w: int, h: int, f_min: float, f_max: float,
                         allow_pp: bool, allow_k1: bool):
    """
    Fit f -- then cx/cy, then k1 -- jointly with pose, admitting each extra
    parameter only when the data measures it.

    The ladder is unchanged in spirit (start from the simplest model that
    could be right; add a parameter only once it is earned) but the test for
    "earned" is no longer a hardcoded point count. A count cannot see
    geometry: match4/bot_right cleared the old 20-point bar with 5 tags
    packed into a 414x44px band and was handed a free principal point and a
    free k1 on that basis, returning k1=-0.118 and a cy below the bottom of
    its own crop. What decides whether cx/cy or k1 are measurable is how the
    correspondences are spread -- which is precisely what each parameter's
    own error bar reports.
    """
    diagnostics = []

    def diag(severity, code, detail):
        diagnostics.append({"severity": severity, "code": code, "detail": detail})
        print(f"  [{view_name}] {detail}", file=sys.stderr)

    pts = _collect_points(view_name, view_data, field_tags, lambda *a: None)
    if pts is None:
        return None
    obj, img = pts["obj"], pts["img"]
    n_resid = 2 * len(pts["obj_pts"])

    seed = seed_scan(view_name, view_data, field_tags, w / 2.0, h / 2.0, f_min, f_max)
    if seed is None:
        return None

    theta0 = np.array([seed["f"], w / 2.0, h / 2.0, 0.0,
                       *seed["rvec"], *seed["tvec"]], dtype=np.float64)
    # Absolute bounds, anchored to the crop and never to the current iterate.
    # A view that is a sub-window of a larger sensor readout can legitimately
    # have its principal point outside the crop -- match4/main's cy=686 on a
    # 710px crop beats the crop centre 2.05px vs 6.31px -- so the box is a
    # full crop-dimension either side of centre: generous, but fixed.
    bounds = [(f_min, f_max),
              (-w / 2.0, 1.5 * w), (-h / 2.0, 1.5 * h),
              (-0.5, 0.5),
              (-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0),
              (-500.0, 500.0), (-500.0, 500.0), (-500.0, 500.0)]

    rungs = [("focal_search_1d", [_I_F] + _I_POSE)]
    if allow_pp:
        rungs.append(("focal_and_pp_search", [_I_F, _I_CX, _I_CY] + _I_POSE))
        if allow_k1:
            rungs.append(("focal_pp_k1_search", [_I_F, _I_CX, _I_CY, _I_K1] + _I_POSE))
    elif allow_k1:
        rungs.append(("focal_k1_search", [_I_F, _I_K1] + _I_POSE))

    def identifiable(idx, r):
        s = r["sigma"][idx]
        if not np.isfinite(s):
            return False, f"{_PARAM_LABEL[idx]} is not measurable at all here " \
                          f"(rank-deficient normal equations)"
        if idx == _I_CX:
            return (s < MAX_PP_SIGMA_FRAC * w,
                    f"cx uncertain to +/-{s:.0f}px on a {w}px-wide crop")
        if idx == _I_CY:
            return (s < MAX_PP_SIGMA_FRAC * h,
                    f"cy uncertain to +/-{s:.0f}px on a {h}px-tall crop")
        if idx == _I_K1:
            k1 = r["theta"][_I_K1]
            return (abs(k1) > MIN_K1_SNR * s,
                    f"k1={k1:+.4f} +/- {s:.4f}, indistinguishable from zero")
        return True, ""

    best = best_name = None
    for name, free in rungs:
        if n_resid - len(free) < MIN_RESIDUAL_REDUNDANCY:
            msg = (f"{n_resid} residuals against {len(free)} free parameters "
                   f"leaves {n_resid - len(free)} degree(s) of freedom, below the "
                   f"{MIN_RESIDUAL_REDUNDANCY} needed for a residual to mean "
                   f"anything")
            if best is None:
                diag("failure", "insufficient_redundancy",
                     msg + " — this view cannot support even a focal-only fit")
                return {"underdetermined": True, "detail": msg,
                        "diagnostics": diagnostics,
                        "n_resid": n_resid, "n_free": len(free)}
            diag("warning", "insufficient_redundancy",
                 msg + f" — stopping at {best_name}")
            break

        r = _refine(obj, img, theta0 if best is None else best["theta"], free, bounds)
        if best is None:
            best, best_name = r, name
            continue

        added = [i for i in free if i not in best["free"]]
        verdicts = [(i,) + identifiable(i, r) for i in added]
        unsupported = [(i, why) for i, ok, why in verdicts if not ok]
        if unsupported:
            diag("warning", "parameter_not_identifiable",
                 "not adding " + "/".join(_PARAM_LABEL[i] for i, _ in unsupported)
                 + " — " + "; ".join(why for _, why in unsupported))
            break
        if r["rms"] >= best["rms"] * 0.95:
            print(f"  [{view_name}] not adding "
                  f"{'/'.join(_PARAM_LABEL[i] for i in added)} — residual "
                  f"{best['rms']:.2f}px -> {r['rms']:.2f}px is not a real "
                  f"improvement", file=sys.stderr)
            break
        best, best_name = r, name

    theta, sigma = best["theta"], best["sigma"]
    f_sigma = sigma[_I_F]
    f_rel = float(f_sigma / theta[_I_F]) if np.isfinite(f_sigma) else None
    return {
        "f": float(theta[_I_F]), "cx": float(theta[_I_CX]),
        "cy": float(theta[_I_CY]), "k1": float(theta[_I_K1]),
        "rms": best["rms"], "max_err": best["max_err"],
        "fit": best_name,
        "focal_rel_sigma": f_rel,
        "sigma": {_PARAM_LABEL[i]: (float(sigma[i]) if np.isfinite(sigma[i]) else None)
                  for i in (_I_F, _I_CX, _I_CY, _I_K1)},
        "at_bound": [_PARAM_LABEL.get(i, f"pose{i}") for i in best["at_bound"]],
        "n_resid": best["n_resid"], "n_free": len(best["free"]),
        "seed_rms": seed["rms"], "seed_f": seed["f"],
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Leave-one-tag-out cross-validation
# ---------------------------------------------------------------------------
#
# The covariance in fit_intrinsics_joint is a LOCAL measure: it describes the
# curvature of the cost right where the optimiser stopped. That is the right
# tool for "is this parameter flat", but it is blind to a cost surface with
# more than one basin -- and this data has those. match2/bot_right reports
# sigma_f under 15% at its solution, yet dropping any single one of its four
# tags and refitting moves f from 589px to 1700-1900px. The local bar says
# "determined"; the data says the solution moved to a different basin the
# moment anything changed.
#
# Held-out error is the check that cannot be fooled this way, and it is the
# same test that justified admitting k1 in the first place (see the module
# docstring): drop a tag, refit everything without it, then ask where the
# model puts that tag's KNOWN field position. A camera model that is a real
# geometric law predicts a point it never saw. Nothing that merely fits well
# in-sample has to.
#
# One caveat this deliberately reports rather than hides: when a view's
# conditioning rests on a single well-placed tag, removing that tag leaves a
# degenerate remainder, and the huge held-out error that follows says the FIT
# IS FRAGILE, not that the tag is bad. match2/bot_left is exactly that -- four
# tags in a 121px cluster plus tag 30 far off to the side carrying all the
# geometry. Both facts are worth surfacing; conflating them is not.

# A refit missing one tag that moves the focal more than this is a fit resting
# on that one tag rather than on the tag set as a whole.
MAX_FOCAL_SWING_ON_DROP = 0.25

# Median held-out reprojection error above which the model is not predicting
# geometry it did not see. Held to the same bar as the in-sample gate: median
# (not max) so the single-load-bearing-tag case above doesn't trip it alone.
MAX_HELD_OUT_RMS_PX = MAX_ACCEPT_RMS_PX

# Below this many usable tags there is no meaningful hold-out to do -- every
# fold would be degenerate and the numbers would describe the folds, not the fit.
MIN_TAGS_FOR_CROSS_VALIDATION = 4


def cross_validate_fit(view_name: str, view_data: dict, field_tags: dict,
                       w: int, h: int, f_min: float, f_max: float,
                       allow_pp: bool, allow_k1: bool, full_f: float) -> dict | None:
    """
    Refit once per held-out tag; report how well the model generalises and
    how much it leans on any single tag. None when there aren't enough tags
    for the exercise to mean anything.
    """
    decoded = view_data.get("decoded_tags", {})
    with redirect_stderr(io.StringIO()):
        probe = solve_view(view_name, view_data, field_tags,
                           np.array([[1000.0, 0, w/2], [0, 1000.0, h/2], [0, 0, 1]]),
                           np.zeros(5), False)
    if probe is None:
        return None
    tids = sorted(int(t) for t in probe["tag_residuals"])
    if len(tids) < MIN_TAGS_FOR_CROSS_VALIDATION:
        return None

    folds = []
    for tid in tids:
        held = dict(view_data)
        held["decoded_tags"] = {t: d for t, d in decoded.items() if int(t) != tid}
        with redirect_stderr(io.StringIO()):
            r = fit_intrinsics_joint(view_name, held, field_tags, w, h,
                                     f_min, f_max, allow_pp, allow_k1)
            if r is None or r.get("underdetermined"):
                continue
            K = np.array([[r["f"], 0, r["cx"]], [0, r["f"], r["cy"]], [0, 0, 1]])
            dist = np.array([r["k1"], 0.0, 0.0, 0.0, 0.0])
            pose = solve_view(view_name, held, field_tags, K, dist, False)
        if pose is None:
            continue
        ft = field_tags.get(str(tid))
        td = decoded.get(str(tid))
        if ft is None or td is None:
            continue
        # Score the held-out tag with the same point convention the solver
        # would have used for it.
        if td.get("mean_corners") and len(td["mean_corners"]) == 4:
            obj = tag_corners_field(ft)
            img = np.array(td["mean_corners"], dtype=np.float64)
        else:
            obj = np.array([[ft["x"], ft["y"], ft["z"]]], dtype=np.float64)
            img = np.array([td["mean_center_px"]], dtype=np.float64)
        proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3),
                                    np.array(pose["rvec"], dtype=np.float64),
                                    np.array(pose["tvec_m"], dtype=np.float64), K, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
        folds.append({"tag": tid, "held_out_px": err, "f": r["f"],
                      "pos": pose["camera_position_field_m"]})

    if len(folds) < MIN_TAGS_FOR_CROSS_VALIDATION - 1:
        return None

    errs = np.array([f["held_out_px"] for f in folds])
    fs = np.array([f["f"] for f in folds])
    swing = float(np.max(np.abs(fs - full_f)) / full_f) if full_f > 0 else None
    worst = max(folds, key=lambda f: abs(f["f"] - full_f))
    return {
        "n_folds": len(folds),
        "held_out_median_px": float(np.median(errs)),
        "held_out_max_px": float(errs.max()),
        "focal_range_px": [float(fs.min()), float(fs.max())],
        "focal_spread_rel": float((fs.max() - fs.min()) / fs.mean()) if fs.mean() else None,
        "focal_swing_rel": swing,
        "most_load_bearing_tag": worst["tag"],
        "per_tag": [{"tag": f["tag"], "held_out_px": round(f["held_out_px"], 2),
                     "f": round(f["f"], 1)} for f in folds],
    }


def _fit_intrinsics_entry(view_name: str, vdata: dict, field_tags: dict, w: int, h: int,
                          args, field_year: str | None = None) -> dict | None:
    """Fit one view's intrinsics and return its cache entry (the dict that
    goes into <stem>_intrinsics.json), or None if nothing could be fit."""
    n_tags = len(vdata.get("decoded_tags", {}))

    with redirect_stderr(io.StringIO()):
        probe = solve_view(view_name, vdata, field_tags,
                           np.array([[1000.0, 0, w/2], [0, 1000.0, h/2], [0, 0, 1]]),
                           np.zeros(5), False)
    n_points = probe["n_points_used"] if probe else 0

    # Whether cx/cy and k1 are ADMITTED is decided by their own error bars
    # inside fit_intrinsics_joint, not here -- these flags only carry the
    # user's explicit --no-principal-point / --no-distortion opt-outs. The
    # old MIN_POINTS_FOR_* count gates are gone: a count cannot tell a well
    # spread set of correspondences from a tight cluster of the same size,
    # and it was the cluster that broke this (see fit_intrinsics_joint).
    allow_pp = not args.no_principal_point
    allow_k1 = not args.no_distortion

    print(f"  [{view_name}] {n_tags} decoded tag(s), {n_points} correspondence point(s), "
          f"fitting f in [{args.f_min:.0f}, {args.f_max:.0f}]px"
          f"{' (+cx,cy if measurable)' if allow_pp else ''}"
          f"{' (+k1 if measurable)' if allow_k1 else ''} ...", file=sys.stderr)

    result = fit_intrinsics_joint(view_name, vdata, field_tags, w, h,
                                  args.f_min, args.f_max, allow_pp, allow_k1)
    if result is None:
        print(f"  [{view_name}] no focal length in range produced a valid pose "
              f"solve -- skipping", file=sys.stderr)
        return None

    # --- Acceptance gate -------------------------------------------------
    # Everything above reports the best point the optimiser reached. These
    # checks decide whether that point is a fit at all. A rejected entry is
    # still written to the cache (with trusted=false) so it can be
    # inspected, but it is never reused as intrinsics and no pose is solved
    # from it -- see _resolve_intrinsics / main().
    if result.get("underdetermined"):
        detail = result["detail"]
        return {
            "image_size": [w, h], "field_year": field_year,
            "trusted": False,
            "rejected_reasons": [detail],
            "n_correspondence_points": n_points,
            "diagnostics": result["diagnostics"],
        }

    diagnostics = list(result["diagnostics"])
    rejected = []

    # Held-out validation runs BEFORE the residual gate, because when both
    # are available held-out error is the better criterion and the in-sample
    # residual is only its cheap stand-in. match6/main is the case that
    # settles the ordering: in-sample rms 15.9px, but leave-one-tag-out puts
    # a tag it never saw within 5.3px. A model that predicts unseen geometry
    # to 5px is a working model whatever its in-sample number says, and
    # rejecting it on the proxy while the direct measurement passes would be
    # preferring the weaker evidence.
    cv = None
    if not args.no_cross_validate:
        cv = cross_validate_fit(view_name, vdata, field_tags, w, h,
                                args.f_min, args.f_max, allow_pp, allow_k1,
                                result["f"])

    # With cross-validation in hand the in-sample bar drops to a loose sanity
    # check; without it (too few tags to hold any out, or --no-cross-validate)
    # it is the only thing standing between a non-fit and the output.
    rms_bar = MAX_ACCEPT_RMS_PX * 3 if cv is not None else MAX_ACCEPT_RMS_PX
    if result["rms"] > rms_bar:
        rejected.append(
            f"best achievable rms_all={result['rms']:.1f}px exceeds "
            f"{rms_bar:.0f}px -- no camera model in range makes these "
            f"correspondences consistent, so this is the least-bad point on a "
            f"surface with no basin, not a fit")
    # A solution resting against the edge of its own bound is an artifact of
    # where the bound was drawn, not something the data picked out. Every
    # failing view in this project's footage pinned f to 250 (or 3000), which
    # also implies a physically absurd 124-151 degree horizontal FOV.
    if "f" in result["at_bound"]:
        hfov = 2 * math.degrees(math.atan((w / 2.0) / result["f"]))
        rejected.append(
            f"focal came to rest on the search bound (f={result['f']:.0f}px in "
            f"[{args.f_min:.0f}, {args.f_max:.0f}]px, implying a {hfov:.0f} deg "
            f"horizontal FOV) -- the cost is still descending at the edge of the "
            f"range, so this is where the search ran out of room, not a minimum")
    for r in rejected:
        diagnostics.append({"severity": "failure", "code": "intrinsics_rejected",
                            "detail": r})
        print(f"  [{view_name}] [reject] {r}", file=sys.stderr)

    # Replaces the old flatness_ratio heuristic (cost at f +/- 100px, an
    # absolute offset that meant 40% of f at one end of the range and 7% at
    # the other, and which silently reported "unknown" whenever a bound
    # blocked one side -- i.e. on exactly the fits that were failing). This
    # is the focal's own 1-sigma bar from the covariance, as a fraction of f.
    f_rel = result["focal_rel_sigma"]
    if f_rel is None:
        constraint = "focal uncertainty unavailable (no residual DOF)"
        diagnostics.append({
            "severity": "warning", "code": "poorly_constrained_focal",
            "detail": "focal uncertainty could not be estimated -- too few "
                      "independent observations to say how well f is pinned down"})
    elif f_rel > MAX_FOCAL_REL_SIGMA:
        constraint = f"POORLY CONSTRAINED, f = {result['f']:.0f} +/- {f_rel:.0%}"
        # A warning, not a rejection: on this project's inset views the focal
        # moves 20-25% across leave-one-tag-out folds while the solved camera
        # POSITION stays put and agrees with other matches' fits of the same
        # physical camera to within 0.3m. The pose is usable; the focal is
        # not something to quote.
        diagnostics.append({
            "severity": "warning", "code": "poorly_constrained_focal",
            "detail": f"f = {result['f']:.0f} +/- {result['sigma']['f']:.0f}px "
                      f"({f_rel:.0%}) -- these correspondences do not pin the focal "
                      f"length down; the pose may still be usable but do not trust f"})
    else:
        constraint = f"f = {result['f']:.0f} +/- {f_rel:.1%}"

    if cv is not None:
        if cv["held_out_median_px"] > MAX_HELD_OUT_RMS_PX:
            rejected.append(
                f"median held-out reprojection error {cv['held_out_median_px']:.1f}px "
                f"exceeds {MAX_HELD_OUT_RMS_PX:.0f}px across {cv['n_folds']} "
                f"leave-one-tag-out refits -- this model reproduces the tags it was "
                f"fit to but does not predict a tag it did not see, which is what a "
                f"camera model has to do")
            diagnostics.append({"severity": "failure", "code": "intrinsics_rejected",
                                "detail": rejected[-1]})
            print(f"  [{view_name}] [reject] {rejected[-1]}", file=sys.stderr)
        if cv["focal_swing_rel"] is not None and cv["focal_swing_rel"] > MAX_FOCAL_SWING_ON_DROP:
            # Deliberately a warning: the pose can still be usable and
            # cross-match consistent while f itself is not pinned down.
            detail = (f"dropping tag {cv['most_load_bearing_tag']} alone moves the "
                      f"focal by {cv['focal_swing_rel']:.0%} (folds span "
                      f"{cv['focal_range_px'][0]:.0f}-{cv['focal_range_px'][1]:.0f}px "
                      f"vs {result['f']:.0f}px on all tags) -- this fit rests on one "
                      f"tag rather than on the tag set, so treat f as unmeasured "
                      f"even though its local error bar looks tight")
            diagnostics.append({"severity": "warning",
                                "code": "fit_depends_on_single_tag", "detail": detail})
            print(f"  [{view_name}] {detail}", file=sys.stderr)

    cx, cy, k1 = result["cx"], result["cy"], result["k1"]
    parts = [f"  [{view_name}] {constraint}  rms_all_pts={result['rms']:.2f}px "
             f"(max {result['max_err']:.1f}px)  model={result['fit']}"]
    if cv is not None:
        parts.append(f"  held-out {cv['held_out_median_px']:.2f}px "
                     f"(median of {cv['n_folds']})")
    if result["fit"] != "focal_search_1d":
        s = result["sigma"]
        if s["cx"] is not None and result["fit"] != "focal_k1_search":
            parts.append(f"  cx={cx:.1f}+/-{s['cx']:.0f} (Δ{cx - w/2:+.1f})"
                         f"  cy={cy:.1f}+/-{s['cy']:.0f} (Δ{cy - h/2:+.1f})")
        if s["k1"] is not None and "k1" in result["fit"]:
            parts.append(f"  k1={k1:+.4f}+/-{s['k1']:.4f}")
    print("".join(parts), file=sys.stderr)

    return {
        "K": [[result["f"], 0.0, cx], [0.0, result["f"], cy], [0.0, 0.0, 1.0]],
        "dist": [k1, 0.0, 0.0, 0.0, 0.0],
        "image_size": [w, h],
        "focal_px": round(result["f"], 1),
        "cx": round(cx, 1), "cy": round(cy, 1),
        "k1": round(k1, 5),
        # all-correspondences RMS, deliberately NOT RANSAC inlier-only -- see
        # seed_scan()'s docstring for why that distinction matters.
        "rms_reproj_all_px": round(result["rms"], 3),
        "max_reproj_px": round(result["max_err"], 3),
        # 1-sigma parameter uncertainties from the covariance at the solution.
        # focal_rel_sigma is the headline one: sigma_f / f.
        "sigma": {k: (round(v, 5) if v is not None else None)
                  for k, v in result["sigma"].items()},
        "focal_rel_sigma": round(f_rel, 4) if f_rel is not None else None,
        "n_residuals": result["n_resid"],
        "n_free_params": result["n_free"],
        "n_correspondence_points": n_points,
        # Leave-one-tag-out generalisation. held_out_median_px is the honest
        # headline: how far off this model puts a tag it never saw.
        "cross_validation": cv,
        "fit": result["fit"],
        # Which field layout these correspondences were fit against. Cached
        # intrinsics are only valid for the layout they were solved on, so a
        # later run against a different one must not reuse them.
        "field_year": field_year,
        "trusted": not rejected,
        "rejected_reasons": rejected,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Field layout detection
# ---------------------------------------------------------------------------

def load_field(path: pathlib.Path) -> dict:
    """Normalise any of the layout file shapes to {str(id): {x,y,z,qw..qz}}."""
    raw = json.loads(path.read_text())
    # WPILib AprilTagFieldLayout, i.e. data/field/<year>_layout.json as
    # fetched by pipeline/01_fetch_field_layout.py. Only some seasons have
    # been converted to this repo's own <year>_tags.json shape, and an
    # unconverted season would otherwise be invisible to layout detection --
    # which matters, since this project's footage spans 2024-2026.
    if isinstance(raw, dict) and "field-tags" in raw:
        out = {}
        for t in raw["field-tags"]:
            pose = t.get("pose", {})
            tr = pose.get("translation", {})
            q = pose.get("rotation", {}).get("quaternion", {})
            out[str(t["ID"])] = {
                "x": tr.get("x", 0.0), "y": tr.get("y", 0.0), "z": tr.get("z", 0.0),
                "qw": q.get("W", 1.0), "qx": q.get("X", 0.0),
                "qy": q.get("Y", 0.0), "qz": q.get("Z", 0.0),
            }
        return out
    if isinstance(raw, list):
        return {str(t["id"]): t for t in raw}
    if isinstance(raw, dict) and "tags" in raw:
        tags_val = raw["tags"]
        if isinstance(tags_val, list):
            return {str(t["id"]): t for t in tags_val}
        return {str(k): v for k, v in tags_val.items()}
    return {str(k): v for k, v in raw.items()}


def _layout_candidates() -> list[pathlib.Path]:
    """
    One layout file per season present on disk.

    Prefers this repo's converted <year>_tags.json where it exists, falling
    back to the raw WPILib <year>_layout.json otherwise, so a season nobody
    has converted yet is still a candidate rather than silently unavailable.
    """
    by_year: dict[str, pathlib.Path] = {}
    for path in sorted(FIELD_DIR.glob("*_layout.json")):
        by_year[path.stem.split("_")[0]] = path
    for path in sorted(FIELD_DIR.glob("*_tags.json")):
        by_year[path.stem.split("_")[0]] = path
    return [by_year[y] for y in sorted(by_year)]


def _layout_probe_rms(view_name: str, vdata: dict, field_tags: dict,
                      w: int, h: int, n_samples: int = 24) -> float:
    """
    Cheapest honest "could this layout be right" probe: best
    all-correspondences RMS over a coarse log-spaced focal sweep, principal
    point at the crop centre and no distortion. Deliberately the
    one-free-parameter model -- handing a wrong layout extra parameters to
    absorb the mismatch is exactly what would blur the signal.
    """
    best = math.inf
    for f in np.geomspace(DEFAULT_F_MIN, DEFAULT_F_MAX, n_samples):
        best = min(best, _cost(float(f), view_name, vdata, field_tags,
                               w / 2.0, h / 2.0, 0.0))
    return best


def detect_field_layout(views: dict, explicit_year: int | None):
    """
    Work out which season's field the footage was actually shot on, by
    geometry, and return (year_str, field_tags, confident).

    `confident` is False when no candidate layout actually fits, in which
    case year_str is a fallback picked on tag-ID coverage and should be
    treated as unknown -- see the fallback branch below.

    Trusting --year was a silent single point of failure. This project's
    match3 is 2025 footage: every tag it decodes has an ID <= 22, and every
    one of those IDs also exists in the 2026 layout at a completely
    different field position -- so no ID-membership check can catch it. It
    just makes every correspondence wrong, and the intrinsics search
    responds by bottoming out at 284px RMS with f pinned to the search
    floor, cached with no diagnostic.

    Geometry is the only thing that can tell them apart, and it does so
    unambiguously: under the correct layout at least one well-covered view
    solves to a few px, under the wrong one nothing gets near it
    (match3: 450px -> 0.5px; symmetrically match2/match4 sit at 7-8px on
    2026 and 645-652px on 2025).

    Scored on the single BEST view rather than an average across views: a
    wrong layout makes every view bad, whereas a correct layout can still
    contain views that are independently unsolvable for their own reasons
    (match4's two insets are degenerate under either layout), and averaging
    lets those drown out the one view that actually carries the signal.
    """
    candidates = _layout_candidates()
    if not candidates:
        sys.exit(f"[error] no field layouts found in {FIELD_DIR}")

    scored = []
    for path in candidates:
        year = path.stem.split("_")[0]
        ft = load_field(path)
        best_rms, best_view, n_scored = math.inf, None, 0
        hit = tot = 0
        for vname, vdata in views.items():
            x0, y0, x1, y1 = vdata["box"]
            w, h = x1 - x0, y1 - y0
            ids = list(vdata.get("decoded_tags", {}))
            tot += len(ids)
            hit += sum(1 for i in ids if str(i) in ft)
            with redirect_stderr(io.StringIO()):
                probe = solve_view(vname, vdata, ft,
                                   np.array([[1000.0, 0, w/2], [0, 1000.0, h/2],
                                             [0, 0, 1]]), np.zeros(5), False)
            if probe is None or probe["n_tags_used"] < MIN_TAGS_FOR_LAYOUT_PROBE:
                continue
            n_scored += 1
            r = _layout_probe_rms(vname, vdata, ft, w, h)
            if r < best_rms:
                best_rms, best_view = r, vname
        scored.append({"year": year, "field": ft, "rms": best_rms,
                       "view": best_view, "n_scored": n_scored,
                       "coverage": hit / tot if tot else 0.0})

    usable = [s for s in scored if s["n_scored"] > 0 and math.isfinite(s["rms"])]
    print("  [layout] scoring candidate field layouts "
          "(best-view rms over a coarse focal sweep):", file=sys.stderr)
    for s in sorted(scored, key=lambda s: s["rms"]):
        rms = f"{s['rms']:8.2f}px via {s['view']}" if math.isfinite(s["rms"]) else "  no scorable view"
        print(f"            {s['year']}  {rms}  "
              f"(tag IDs present: {100*s['coverage']:.0f}%)", file=sys.stderr)

    if not usable:
        # Nothing had enough tag coverage to score -- fall back rather than
        # guess, and say so.
        year = str(explicit_year or FIELD_YEAR)
        match = next((s for s in scored if s["year"] == year), None)
        if match is None:
            sys.exit(f"[error] no field layout for {year} in {FIELD_DIR} "
                     f"(have: {', '.join(s['year'] for s in scored)})")
        print(f"  [layout] no view has >= {MIN_TAGS_FOR_LAYOUT_PROBE} usable tags to "
              f"score on -- falling back to {year}", file=sys.stderr)
        return year, match["field"], False

    best = min(usable, key=lambda s: s["rms"])

    # Picking the least-bad layout is only meaningful if one of them is
    # actually good. When none is, the ranking is noise between wrong
    # answers and reporting a winner launders that into a stated fact.
    # match7 is the case: 2024 at 155px "beats" 2026 at 289px, but 2024
    # accounts for only 56% of its decoded tag IDs while 2026 accounts for
    # 100% -- the two signals disagree and neither layout explains the
    # geometry. Fall back to ID coverage, which is at least a fact about the
    # data rather than a comparison of two failures, and say plainly that
    # the layout is undetermined.
    confident = best["rms"] <= MAX_ACCEPT_RMS_PX
    if not confident:
        by_coverage = max(scored, key=lambda s: (s["coverage"], -s["rms"]))
        print(f"  [layout] !! no layout fits this footage -- the best "
              f"({best['year']}, {best['rms']:.0f}px on {best['view']}) is far past "
              f"the {MAX_ACCEPT_RMS_PX:.0f}px a real fit reaches, so this ranking is "
              f"comparing wrong answers. Falling back to the layout accounting for "
              f"the most decoded tag IDs ({by_coverage['year']}, "
              f"{by_coverage['coverage']:.0%}); treat the season as UNDETERMINED and "
              f"expect every view to be refused.", file=sys.stderr)
        best = by_coverage

    if explicit_year is not None:
        forced = str(explicit_year)
        match = next((s for s in scored if s["year"] == forced), None)
        if match is None:
            sys.exit(f"[error] no field layout for {forced} in {FIELD_DIR} "
                     f"(have: {', '.join(s['year'] for s in scored)})")
        if forced != best["year"] and confident:
            print(f"  [layout] !! --year {forced} was requested but {best['year']} fits "
                  f"this footage far better ({best['rms']:.2f}px vs {match['rms']:.2f}px "
                  f"on {best['view']}). Honouring --year; drop it to auto-detect.",
                  file=sys.stderr)
        return forced, match["field"], confident

    print(f"  [layout] using {best['year']} "
          f"({best['rms']:.2f}px on {best['view']})", file=sys.stderr)
    return best["year"], best["field"], confident


# ---------------------------------------------------------------------------
# Per-view intrinsics resolution -- cached search / explicit override / bypass
# ---------------------------------------------------------------------------

def _resolve_intrinsics(view_name: str, vdata: dict, field_tags: dict,
                        existing_intrinsics: dict, args, field_year: str | None = None):
    """Return (K, dist, cache_entry_or_None). cache_entry is the dict to
    store under existing_intrinsics[view_name] -- None means nothing new
    was computed (cached / explicit / no-search path)."""
    x0, y0, x1, y1 = vdata["box"]
    w, h = x1 - x0, y1 - y0

    if args.focal_px is not None:
        f = args.focal_px
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)
        d = np.zeros(5, dtype=np.float64)
        print(f"  [{view_name}] [intrinsics] explicit --focal-px={f:.1f}px "
              f"(search skipped)", file=sys.stderr)
        return K, d, None

    if view_name in existing_intrinsics and not args.refit:
        intr = existing_intrinsics[view_name]
        # A cached entry is only reusable if it was actually accepted, and
        # only for the layout it was fit against. Reusing either kind of
        # stale entry is how a bad fit becomes permanent: it gets written
        # once, then silently loaded on every subsequent run with no
        # warning, and only --refit would ever dislodge it.
        cached_year = intr.get("field_year")
        if intr.get("trusted") is False:
            print(f"  [{view_name}] [intrinsics] cached fit was rejected "
                  f"({'; '.join(intr.get('rejected_reasons') or ['no reason recorded'])}) "
                  f"-- re-searching", file=sys.stderr)
        elif field_year is not None and cached_year is not None and cached_year != field_year:
            print(f"  [{view_name}] [intrinsics] cached fit was solved against the "
                  f"{cached_year} field layout but this run is using {field_year} "
                  f"-- re-searching", file=sys.stderr)
        else:
            K = np.array(intr["K"], dtype=np.float64)
            d = np.array(intr["dist"], dtype=np.float64)
            print(f"  [{view_name}] [intrinsics] loaded cached  f={K[0,0]:.1f}px  "
                  f"(pass --refit to re-search)", file=sys.stderr)
            return K, d, None

    if args.no_search:
        f = (w / 2.0) / math.tan(math.radians(args.fov_deg / 2.0))
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)
        d = np.zeros(5, dtype=np.float64)
        print(f"  [{view_name}] [intrinsics] estimated from --fov-deg={args.fov_deg} "
              f"(search disabled)  f={f:.1f}px", file=sys.stderr)
        return K, d, None

    entry = _fit_intrinsics_entry(view_name, vdata, field_tags, w, h, args, field_year)
    if entry is None:
        # No f in range produced a valid pose solve at all. This used to fall
        # back to the crude --fov-deg guess and solve a pose from it anyway,
        # which is the same failure as any other rejected fit: a confident
        # output with nothing behind it. --no-search/--focal-px above are the
        # supported ways to say "use this f, I know what I'm doing".
        reason = (f"no focal length in [{args.f_min:.0f}, {args.f_max:.0f}]px produced "
                  f"a valid pose solve")
        print(f"  [{view_name}] [reject] {reason}", file=sys.stderr)
        return None, None, {
            "image_size": [w, h], "field_year": field_year,
            "trusted": False, "rejected_reasons": [reason],
            "diagnostics": [{"severity": "failure", "code": "intrinsics_rejected",
                             "detail": reason}],
        }

    if "K" not in entry:
        # Rejected before a camera matrix was ever formed -- an
        # underdetermined view, where the fit stopped rather than reporting
        # parameters it had no data for. main() drops the shot on
        # trusted=False; there is nothing to hand it.
        return None, None, entry

    K = np.array(entry["K"], dtype=np.float64)
    d = np.array(entry["dist"], dtype=np.float64)
    return K, d, entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video",     required=True, metavar="PATH")
    ap.add_argument("--view",      metavar="NAME",
                    help="process only this view (default: all views)")
    ap.add_argument("--year",      type=int, default=None,
                    help="force a field layout year. Default is to detect it from "
                         "the geometry (see detect_field_layout) -- tag IDs alone "
                         "can't tell seasons apart, and guessing wrong silently "
                         "corrupts every correspondence")
    ap.add_argument("--tags",      metavar="PATH",
                    help="path to tags JSON (default: data/detections/<stem>_tags.json)")
    ap.add_argument("--out",       metavar="PATH",
                    help="poses output path (default: data/detections/<stem>_poses.json)")
    ap.add_argument("--out-intrinsics", metavar="PATH",
                    help="intrinsics output path (default: "
                         "data/calibration/<stem>_intrinsics.json, merged)")
    ap.add_argument("--per-frame", action="store_true",
                    help="also solve pose per frame and include in output")

    ap.add_argument("--focal-px",  type=float, default=None,
                    help="skip intrinsics search, use this known focal length (px), "
                         "square pixels, principal point at crop centre, no distortion")
    ap.add_argument("--fov-deg",   type=float, default=DEFAULT_FOV,
                    help="fallback horizontal FOV (default: %(default)s), used only "
                         "with --no-search or if the search itself fails")
    ap.add_argument("--no-search", action="store_true",
                    help="don't run the intrinsics search -- use cached intrinsics if "
                         "present, else the crude --fov-deg/--focal-px estimate")
    ap.add_argument("--refit",     action="store_true",
                    help="re-run the intrinsics search even if this view already has "
                         "cached intrinsics")
    ap.add_argument("--f-min", type=float, default=DEFAULT_F_MIN)
    ap.add_argument("--f-max", type=float, default=DEFAULT_F_MAX)
    ap.add_argument("--no-principal-point", action="store_true",
                    help="keep cx,cy fixed at the crop centre, even where the fit "
                         "can measure them")
    ap.add_argument("--no-distortion", action="store_true",
                    help="keep k1 fixed at 0, even where the fit can measure it")
    ap.add_argument("--no-cross-validate", action="store_true",
                    help="skip leave-one-tag-out validation of each fit (faster, but "
                         "in-sample residual alone cannot tell a fit from an overfit)")
    args = ap.parse_args()

    stem = pathlib.Path(args.video).stem

    tags_path = (pathlib.Path(args.tags) if args.tags
                 else DETECTIONS_DIR / f"{stem}_tags.json")
    if not tags_path.exists():
        sys.exit(f"[error] tags not found: {tags_path}\n"
                 f"        run pipeline/01_detect_tags.py --video first")
    tags_data = json.loads(tags_path.read_text())

    intrinsics_path = (pathlib.Path(args.out_intrinsics) if args.out_intrinsics
                       else CALIB_DIR / f"{stem}_intrinsics.json")
    existing_intrinsics = (json.loads(intrinsics_path.read_text())
                           if intrinsics_path.exists() else {})
    intrinsics_dirty = False

    views = tags_data.get("views", {})
    if args.view:
        if args.view not in views:
            sys.exit(f"[error] view {args.view!r} not in tags file")
        views = {args.view: views[args.view]}

    # Which field the footage was shot on is detected from the geometry
    # rather than assumed -- see detect_field_layout(). Scored across every
    # view in the file, not just --view, since the best-covered view is the
    # one that can actually tell the layouts apart.
    field_year, field_tags, layout_confident = detect_field_layout(
        tags_data.get("views", {}), args.year)

    poses_out = {}
    rejected_views = []
    for vname, vdata in views.items():
        print(f"\n[{vname}]", file=sys.stderr)
        # Each static shot is its own camera: a cut can re-frame, re-zoom or
        # switch feed entirely, so intrinsics and pose are fit per shot
        # rather than across the cut. A view that never cuts yields exactly
        # one segment and behaves as before.
        segments = split_static_segments(vname, vdata)
        solved = []
        for seg in segments:
            label = (vname if seg["is_only"]
                     else f"{vname}@{seg['frames'][0]}-{seg['frames'][1]}")
            if not seg["is_only"]:
                print(f"  [{label}] shot {seg['index']}: {seg['n_frames']} sampled "
                      f"frame(s)", file=sys.stderr)
            sdata = seg["view_data"]
            K, d, cache_entry = _resolve_intrinsics(label, sdata, field_tags,
                                                    existing_intrinsics, args, field_year)
            if cache_entry is not None:
                # An undetermined season is a property of the whole file, not
                # of this view, but it has to travel with the entry -- a
                # stderr line does not survive into the JSON somebody reads a
                # week later, and "field_year" alone looks like a fact.
                if not layout_confident:
                    cache_entry["field_year_confident"] = False
                    cache_entry.setdefault("diagnostics", []).insert(0, {
                        "severity": "warning", "code": "field_layout_undetermined",
                        "detail": f"no field layout fits this footage; {field_year} "
                                  f"was chosen only as the best tag-ID coverage, so "
                                  f"every correspondence here may be against the "
                                  f"wrong field"})
                existing_intrinsics[label] = cache_entry
                intrinsics_dirty = True
            # A rejected fit is not usable intrinsics. Solving a pose from it
            # anyway is what produced this project's silently-wrong outputs
            # (cameras 3m below the floor, 59-degree pitches), so the shot is
            # left out of the poses file entirely -- the same thing that
            # already happens to a view with no usable tags.
            if cache_entry is not None and cache_entry.get("trusted") is False:
                print(f"  [{label}] REFUSED -- no pose written for this shot",
                      file=sys.stderr)
                continue
            # Whether freshly searched this run or loaded from a prior run's
            # cache, fold the intrinsics search's own diagnostics (e.g. a
            # poorly-constrained focal) into this view's pose diagnostics --
            # a pose solved from a bad focal is exactly what you want flagged
            # in one place, not split across two JSON files.
            idiag = existing_intrinsics.get(label, {}).get("diagnostics", [])
            result = solve_view(label, sdata, field_tags, K, d, args.per_frame)
            if result is None:
                continue
            result["diagnostics"] = idiag + result["diagnostics"]
            result["segment"] = {
                "index": seg["index"], "n_segments": len(segments),
                "frame_range": list(seg["frames"]) if seg["frames"] else None,
                "n_frames_sampled": seg["n_frames"], "label": label,
            }
            solved.append((label, result))

        if not solved:
            print(f"  [{vname}] REFUSED -- no shot in this view produced a "
                  f"trustworthy pose", file=sys.stderr)
            rejected_views.append(vname)
            continue

        # The view's headline pose is its DOMINANT shot -- the one covering
        # the most sampled frames -- not its best-fitting one. The short
        # shots at the head and tail of a broadcast are pre-match and
        # post-match filler; a pose solved from those is correct for frames
        # nobody wants to track, and picking it because it happened to show
        # more tags would silently point downstream at the wrong footage.
        # Tag count and residual only break ties between shots of similar
        # length. Everything downstream reads poses[view], so that key keeps
        # meaning "this view's pose"; the other shots ride along beside it.
        solved.sort(key=lambda lr: (-lr[1]["segment"]["n_frames_sampled"],
                                    -lr[1]["n_tags_used"],
                                    lr[1]["rms_reproj_all_px"]))
        best_label, best = solved[0]
        if len(solved) > 1:
            best["other_segments"] = [r for lbl, r in solved[1:]]
            print(f"  [{vname}] {len(solved)} shot(s) solved; reporting the dominant "
                  f"one ({best_label}, {best['segment']['n_frames_sampled']} frames, "
                  f"{best['n_tags_used']} tags, rms {best['rms_reproj_all_px']:.2f}px) "
                  f"as this view's pose", file=sys.stderr)
        poses_out[vname] = best
        # Mirror the chosen shot's intrinsics under the plain view name too,
        # so viz/* (which look up intrinsics by view) keep working unchanged.
        if best_label != vname and best_label in existing_intrinsics:
            existing_intrinsics[vname] = dict(existing_intrinsics[best_label],
                                              segment_label=best_label)
            intrinsics_dirty = True

    if intrinsics_dirty:
        intrinsics_path.parent.mkdir(parents=True, exist_ok=True)
        intrinsics_path.write_text(json.dumps(existing_intrinsics, indent=2))
        print(f"\n[out] {intrinsics_path}", file=sys.stderr)

    out_path = (pathlib.Path(args.out) if args.out
                else DETECTIONS_DIR / f"{stem}_poses.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # When --view is used, merge into the existing file so other views aren't
    # lost. Without --view we processed everything, so a clean write is fine.
    if args.view and out_path.exists():
        existing = json.loads(out_path.read_text())
        existing.update(poses_out)
        # A view rejected this run must not keep a pose an earlier run wrote
        # for it -- that would leave the exact stale wrong answer this gate
        # exists to remove.
        for v in rejected_views:
            existing.pop(v, None)
        poses_out = existing

    out_path.write_text(json.dumps(poses_out, indent=2))
    print(f"[out] {out_path}", file=sys.stderr)
    if rejected_views:
        print(f"[refused] {len(rejected_views)} view(s) had no trustworthy intrinsics "
              f"and were left out: {rejected_views}\n"
              f"          reasons are recorded under \"rejected_reasons\" in "
              f"{intrinsics_path.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
