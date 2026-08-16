#!/usr/bin/env python3
"""
Step 3 -- Solve camera pose (position + orientation) relative to the FRC field.

Per-tag hybrid: corners for head-on tags, centroid otherwise.
----------------------------------------------------------------
At broadcast resolution (15–50 px tags) AT3 corner positions jitter ±2–3 px per
frame from H.264 block-boundary artifacts. For a 30 px tag that translates to
5–10° of corner-angle error -- but that error comes from FORESHORTENING
amplifying a few px of pixel noise into a large angular one, so it's worst
for tags seen at an oblique angle and much smaller for tags seen close to
head-on. The centroid of 4 corners suppresses this jitter regardless of
angle, at the cost of throwing away 3 of a tag's 4 points.

So: solve_view() computes each tag's "squareness" (_quad_squareness --
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
tag that has corner data, regardless of squareness. This is the same
low-tag-count situation earlier revisions of this module called
"point_mode: corners"; a view with only 1-2 tags still solves this way
(2 tags -> 8 points spread across two field locations, not degenerate the
way 2 bare centers would be). A single tag falls back to exactly its 4
corners, the minimum solvePnP needs; that's a perfectly planar point set
so it carries a real 2-fold pose ambiguity at near-fronto-parallel viewing
angles -- treat a 1-tag pose as low-confidence. Output records what ran
as "point_mode": "hybrid (N corner tag(s) + M center tag(s))".

Apparent-size consistency check
--------------------------------
After solving, the expected apparent tag size (pixels) at each tag's solved
distance is compared against the observed mean_size_px. This is a sanity check
on the focal length used.

  expected_size_px = TAG_SIZE_M * K[0,0] / distance_m

If expected / observed deviates by more than 20% the focal length assumption is
probably wrong (pass --focal-px or --fov-deg to correct it, or re-run
pipeline/02_search_focal.py -- see docs/pose_calibration_research.md).

Aggregated vs per-frame
-----------------------
Primary output uses mean_center_px (aggregated across all frames) for each tag,
giving the most stable estimate. Pass --per-frame to also solve each sampled
frame independently and report temporal spread.

Outputs
-------
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
        "tag_residuals": {
          "<tag_id>": {
            "reproj_px": ..., "n_points": ..., "dist_m": ...,
            "expected_size_px": ..., "observed_size_px": ...
          }
        },
        "per_frame": [...]             // if --per-frame
      }
    }

Usage
-----
  python pipeline/03_solve_pose.py --video match.mp4
  python pipeline/03_solve_pose.py --video match.mp4 --per-frame
  python pipeline/03_solve_pose.py --video match.mp4 --fov-deg 80
  python pipeline/03_solve_pose.py --video match.mp4 --focal-px 1100

Install: pip install opencv-python numpy
"""

import argparse, json, math, pathlib, sys
import numpy as np
import cv2

DATA_DIR       = pathlib.Path(__file__).parent.parent / "data"
DETECTIONS_DIR = DATA_DIR / "detections"
CALIB_DIR      = DATA_DIR / "calibration"
FIELD_DIR      = DATA_DIR / "field"

TAG_SIZE_M   = 0.1651
FIELD_YEAR   = 2026
DEFAULT_FOV  = 70.0
MAX_REPROJ   = 30.0   # RANSAC inlier threshold (px)

# Floor on total correspondence points (not tags -- a corner-mode tag
# contributes 4) below which solvePnP doesn't have enough to work with.
MIN_TAGS_FOR_CENTERS = 4

# Below this fraction of sampled frames, a tag's mean position is too few
# raw observations to have suppressed any real pixel noise -- see
# solve_view()'s comment for the 105px -> 1.8px case that motivated this.
MIN_FRAME_FRACTION = 0.1

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
# Core solver
# ---------------------------------------------------------------------------

def solve_view(view_name: str, view_data: dict, field_tags: dict,
               K: np.ndarray, dist: np.ndarray,
               do_per_frame: bool) -> dict | None:
    decoded = view_data.get("decoded_tags", {})
    if not decoded:
        print(f"  [{view_name}] no decoded tags — skipping", file=sys.stderr)
        return None

    # Collect usable tags -- center/size are kept regardless of point_mode
    # below, since the size-consistency check and residual reporting always
    # need them. A tag detected in only 1-2 of the sampled frames doesn't
    # get the jitter-suppression mean_center_px/mean_corners exist for --
    # its "mean" is just that raw, noisy detection, full pixel error and
    # all (confirmed expensively: one view's fit went from 105px rms to
    # 1.8px after dropping two tags seen in 1-2/150 frames -- see git
    # history). Require at least MIN_FRAME_FRACTION of the sampled frames.
    n_sampled = view_data.get("n_frames_sampled", 1)
    min_frames = max(1, MIN_FRAME_FRACTION * n_sampled)
    usable, sparse = [], []
    for tid_str, tag_data in decoded.items():
        tid = int(tid_str)
        ft = field_tags.get(tid_str) or field_tags.get(str(tid))
        if ft is None:
            print(f"  [{view_name}] tag {tid} not in field layout — skipping",
                  file=sys.stderr)
            continue
        if tag_data.get("n_frames_detected", 0) < min_frames:
            sparse.append(tid)
            continue
        usable.append({
            "tid":        tid,
            "field_tag":  ft,
            "center_px":  tag_data["mean_center_px"],
            "size_px":    tag_data.get("mean_size_px", 0.0),
            "corners_px": tag_data.get("mean_corners"),
        })
    if sparse:
        print(f"  [{view_name}] tag(s) {sparse} detected in too few frames "
              f"(< {min_frames:.0f}/{n_sampled}) — dropping, not enough "
              f"observations to average the noise out", file=sys.stderr)

    if not usable:
        print(f"  [{view_name}] no usable tags — skipping", file=sys.stderr)
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
    # "Head-on enough" is measured directly off the detected corner quad
    # (see _quad_squareness): a true rectangle has equal opposite edges
    # and equal diagonals, so the more a tag's perspective foreshortens it,
    # the further both ratios drop from 1.0. Empirically on this project's
    # own broadcast footage, tags land anywhere from ~0.55 (steep angle)
    # to ~0.90 (closest to head-on this footage has); nothing reaches a
    # "true" 1.0 since even the best view has some obliqueness + AT3's own
    # corner-refinement noise.
    for u in usable:
        u["squareness"] = (_quad_squareness(u["corners_px"])
                           if u["corners_px"] and len(u["corners_px"]) == 4 else 0.0)
        u["use_corners"] = u["squareness"] >= HEAD_ON_SQUARENESS

    n_points = sum(4 if u["use_corners"] else 1 for u in usable)
    if n_points < MIN_TAGS_FOR_CENTERS and any(u["corners_px"] for u in usable):
        # Not enough points even to reach centers-only's own floor -- same
        # low-tag-count situation the corner fallback was originally built
        # for (see git history), just reframed: relax the head-on bar to
        # "has corners at all" rather than drop to centers-only, since more
        # (noisier) points beat too few points to solve at all. A tag with
        # NO corner data (mean_corners missing) still can't contribute more
        # than its centroid regardless.
        for u in usable:
            if u["corners_px"] and len(u["corners_px"]) == 4:
                u["use_corners"] = True
        n_points = sum(4 if u["use_corners"] else 1 for u in usable)
        print(f"  [{view_name}] only {n_points} point(s) from head-on tags alone "
              f"— relaxing to all available corners", file=sys.stderr)

    n_corner_tags = sum(1 for u in usable if u["use_corners"])
    point_mode = (f"hybrid ({n_corner_tags} corner tag(s) + "
                  f"{len(usable) - n_corner_tags} center tag(s))")
    if n_corner_tags:
        print(f"  [{view_name}] using 4-corner points for head-on tag(s) "
              f"{[u['tid'] for u in usable if u['use_corners']]}, "
              f"centroid for the rest ({n_points} points total from "
              f"{len(usable)} tags)", file=sys.stderr)

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
        print(f"  [{view_name}] only {len(obj_pts)} point(s) total "
              f"(need >= 4 for solvePnP) — skipping", file=sys.stderr)
        return None
    if len(usable) == 1:
        print(f"  [{view_name}] single-tag pose — coplanar point set, "
              f"real ambiguity risk, treat as low-confidence", file=sys.stderr)

    obj_arr = np.array(obj_pts, dtype=np.float64).reshape(-1, 1, 3)
    img_arr = np.array(img_pts, dtype=np.float64).reshape(-1, 1, 2)

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
        print(f"  [{view_name}] RANSAC failed — falling back to SQPNP on all {len(obj_pts)} pts",
              file=sys.stderr)
        ok, rvec, tvec = cv2.solvePnP(
            obj_arr, img_arr, K, dist,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok:
            print(f"  [{view_name}] solvePnP also failed — skipping", file=sys.stderr)
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

    # Size consistency check
    for tid_str, res in tag_residuals.items():
        exp = res["expected_size_px"]
        obs = res["observed_size_px"]
        if obs > 0 and exp > 0:
            ratio = exp / obs
            if ratio < 0.7 or ratio > 1.4:
                print(f"  [warn] tag {tid_str} size mismatch: "
                      f"expected {exp:.1f}px observed {obs:.1f}px (ratio={ratio:.2f}) "
                      f"-- focal length may be off", file=sys.stderr)

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
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video",     required=True, metavar="PATH")
    ap.add_argument("--view",      metavar="NAME",
                    help="process only this view (default: all views)")
    ap.add_argument("--year",      type=int, default=FIELD_YEAR,
                    help="field year for tag layout (default: %(default)s)")
    ap.add_argument("--fov-deg",   type=float, default=DEFAULT_FOV,
                    help="fallback horizontal FOV if no intrinsics file found "
                         "(default: %(default)s)")
    ap.add_argument("--focal-px",  type=float, default=None,
                    help="fallback focal length in pixels (overrides --fov-deg)")
    ap.add_argument("--per-frame", action="store_true",
                    help="also solve pose per frame and include in output")
    ap.add_argument("--tags",      metavar="PATH",
                    help="path to tags JSON (default: data/detections/<stem>_tags.json)")
    ap.add_argument("--out",       metavar="PATH",
                    help="output path (default: data/detections/<stem>_poses.json)")
    args = ap.parse_args()

    stem = pathlib.Path(args.video).stem

    tags_path = (pathlib.Path(args.tags) if args.tags
                 else DETECTIONS_DIR / f"{stem}_tags.json")
    if not tags_path.exists():
        sys.exit(f"[error] tags not found: {tags_path}\n"
                 f"        run pipeline/01_detect_tags.py --video first")
    tags_data = json.loads(tags_path.read_text())

    field_path = FIELD_DIR / f"{args.year}_tags.json"
    if not field_path.exists():
        sys.exit(f"[error] field tag layout not found: {field_path}")
    raw_field = json.loads(field_path.read_text())
    # Normalise to {str(id): {x,y,z,qw,qx,qy,qz}}
    if isinstance(raw_field, list):
        field_tags = {str(t["id"]): t for t in raw_field}
    elif isinstance(raw_field, dict) and "tags" in raw_field:
        tags_val = raw_field["tags"]
        if isinstance(tags_val, list):
            field_tags = {str(t["id"]): t for t in tags_val}
        else:
            field_tags = {str(k): v for k, v in tags_val.items()}
    else:
        field_tags = {str(k): v for k, v in raw_field.items()}

    intrinsics_path = CALIB_DIR / f"{stem}_intrinsics.json"

    views = tags_data.get("views", {})
    if args.view:
        if args.view not in views:
            sys.exit(f"[error] view {args.view!r} not in tags file")
        views = {args.view: views[args.view]}

    poses_out = {}
    for vname, vdata in views.items():
        print(f"\n[{vname}]", file=sys.stderr)
        K, d = load_or_estimate_K(intrinsics_path, vname, vdata["box"],
                                  args.fov_deg, args.focal_px)
        result = solve_view(vname, vdata, field_tags, K, d, args.per_frame)
        if result is not None:
            poses_out[vname] = result

    out_path = (pathlib.Path(args.out) if args.out
                else DETECTIONS_DIR / f"{stem}_poses.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # When --view is used, merge into the existing file so other views aren't
    # lost. Without --view we processed everything, so a clean write is fine.
    if args.view and out_path.exists():
        existing = json.loads(out_path.read_text())
        existing.update(poses_out)
        poses_out = existing

    out_path.write_text(json.dumps(poses_out, indent=2))
    print(f"\n[out] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
