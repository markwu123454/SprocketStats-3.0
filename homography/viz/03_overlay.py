#!/usr/bin/env python3
"""
Step 3 -- Overlay projected field geometry onto one grabbed frame per camera
view. This is the sanity check for the whole pipeline: if tag detection,
the field layout convention, solvePnP, and camera intrinsics are all
correct, the projected field should sit right on top of the real image
content, including tags that were never detected and the field boundary
itself (things the pose solve had no way to "cheat" toward, since it
never saw them).

Draws, per view, on a single grabbed frame:
  1. The field boundary rectangle + center line, projected from the
     solved camera pose. Needs intrinsics to land in the right place even
     if the pose itself is perfect -- see "Camera intrinsics" below.
  2. Every AprilTag in the field layout (data/field/<year>_tags.json),
     projected unconditionally -- whether or not pipeline/01 ever decoded
     it in this view:
       green  = this view's solved pose used this tag (in tag_residuals)
       orange = never used here, purely a 3-D projection
  3. A fresh AT3 detection run directly on the grabbed frame (single
     config, not pipeline/01's full ensemble -- this is a quick per-frame
     check, not a detection pass), drawn as cyan quads. This is ground
     truth for that exact frame, independent of the averaged-across-frames
     data the pose was solved from.

If the green/orange projected squares sit on top of the cyan freshly-
detected squares, and the boundary rectangle traces the real field edge,
the pipeline is self-consistent for that view.

Camera intrinsics
------------------
Yes, this needs them -- projecting a 3-D field point into 2-D pixels is
exactly what a camera matrix (K) is for. Using the WRONG K would still
produce a rectangle/tag grid that looks roughly tag-shaped, but shifted
and scaled off the real image by an amount that grows with distance from
the image center -- an easy way to mistake a bad intrinsics guess for a
bad pose. So this script loads intrinsics the identical way
pipeline/03_solve_pose.py did when it solved the pose being visualized
(same data/calibration/<stem>_intrinsics.json, same FOV/focal fallback) --
using a different K here would make this check meaningless.

Reads:
  data/field/<year>_tags.json          -- field AprilTag layout
  data/detections/<stem>_tags.json     -- view boxes (pipeline/01)
  data/detections/<stem>_poses.json    -- solved camera poses (pipeline/03)
  data/calibration/<stem>_intrinsics.json  -- if present (pipeline/02)
  <video>                              -- same video passed to pipeline/01

Writes:
  data/overlays/<stem>_<view>_overlay.jpg   -- one image per camera view

Usage:
  python viz/03_overlay.py --video match.mp4
  python viz/03_overlay.py --video match.mp4 --frame 1200
  python viz/03_overlay.py --video match.mp4 --fov-deg 80   # no intrinsics file

Install: pip install opencv-python numpy pupil-apriltags
"""

import argparse, json, math, pathlib, sys
import numpy as np
import cv2
from pupil_apriltags import Detector as AT3Detector

DATA_DIR       = pathlib.Path(__file__).parent.parent / "data"
FIELD_DIR      = DATA_DIR / "field"
DETECTIONS_DIR = DATA_DIR / "detections"
CALIB_DIR      = DATA_DIR / "calibration"
OVERLAY_DIR    = DATA_DIR / "overlays"

TAG_SIZE_M  = 0.1651  # 6.5 in outer size, see pipeline/01_fetch_field_layout.py
TAG_HALF_M  = TAG_SIZE_M / 2.0
DEFAULT_FOV = 70.0

COLOR_PROJECTED_UNSEEN = (0, 165, 255)   # orange (BGR) -- 3-D projection only
COLOR_PROJECTED_SEEN   = (0, 200, 0)     # green          -- used in this view's pose
COLOR_DETECTED_NOW     = (255, 255, 0)   # cyan           -- fresh detection this frame
COLOR_BOUNDARY         = (0, 80, 255)    # red-orange     -- field boundary


# ---------------------------------------------------------------------------
# Geometry helpers -- same conventions as pipeline/03_solve_pose.py
# ---------------------------------------------------------------------------

def _quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


def _tag_corners_field(tag: dict, half: float = TAG_HALF_M) -> np.ndarray:
    """
    (4, 3) corners of a tag in field coordinates, +Z up, in the same order
    AT3 returns for det.corners -- empirically BL,BR,TR,TL (confirmed
    against real detections: the "obvious" TL,TR,BR,BL reading put every
    corner ~20px off with a consistent top/bottom flip; see
    pipeline/03_solve_pose.py::tag_corners_field for the same fix).
    """
    local = np.array([[0, -half, -half], [0,  half, -half],
                      [0,  half,  half], [0, -half,  half]])
    R = _quat_to_rot(tag["qw"], tag["qx"], tag["qy"], tag["qz"])
    t = np.array([tag["x"], tag["y"], tag["z"]])
    return (R @ local.T).T + t


# A weakly-conditioned intrinsics guess can extrapolate wildly for points
# far outside the region it was fit on, producing huge or non-finite pixel
# coordinates. Reject anything past this many crop-widths from the image
# as unusable rather than let it reach cv2.polylines/putText, whose C++
# bindings hard-crash on out-of-int32-range coordinates.
_MAX_PROJECTION_MULTIPLE = 20


def _project(points_field: np.ndarray, K: np.ndarray, R_wc: np.ndarray,
            t_wc: np.ndarray, dist: np.ndarray, image_size: tuple[int, int]):
    """
    Project (N,3) field points into pixel space. Returns pixels (N,2), or
    None if any point is behind the camera (z<=0) or projects to a
    non-finite / wildly out-of-frame coordinate.
    """
    p_cam = (R_wc @ points_field.T).T + t_wc
    if np.any(p_cam[:, 2] <= 1e-6):
        return None
    rvec, _ = cv2.Rodrigues(R_wc)
    uv, _ = cv2.projectPoints(points_field, rvec, t_wc, K, dist)
    uv = uv.reshape(-1, 2)
    if not np.all(np.isfinite(uv)):
        return None
    w, h = image_size
    if np.any(np.abs(uv) > _MAX_PROJECTION_MULTIPLE * max(w, h)):
        return None
    return uv


_NEAR_EPS = 0.05   # metres in front of the camera; segments are clipped to this


def _project_segment_clipped(p0: np.ndarray, p1: np.ndarray, K: np.ndarray,
                             R_wc: np.ndarray, t_wc: np.ndarray, dist: np.ndarray,
                             image_size: tuple[int, int]):
    """
    Project one field-frame segment (p0, p1), clipping to the camera's near
    plane first if one endpoint is behind the camera -- unlike _project(),
    a segment with only one endpoint behind the camera still draws its
    visible portion instead of being dropped whole. This matters close up:
    a camera mounted right at a field corner (see viz/03_overlay.py's own
    docstring on why the field boundary needs this) can have that exact
    corner sit a few centimetres behind its own near plane while the rest
    of the boundary is perfectly valid.

    Returns a clipped (p0, p1) pair in field-frame, or None if the whole
    segment is behind the camera.
    """
    z0 = (R_wc @ p0 + t_wc)[2]
    z1 = (R_wc @ p1 + t_wc)[2]
    if z0 <= _NEAR_EPS and z1 <= _NEAR_EPS:
        return None
    if z0 <= _NEAR_EPS or z1 <= _NEAR_EPS:
        # Linear interpolate in field space to the point whose camera-frame
        # z lands exactly at the near plane (z depends affinely on the
        # field-frame point, so this is exact, not an approximation).
        t = (_NEAR_EPS - z0) / (z1 - z0)
        p_split = p0 + t * (p1 - p0)
        if z0 <= _NEAR_EPS:
            p0 = p_split
        else:
            p1 = p_split
    uv = _project(np.array([p0, p1]), K, R_wc, t_wc, dist, image_size)
    return None if uv is None else (uv[0], uv[1])


# ---------------------------------------------------------------------------
# Intrinsics -- identical logic to pipeline/03_solve_pose.py so the overlay
# uses the exact K/dist the pose was solved with (see module docstring).
# ---------------------------------------------------------------------------

def load_or_estimate_K(intrinsics_path: pathlib.Path, view_name: str,
                       view_box: list, fov_deg: float, focal_px: float | None):
    if intrinsics_path.exists():
        intr = json.loads(intrinsics_path.read_text())
        if view_name in intr:
            K = np.array(intr[view_name]["K"], dtype=np.float64)
            d = np.array(intr[view_name]["dist"], dtype=np.float64)
            print(f"  [intrinsics] loaded from {intrinsics_path.name}  "
                  f"f={K[0,0]:.1f}px", file=sys.stderr)
            return K, d
    x0, y0, x1, y1 = view_box
    w, h = x1 - x0, y1 - y0
    f = focal_px if focal_px is not None else (w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)
    d = np.zeros(5, dtype=np.float64)
    print(f"  [intrinsics] estimated  f={f:.1f}px  fov={fov_deg:.1f}deg "
          f"(no intrinsics file -- projection may be off)", file=sys.stderr)
    return K, d


# ---------------------------------------------------------------------------
# Fresh single-frame detection (ground truth for this exact frame)
# ---------------------------------------------------------------------------

def _make_detector() -> AT3Detector:
    # Single config, not pipeline/01's ensemble -- this is a quick per-frame
    # check, not a detection pass. See pipeline/01_detect_tags.py for why
    # these params.
    return AT3Detector(families="tag36h11", nthreads=4, quad_decimate=1.0,
                       quad_sigma=0.0, refine_edges=1, decode_sharpening=1.25)


def _detect_in_crop(crop: np.ndarray, detector: AT3Detector) -> dict:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return {int(d.tag_id): d.corners.astype(np.float64) for d in detector.detect(gray)}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_quad(img, pixels: np.ndarray, color, thickness, label=None):
    pts = pixels.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness,
                  lineType=cv2.LINE_AA)
    if label:
        cx, cy = pixels.mean(axis=0)
        cv2.putText(img, label, (int(cx) - 6, int(cy) + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_clipped_line(img, p0, p1, K, R_wc, t_wc, dist, image_size, color, thickness):
    clipped = _project_segment_clipped(p0, p1, K, R_wc, t_wc, dist, image_size)
    if clipped is None:
        return
    a, b = clipped
    cv2.line(img, tuple(a.astype(np.int32)), tuple(b.astype(np.int32)),
             color, thickness, cv2.LINE_AA)


def _draw_boundary(img, K, R_wc, t_wc, dist, image_size, fl, fw):
    corners = [np.array(c) for c in
              [[0, 0, 0], [fl, 0, 0], [fl, fw, 0], [0, fw, 0]]]
    # Each edge clipped/drawn independently -- a camera mounted right at one
    # field corner (see _project_segment_clipped's docstring) can have just
    # that corner behind its near plane while the rest of the boundary is
    # perfectly valid; whole-polygon projection would drop the entire
    # boundary over one bad vertex.
    for i in range(4):
        _draw_clipped_line(img, corners[i], corners[(i + 1) % 4],
                           K, R_wc, t_wc, dist, image_size, COLOR_BOUNDARY, 2)
    _draw_clipped_line(img, np.array([fl / 2, 0, 0]), np.array([fl / 2, fw, 0]),
                       K, R_wc, t_wc, dist, image_size, COLOR_BOUNDARY, 1)


def _draw_legend(img):
    entries = [
        ("field boundary (projected)",       COLOR_BOUNDARY),
        ("field tag, used in this pose",     COLOR_PROJECTED_SEEN),
        ("field tag, projection only",       COLOR_PROJECTED_UNSEEN),
        ("fresh detection, this frame",       COLOR_DETECTED_NOW),
    ]
    y = 18
    for text, color in entries:
        cv2.rectangle(img, (8, y - 10), (22, y + 2), color, -1)
        cv2.putText(img, text, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                   (240, 240, 240), 1, cv2.LINE_AA)
        y += 18


# ---------------------------------------------------------------------------
# Per-view overlay
# ---------------------------------------------------------------------------

def build_overlay(crop: np.ndarray, view_name: str, pose: dict,
                  all_tags: dict, K: np.ndarray, dist: np.ndarray,
                  detector: AT3Detector, fl: float, fw: float) -> np.ndarray:
    image_size = (crop.shape[1], crop.shape[0])
    R_wc, _ = cv2.Rodrigues(np.array(pose["rvec"], dtype=np.float64))
    t_wc = np.array(pose["tvec_m"], dtype=np.float64)
    used_ids = set(pose.get("tag_residuals", {}))

    out = crop.copy()

    _draw_boundary(out, K, R_wc, t_wc, dist, image_size, fl, fw)

    # --- unconditionally project every field tag ---
    for tid, tag_field in sorted(all_tags.items(), key=lambda kv: int(kv[0])):
        corners = _tag_corners_field(tag_field)
        uv = _project(corners, K, R_wc, t_wc, dist, image_size)
        if uv is None:
            continue  # behind the camera, or too far outside the frame
        seen = tid in used_ids
        color, thickness = (COLOR_PROJECTED_SEEN, 2) if seen else (COLOR_PROJECTED_UNSEEN, 1)
        _draw_quad(out, uv, color, thickness, label=tid)

    # --- fresh detection on this exact frame, for ground-truth comparison ---
    fresh = _detect_in_crop(crop, detector)
    for tid_int, corners_px in fresh.items():
        _draw_quad(out, corners_px, COLOR_DETECTED_NOW, 1)

    # --- numeric reprojection error where we have both ---
    for tid_int, corners_px in sorted(fresh.items()):
        tid = str(tid_int)
        tag_field = all_tags.get(tid)
        if tag_field is None:
            continue
        uv = _project(_tag_corners_field(tag_field), K, R_wc, t_wc, dist, image_size)
        if uv is None:
            continue
        err = float(np.mean(np.linalg.norm(uv - corners_px, axis=1)))
        print(f"  [{view_name}] tag {tid}: reprojection error = {err:.1f} px "
              f"(projected vs freshly detected, this frame)", file=sys.stderr)

    _draw_legend(out)
    cv2.putText(out, view_name, (8, out.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, metavar="PATH")
    ap.add_argument("--year", type=int, default=2026,
                    choices=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--view", metavar="NAME",
                    help="process only this view (default: all views with a solved pose)")
    ap.add_argument("--frame", type=int, metavar="N",
                    help="video frame index to grab (default: middle of the video)")
    ap.add_argument("--tags",  metavar="PATH",
                    help="tags JSON (default: data/detections/<stem>_tags.json)")
    ap.add_argument("--poses", metavar="PATH",
                    help="poses JSON (default: data/detections/<stem>_poses.json)")
    ap.add_argument("--fov-deg",  type=float, default=DEFAULT_FOV,
                    help="fallback horizontal FOV if no intrinsics file found "
                         "(default: %(default)s, must match what "
                         "03_solve_pose.py used)")
    ap.add_argument("--focal-px", type=float, default=None,
                    help="fallback focal length in pixels (overrides --fov-deg)")
    ap.add_argument("--out-dir", metavar="DIR",
                    help=f"output directory (default: {OVERLAY_DIR})")
    args = ap.parse_args()

    field_path = FIELD_DIR / f"{args.year}_tags.json"
    if not field_path.exists():
        sys.exit(f"[error] field layout not found: {field_path}\n"
                 f"        run: python pipeline/01_fetch_field_layout.py --year {args.year}")
    field_data = json.loads(field_path.read_text())
    all_tags = field_data["tags"]
    fl = field_data.get("field_length_m", 16.541)
    fw = field_data.get("field_width_m", 8.069)

    stem = pathlib.Path(args.video).stem
    tags_path = pathlib.Path(args.tags) if args.tags else DETECTIONS_DIR / f"{stem}_tags.json"
    if not tags_path.exists():
        sys.exit(f"[error] tags not found: {tags_path}\n"
                 f"        run pipeline/01_detect_tags.py --video first")
    tags_data = json.loads(tags_path.read_text())

    poses_path = pathlib.Path(args.poses) if args.poses else DETECTIONS_DIR / f"{stem}_poses.json"
    if not poses_path.exists():
        sys.exit(f"[error] poses not found: {poses_path}\n"
                 f"        run pipeline/03_solve_pose.py --video first")
    poses = json.loads(poses_path.read_text())
    if not poses:
        sys.exit(f"[error] {poses_path} has no solved camera poses")

    intrinsics_path = CALIB_DIR / f"{stem}_intrinsics.json"

    views = tags_data.get("views", {})
    view_names = [args.view] if args.view else sorted(poses)
    missing = [v for v in view_names if v not in poses]
    if missing:
        print(f"[warn] no solved pose for view(s) {missing} -- skipping", file=sys.stderr)
    view_names = [v for v in view_names if v in poses and v in views]
    if not view_names:
        sys.exit("[error] no view has both a box (tags JSON) and a solved pose")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[error] cannot open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = args.frame if args.frame is not None else total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"[error] could not read frame {frame_idx} from {args.video}")
    print(f"[video] using frame {frame_idx}/{total}", file=sys.stderr)

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else OVERLAY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = _make_detector()

    written = 0
    for view_name in view_names:
        print(f"[{view_name}]", file=sys.stderr)
        x0, y0, x1, y1 = views[view_name]["box"]
        crop = frame[y0:y1, x0:x1]
        K, dist = load_or_estimate_K(intrinsics_path, view_name, views[view_name]["box"],
                                     args.fov_deg, args.focal_px)
        overlay = build_overlay(crop, view_name, poses[view_name], all_tags,
                                K, dist, detector, fl, fw)
        out_path = out_dir / f"{stem}_{view_name}_overlay.jpg"
        cv2.imwrite(str(out_path), overlay)
        print(f"  [out] {out_path}", file=sys.stderr)
        written += 1

    if written == 0:
        sys.exit("[error] no overlays could be produced")


if __name__ == "__main__":
    main()
