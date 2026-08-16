#!/usr/bin/env python3
"""
Step 1 — Field layout + per-camera AprilTag 3-D localisation.

Field layout
------------
Loads the bundled WPILib AprilTag field layout for the requested year from
data/field/<year>_layout.json and writes a flattened version:
  data/field/<year>_tags.json
    { "tags": { "1": {x, y, z, qw, qx, qy, qz}, ... }, ... }

Tag detection across frames
---------------------------
Given a video and the camera-view profile written by pipeline/00_split_views.py,
samples --n-frames frames evenly from the middle of the match, runs AprilTag
detection on every camera-view crop, and accumulates unique tag IDs across
all frames.  The best detection per tag (largest apparent pixel size = most
frontal / least foreshortened) is used to estimate the tag's 3-D pose
relative to that camera via solvePnP.

Only camera views that contain at least one decoded tag are included in the
output.

Tag physical size
-----------------
FRC has used 36h11 tags at 6.5 in (0.1651 m) outer size since 2023.
The same size is assumed for 2022 and 2026 pending game-manual confirmation.

Camera intrinsics
-----------------
Broadcast cameras are not calibrated.  Without an --intrinsics file a
pinhole estimate is derived from the crop dimensions and an assumed
horizontal FOV of 70°.  This is sufficient for rough distance estimates;
supply --intrinsics K.json for accurate poses. --intrinsics accepts
either a single-camera file ({"K":..., "dist":...}, applied to every
view) or a per-view file ({"main": {"K":..., "dist":...},
"bot_left": {...}, ...}) — different camera views are different physical
lenses and generally should NOT share one K.
pipeline/02_search_focal.py writes this per-view format, fit from
AprilTag correspondences via a validated 1-D search (see
docs/pose_calibration_research.md for the approaches that came before it
and why they didn't hold up).

Outputs
-------
  data/field/<year>_tags.json         — flattened tag layout (always)
  data/detections/<stem>_poses.json   — per-view tag poses (with --video)

Usage
-----
  python pipeline/01_fetch_field_layout.py --year 2026
  python pipeline/01_fetch_field_layout.py --video match.mp4 --year 2026
  python pipeline/01_fetch_field_layout.py --video match.mp4 --year 2026 --n-frames 80
"""

import argparse, json, math, pathlib, sys
import numpy as np
import cv2
from pupil_apriltags import Detector as AT3Detector

DATA_DIR       = pathlib.Path(__file__).parent.parent / "data"
FIELD_DIR      = DATA_DIR / "field"
PROF_DIR       = DATA_DIR / "profiles"
DETECTIONS_DIR = DATA_DIR / "detections"

DEFAULT_YEAR    = 2026
AVAILABLE_YEARS = [2022, 2023, 2024, 2025, 2026]

# Outer tag size in metres (6.5 in), confirmed by 2023-2025 FRC game manuals.
# Assumed constant for 2022 and 2026.
TAG_SIZE_M = 0.1651

# Assumed broadcast-camera horizontal FOV when no intrinsics file is provided.
ASSUMED_FOV_DEG = 70.0

# Sample frames from the middle portion of the video (avoids intro/outro CG).
SAMPLE_START = 0.25
SAMPLE_END   = 0.75


# ---------------------------------------------------------------------------
# Field layout
# ---------------------------------------------------------------------------

def flatten_tags(wpilib_json: dict) -> dict:
    tags = {}
    for entry in wpilib_json.get("field-tags", []):
        tid   = str(entry["ID"])
        trans = entry["pose"]["translation"]
        quat  = entry["pose"]["rotation"]["quaternion"]
        tags[tid] = {
            "x":  trans["x"],
            "y":  trans["y"],
            "z":  trans["z"],
            "qw": quat["W"],
            "qx": quat["X"],
            "qy": quat["Y"],
            "qz": quat["Z"],
        }
    dims = wpilib_json.get("field-dimensions", {})
    return {
        "year":           wpilib_json.get("season"),
        "game":           wpilib_json.get("game"),
        "field_length_m": dims.get("length"),
        "field_width_m":  dims.get("width"),
        "tags":           tags,
    }


# ---------------------------------------------------------------------------
# Video frame sampling
# ---------------------------------------------------------------------------

def sample_frames(video_path: str, n: int) -> list[np.ndarray]:
    """
    Return n evenly-spaced BGR frames from the middle portion of the video.
    Skips unreadable frames silently.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[error] cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo    = max(0, int(total * SAMPLE_START))
    hi    = min(total - 1, int(total * SAMPLE_END))
    step  = max(1, (hi - lo) // max(1, n - 1))
    indices = list(range(lo, hi + 1, step))[:n]
    print(f"[video] {total} total frames; sampling {len(indices)} "
          f"[{indices[0]}..{indices[-1]}]", file=sys.stderr)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
        else:
            print(f"[warn] could not read frame {idx}", file=sys.stderr)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# AprilTag detection
# ---------------------------------------------------------------------------

def _make_detector() -> AT3Detector:
    # Same tuned parameters as pipeline/01_detect_tags.py: no pre-blur
    # (AT3's gradient approach doesn't need it), aggressive sharpening to
    # recover H.264-compressed tags, no subsampling for small/distant tags.
    return AT3Detector(
        families          = "tag36h11",
        nthreads          = 4,
        quad_decimate     = 1.0,
        quad_sigma        = 0.0,
        refine_edges      = 1,
        decode_sharpening = 1.25,
    )


def _detect_in_crop(crop: np.ndarray,
                    detector: AT3Detector) -> list[dict]:
    """Return decoded tags: [{id, corners (4,2), size_px, center_px}].

    AT3 corners come back in BL,BR,TR,TL order (empirically confirmed in
    pipeline/03_solve_pose.py::tag_corners_field). _tag_obj_pts() uses the
    same order so solvePnP's point correspondence is correct.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    results = []
    for d in detector.detect(gray):
        c     = d.corners.astype(np.float64)   # (4, 2), order: BL,BR,TR,TL
        edges = [float(np.linalg.norm(c[(i+1)%4] - c[i])) for i in range(4)]
        size  = float(np.mean(edges))
        cx, cy = c.mean(axis=0)
        results.append({
            "id":        int(d.tag_id),
            "corners":   c,
            "size_px":   size,
            "center_px": [round(float(cx), 1), round(float(cy), 1)],
        })
    return results


# ---------------------------------------------------------------------------
# 3-D pose via solvePnP
# ---------------------------------------------------------------------------

def _estimate_K(w: int, h: int, fov_deg: float = ASSUMED_FOV_DEG) -> np.ndarray:
    f = w / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


# Tag corners in the tag's own coordinate frame (z=0, origin at centre).
# Order matches AT3's det.corners output: BL, BR, TR, TL.
# (AT3 uses a different winding than ArUco's TL,TR,BR,BL — see the
# corner-order note in pipeline/03_solve_pose.py::tag_corners_field.)
def _tag_obj_pts(tag_size_m: float) -> np.ndarray:
    h = tag_size_m / 2.0
    return np.array([
        [-h, -h, 0],  # BL
        [ h, -h, 0],  # BR
        [ h,  h, 0],  # TR
        [-h,  h, 0],  # TL
    ], dtype=np.float64)


def solve_pose(corners_px: np.ndarray, tag_size_m: float,
               K: np.ndarray, dist: np.ndarray) -> dict | None:
    """
    Run solvePnP and return pose dict, or None if it fails.
    Uses IPPE_SQUARE which is optimal for square planar targets.
    """
    obj_pts = _tag_obj_pts(tag_size_m)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, corners_px.astype(np.float64), K, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None
    dist_m = float(np.linalg.norm(tvec))
    return {
        "rvec":   [round(float(v), 6) for v in rvec.flatten()],
        "tvec_m": [round(float(v), 4) for v in tvec.flatten()],
        "dist_m": round(dist_m, 3),
    }


# ---------------------------------------------------------------------------
# Per-view accumulation
# ---------------------------------------------------------------------------

def accumulate_views(frames: list[np.ndarray],
                     layout: dict,
                     tag_size_m: float,
                     intrinsics: dict | None) -> dict:
    """
    For each camera view, detect tags across all frames, accumulate unique
    IDs, keep the best detection (largest size_px) per tag, run solvePnP.

    `intrinsics`, if given, is keyed by view name (e.g. {"main": {"K":...,
    "dist":...}, "bot_left": {...}}), with an optional "*" entry applied to
    any view not otherwise listed — see `_load_intrinsics`. Different camera
    views are genuinely different physical lenses/zoom, so intrinsics never
    apply uniformly across views except via that explicit "*" fallback.

    Returns only views that have at least one decoded tag.
    """
    # De-duplicate views sharing the same box (e.g. the synthetic "main" alias).
    seen_boxes, unique_views = set(), []
    for v in layout.get("views", []):
        key = tuple(v["box"])
        if key not in seen_boxes:
            seen_boxes.add(key)
            unique_views.append(v)

    detector = _make_detector()

    results = {}

    for view in unique_views:
        name         = view["name"]
        x0, y0, x1, y1 = view["box"]
        crop_w, crop_h  = x1 - x0, y1 - y0

        view_intr = intrinsics.get(name) or intrinsics.get("*") if intrinsics else None
        if view_intr:
            K    = np.array(view_intr["K"], dtype=np.float64)
            dist = np.array(view_intr.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64)
            intrinsics_estimated = False
        else:
            K    = _estimate_K(crop_w, crop_h)
            dist = np.zeros(5, dtype=np.float64)
            intrinsics_estimated = True

        # best_per_tag[tag_id] = {size_px, corners, center_px, n_frames}
        best_per_tag: dict[int, dict] = {}

        for frame_bgr in frames:
            crop = frame_bgr[y0:y1, x0:x1]
            for det in _detect_in_crop(crop, detector):
                tid  = det["id"]
                prev = best_per_tag.get(tid)
                if prev is None:
                    best_per_tag[tid] = {**det, "n_frames": 1}
                else:
                    prev["n_frames"] += 1
                    if det["size_px"] > prev["size_px"]:
                        prev.update({k: det[k]
                                     for k in ("corners", "size_px", "center_px")})

        if not best_per_tag:
            print(f"[{name}] no tags detected", file=sys.stderr)
            continue

        tags_out = {}
        for tid, best in sorted(best_per_tag.items()):
            pose = solve_pose(best["corners"], tag_size_m, K, dist)
            entry = {
                "n_frames_detected": best["n_frames"],
                "size_px":           round(best["size_px"], 1),
                "center_px":         best["center_px"],
            }
            if pose:
                entry.update(pose)
            tags_out[str(tid)] = entry

        n = len(tags_out)
        print(f"[{name}] {n} unique tag(s): {sorted(tags_out.keys())}",
              file=sys.stderr)

        results[name] = {
            "box":                  view["box"],
            "crop_w":               crop_w,
            "crop_h":               crop_h,
            "intrinsics_estimated": intrinsics_estimated,
            "fov_deg_assumed":      ASSUMED_FOV_DEG if intrinsics_estimated else None,
            "K":                    K.tolist(),
            "dist":                 dist.tolist(),
            "n_tags_found":         n,
            "tags":                 tags_out,
        }

    return results


def _load_intrinsics(path: str) -> dict:
    """
    Load an --intrinsics file and normalize it to {view_name: {K, dist}}.

    Two accepted shapes:
      - single-camera:  {"K": [[...]], "dist": [...]}
        -> applied to every view via the "*" fallback key.
      - per-view:        {"main": {"K":..., "dist":...}, "bot_left": {...}, ...}
        -> as produced by pipeline/02_search_focal.py. Views not listed fall
           back to the estimated pinhole (or to a "*" entry if one is present).
    """
    data = json.loads(pathlib.Path(path).read_text())
    if "K" in data:
        return {"*": data}
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=DEFAULT_YEAR,
                    choices=AVAILABLE_YEARS,
                    help=f"FRC season year (default: {DEFAULT_YEAR})")
    ap.add_argument("--video", metavar="PATH",
                    help="local video; requires pipeline/00_split_views.py profile in data/profiles/")
    ap.add_argument("--n-frames", type=int, default=60,
                    help="frames to sample for tag detection (default: 60)")
    ap.add_argument("--intrinsics", metavar="K.json",
                    help="camera intrinsics JSON {K: [[...]], dist: [...]}; "
                         "if omitted, estimated from crop size + assumed FOV")
    ap.add_argument("--tag-size", type=float, default=TAG_SIZE_M,
                    metavar="METRES",
                    help=f"tag outer size in metres (default: {TAG_SIZE_M})")
    args = ap.parse_args()

    # --- field layout (always) ---
    raw_path = FIELD_DIR / f"{args.year}_layout.json"
    if not raw_path.exists():
        sys.exit(f"[error] bundled layout not found: {raw_path}")
    wpilib_json = json.loads(raw_path.read_text())
    flat        = flatten_tags(wpilib_json)

    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    flat_path = FIELD_DIR / f"{args.year}_tags.json"
    flat_path.write_text(json.dumps(flat, indent=2))
    print(f"[out] {flat_path}", file=sys.stderr)
    print(f"[field] {flat['game']} {args.year} — {len(flat['tags'])} tags  "
          f"{flat['field_length_m']}m × {flat['field_width_m']}m",
          file=sys.stderr)

    if not args.video:
        return

    # --- load camera-view profile (from pipeline/00_split_views.py) ---
    stem         = pathlib.Path(args.video).stem
    layout_path  = PROF_DIR / f"{stem}_layout.json"
    if not layout_path.exists():
        sys.exit(f"[error] view profile not found: {layout_path}\n"
                 f"        run pipeline/00_split_views.py --video first")
    layout = json.loads(layout_path.read_text())
    print(f"[layout] {layout.get('layout')}  "
          f"views={[v['name'] for v in layout.get('views', [])]}",
          file=sys.stderr)

    # --- intrinsics (optional) ---
    intrinsics = None
    if args.intrinsics:
        intrinsics = _load_intrinsics(args.intrinsics)

    # --- sample frames ---
    frames = sample_frames(args.video, args.n_frames)
    if not frames:
        sys.exit("[error] no frames could be read from video")
    print(f"[video] loaded {len(frames)} frames", file=sys.stderr)

    # --- detect and localise ---
    views = accumulate_views(frames, layout,
                             tag_size_m=args.tag_size,
                             intrinsics=intrinsics)

    if not views:
        print("[warn] no tags detected in any camera view", file=sys.stderr)

    # --- write output ---
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "video":            args.video,
        "year":             args.year,
        "game":             flat["game"],
        "n_frames_sampled": len(frames),
        "tag_size_m":       args.tag_size,
        "camera_views":     views,
    }
    out_path = DETECTIONS_DIR / f"{stem}_poses.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[out] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
