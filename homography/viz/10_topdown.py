#!/usr/bin/env python3
"""
Top-down field warp -- unwrap the broadcast frame onto a rectangle matching
the FRC field dimensions, viewed straight down from above.

For every pixel in the output image the corresponding field coordinate
(x, y, z=0) is projected through the solved camera pose back into the
original crop. cv2.remap then samples the crop there. The result is a
geometrically-correct bird's-eye view of whatever the camera can see.

Pixels that project behind the camera or outside the crop are filled with a
dark checkerboard (distinguishable from valid black content).

When multiple views have solved poses they are composited: smaller views
layer first so the main (widest-coverage) view sits on top where they
overlap.

Reads
-----
  data/field/<year>_tags.json              -- field layout + dimensions
  data/detections/<stem>_tags.json         -- view boxes  (pipeline/01)
  data/detections/<stem>_poses.json        -- solved poses (pipeline/03)
  data/calibration/<stem>_intrinsics.json  -- if present  (pipeline/02)

Writes
------
  data/topdown/<stem>_<view>_topdown.jpg   -- per-view warp
  data/topdown/<stem>_topdown.jpg          -- composite of all views

Usage
-----
  python viz/10_topdown.py --video match.mp4
  python viz/10_topdown.py --video match.mp4 --view main
  python viz/10_topdown.py --video match.mp4 --scale 120 --frame 900

Install: pip install opencv-python numpy
"""

import argparse, json, math, pathlib, sys
import numpy as np
import cv2

DATA_DIR       = pathlib.Path(__file__).parent.parent / "data"
FIELD_DIR      = DATA_DIR / "field"
DETECTIONS_DIR = DATA_DIR / "detections"
CALIB_DIR      = DATA_DIR / "calibration"
TOPDOWN_DIR    = DATA_DIR / "topdown"

DEFAULT_SCALE_PX_PER_M = 100.0   # output pixels per metre
DEFAULT_MARGIN_M       = 0.5     # padding around the field rectangle
DEFAULT_FOV            = 70.0

COLOR_BOUNDARY   = (0,  80, 255)   # red-orange
COLOR_CENTERLINE = (0,  60, 200)   # dim red
COLOR_TAG_USED   = (0, 220,   0)   # green  -- tag was used in this pose solve
COLOR_TAG_OTHER  = (0, 140, 255)   # orange -- projected only
COLOR_LABEL      = (220, 220, 220)


# ---------------------------------------------------------------------------
# Intrinsics -- identical to how pipeline/03 and viz/03 load them
# ---------------------------------------------------------------------------

def _load_or_estimate_K(intrinsics_path: pathlib.Path, view_name: str,
                         view_box: list, fov_deg: float,
                         focal_px: float | None) -> tuple[np.ndarray, np.ndarray]:
    if intrinsics_path.exists():
        intr = json.loads(intrinsics_path.read_text())
        if view_name in intr:
            K = np.array(intr[view_name]["K"],    dtype=np.float64)
            d = np.array(intr[view_name]["dist"],  dtype=np.float64)
            print(f"  [intrinsics] loaded  f={K[0,0]:.1f}px", file=sys.stderr)
            return K, d
    x0, y0, x1, y1 = view_box
    w = x1 - x0
    f = (focal_px if focal_px is not None
         else (w / 2.0) / math.tan(math.radians(fov_deg / 2.0)))
    h = y1 - y0
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)
    d = np.zeros(5, dtype=np.float64)
    print(f"  [intrinsics] estimated  f={f:.1f}px  fov={fov_deg:.1f}°", file=sys.stderr)
    return K, d


# ---------------------------------------------------------------------------
# Top-down warp
# ---------------------------------------------------------------------------

def _checkerboard(h: int, w: int, sq: int = 16) -> np.ndarray:
    r = np.arange(h, dtype=np.uint8)[:, None]
    c = np.arange(w, dtype=np.uint8)[None, :]
    v = np.where((r // sq + c // sq) % 2 == 0, np.uint8(18), np.uint8(28))
    return np.stack([v, v, v], axis=2)


def _field_to_px(field_x: float, field_y: float,
                 out_h: int, scale: float, margin_m: float) -> tuple[int, int]:
    """Field (x, y) in metres → output image (col, row).

    x increases left-to-right.  y increases bottom-to-top in field coords,
    but top-to-bottom in image coords, so we flip it.
    """
    col = int(round((field_x + margin_m) * scale))
    row = out_h - 1 - int(round((field_y + margin_m) * scale))
    return col, row


def warp_view(crop: np.ndarray,
              K: np.ndarray, dist: np.ndarray,
              rvec, tvec,
              fl: float, fw: float,
              scale: float, margin_m: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Reverse-project the field plane (z=0) into `crop` pixel coordinates
    and remap.

    Returns
    -------
    warped : (out_h, out_w, 3) BGR image
    valid  : (out_h, out_w) uint8 mask -- 255 where crop data exists, 0 elsewhere
    """
    out_w   = int(round((fl + 2 * margin_m) * scale))
    out_h   = int(round((fw + 2 * margin_m) * scale))
    crop_h, crop_w = crop.shape[:2]

    # Grid of output pixel indices
    cols = np.arange(out_w, dtype=np.float64)
    rows = np.arange(out_h, dtype=np.float64)
    cc, rr = np.meshgrid(cols, rows)                  # both (out_h, out_w)

    # Corresponding field coordinates at z=0
    field_x = (cc - margin_m * scale) / scale
    field_y = (out_h - 1 - rr - margin_m * scale) / scale   # flip y
    field_z = np.zeros_like(field_x)

    pts_field = np.stack(                              # (N, 3)
        [field_x.ravel(), field_y.ravel(), field_z.ravel()], axis=1)

    # Cull points behind the camera before calling projectPoints
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
    t    = np.array(tvec, dtype=np.float64).flatten()
    pts_cam   = (R @ pts_field.T).T + t               # (N, 3) in camera frame
    in_front  = pts_cam[:, 2] > 0.01                  # (N,) bool

    # Project all field points (invalid ones will end up out-of-bounds)
    img_pts, _ = cv2.projectPoints(
        pts_field.reshape(-1, 1, 3).astype(np.float64),
        np.array(rvec, dtype=np.float64),
        np.array(tvec, dtype=np.float64),
        K, dist,
    )
    img_pts = img_pts.reshape(-1, 2)                  # (N, 2) crop-pixel coords

    map_x = img_pts[:, 0].astype(np.float32).reshape(out_h, out_w)
    map_y = img_pts[:, 1].astype(np.float32).reshape(out_h, out_w)

    # Mark pixels behind the camera or outside the crop as invalid
    invalid = (
        ~in_front.reshape(out_h, out_w)
        | (map_x < 0) | (map_x >= crop_w)
        | (map_y < 0) | (map_y >= crop_h)
    )
    # Sending invalid coords to (-1, -1) makes remap's BORDER_CONSTANT fill them
    map_x[invalid] = -1.0
    map_y[invalid] = -1.0

    warped = cv2.remap(crop, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(0, 0, 0))

    valid = np.where(invalid, np.uint8(0), np.uint8(255))
    return warped, valid


# ---------------------------------------------------------------------------
# Field overlay
# ---------------------------------------------------------------------------

def _draw_field_overlay(canvas: np.ndarray,
                         fl: float, fw: float,
                         scale: float, margin_m: float,
                         all_tags: dict, used_ids: set):
    out_h = canvas.shape[0]

    def fp(x, y):
        return _field_to_px(x, y, out_h, scale, margin_m)

    # Boundary rectangle
    corners = [fp(0, 0), fp(fl, 0), fp(fl, fw), fp(0, fw)]
    cv2.polylines(canvas,
                  [np.array(corners, dtype=np.int32).reshape(-1, 1, 2)],
                  isClosed=True, color=COLOR_BOUNDARY, thickness=2,
                  lineType=cv2.LINE_AA)

    # Center line
    cv2.line(canvas, fp(fl / 2, 0), fp(fl / 2, fw),
             COLOR_CENTERLINE, 1, cv2.LINE_AA)

    # AprilTag dot + id label
    for tid_str, tag in sorted(all_tags.items(), key=lambda kv: int(kv[0])):
        cx, cy = fp(tag["x"], tag["y"])
        color = COLOR_TAG_USED if tid_str in used_ids else COLOR_TAG_OTHER
        cv2.circle(canvas, (cx, cy), 5, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, tid_str, (cx + 7, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Corner labels
    for label, (fx, fy) in [("(0,0)", (0, 0)), (f"({fl:.1f}m,0)", (fl, 0)),
                              (f"(0,{fw:.1f}m)", (0, fw))]:
        col, row = fp(fx, fy)
        cv2.putText(canvas, label, (col + 4, row - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_LABEL, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video",    required=True, metavar="PATH")
    ap.add_argument("--year",     type=int, default=2026,
                    choices=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--view",     metavar="NAME",
                    help="warp only this view (default: all views with solved poses)")
    ap.add_argument("--frame",    type=int, metavar="N",
                    help="video frame index to grab (default: middle of video)")
    ap.add_argument("--scale",    type=float, default=DEFAULT_SCALE_PX_PER_M,
                    metavar="PX_PER_M",
                    help=f"output pixels per metre (default: {DEFAULT_SCALE_PX_PER_M})")
    ap.add_argument("--margin",   type=float, default=DEFAULT_MARGIN_M,
                    metavar="METRES",
                    help=f"padding around the field in the output (default: {DEFAULT_MARGIN_M}m)")
    ap.add_argument("--tags",     metavar="PATH",
                    help="tags JSON (default: data/detections/<stem>_tags.json)")
    ap.add_argument("--poses",    metavar="PATH",
                    help="poses JSON (default: data/detections/<stem>_poses.json)")
    ap.add_argument("--fov-deg",  type=float, default=DEFAULT_FOV,
                    help="fallback FOV if no intrinsics file (default: %(default)s°)")
    ap.add_argument("--focal-px", type=float, default=None,
                    help="fallback focal length in pixels (overrides --fov-deg)")
    ap.add_argument("--out-dir",  metavar="DIR",
                    help=f"output directory (default: {TOPDOWN_DIR})")
    args = ap.parse_args()

    # Field layout
    field_path = FIELD_DIR / f"{args.year}_tags.json"
    if not field_path.exists():
        sys.exit(f"[error] field layout not found: {field_path}\n"
                 f"        run: python pipeline/01_fetch_field_layout.py --year {args.year}")
    field_data = json.loads(field_path.read_text())
    all_tags = field_data["tags"]
    fl = field_data.get("field_length_m", 16.541)
    fw = field_data.get("field_width_m",  8.069)
    print(f"[field] {fl:.2f}m × {fw:.2f}m", file=sys.stderr)

    # Detection / pose data
    stem       = pathlib.Path(args.video).stem
    tags_path  = pathlib.Path(args.tags)  if args.tags  else DETECTIONS_DIR / f"{stem}_tags.json"
    poses_path = pathlib.Path(args.poses) if args.poses else DETECTIONS_DIR / f"{stem}_poses.json"
    for p in (tags_path, poses_path):
        if not p.exists():
            sys.exit(f"[error] not found: {p}\n"
                     f"        run the pipeline steps (01_detect_tags, "
                     f"03_solve_pose) first")
    tags_data  = json.loads(tags_path.read_text())
    poses      = json.loads(poses_path.read_text())

    intrinsics_path = CALIB_DIR / f"{stem}_intrinsics.json"
    views = tags_data.get("views", {})

    view_names = [args.view] if args.view else sorted(poses)
    view_names = [v for v in view_names if v in poses and v in views]
    if not view_names:
        sys.exit("[error] no view has both a box (tags JSON) and a solved pose")

    # Grab a single frame
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[error] cannot open video: {args.video}")
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = args.frame if args.frame is not None else total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"[error] could not read frame {frame_idx} from {args.video}")
    print(f"[video] frame {frame_idx}/{total}", file=sys.stderr)

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else TOPDOWN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    out_w = int(round((fl + 2 * args.margin) * args.scale))
    out_h = int(round((fw + 2 * args.margin) * args.scale))
    print(f"[output] {out_w}×{out_h}px  scale={args.scale}px/m  "
          f"margin={args.margin}m", file=sys.stderr)

    # Composite canvas: checkerboard background, views layered smallest → largest
    # so the widest-coverage view (main) sits on top where views overlap.
    def _view_crop_area(v):
        x0, y0, x1, y1 = views[v]["box"]
        return (x1 - x0) * (y1 - y0)

    composite      = _checkerboard(out_h, out_w)
    composite_mask = np.zeros((out_h, out_w), dtype=np.uint8)

    for view_name in sorted(view_names, key=_view_crop_area):
        print(f"[{view_name}]", file=sys.stderr)
        vbox = views[view_name]["box"]
        x0, y0, x1, y1 = vbox
        crop = frame_bgr[y0:y1, x0:x1]
        pose = poses[view_name]
        K, dist = _load_or_estimate_K(intrinsics_path, view_name, vbox,
                                      args.fov_deg, args.focal_px)

        warped, valid = warp_view(crop, K, dist,
                                  pose["rvec"], pose["tvec_m"],
                                  fl, fw, args.scale, args.margin)

        # Coverage stat
        n_valid = int(valid.astype(bool).sum())
        pct     = 100 * n_valid / (out_w * out_h)
        print(f"  coverage: {pct:.1f}%  ({n_valid}/{out_w*out_h} px)", file=sys.stderr)

        # Layer onto composite
        composite[valid > 0] = warped[valid > 0]
        composite_mask      |= valid

        # Per-view output: checkerboard bg + this view's warp + overlay
        used_ids = set(pose.get("tag_residuals", {}))
        per_view = _checkerboard(out_h, out_w)
        per_view[valid > 0] = warped[valid > 0]
        _draw_field_overlay(per_view, fl, fw, args.scale, args.margin, all_tags, used_ids)
        cv2.putText(per_view, f"{view_name}  frame={frame_idx}  coverage={pct:.0f}%",
                    (6, out_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_LABEL, 1)
        out_path = out_dir / f"{stem}_{view_name}_topdown.jpg"
        cv2.imwrite(str(out_path), per_view, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  [out] {out_path}", file=sys.stderr)

    # Composite output
    used_all = set().union(*(set(poses[v].get("tag_residuals", {})) for v in view_names))
    _draw_field_overlay(composite, fl, fw, args.scale, args.margin, all_tags, used_all)
    total_pct = 100 * int(composite_mask.astype(bool).sum()) / (out_w * out_h)
    cv2.putText(composite,
                f"all views  frame={frame_idx}  coverage={total_pct:.0f}%",
                (6, out_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_LABEL, 1)
    comp_path = out_dir / f"{stem}_topdown.jpg"
    cv2.imwrite(str(comp_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[composite] {total_pct:.1f}% coverage  ->  {comp_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
