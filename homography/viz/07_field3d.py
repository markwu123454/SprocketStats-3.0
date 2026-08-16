#!/usr/bin/env python3
"""
Step -- Generate the hand-rolled canvas 3-D field/camera viewer
(viz/field3d.html) from live pipeline output.

Reads the current pipeline output directly (no Plotly, no dependency on
the old "camera_views"/"tags" poses schema an earlier, now-retired
visualizer expected):
  data/field/<year>_tags.json          -- field AprilTag layout
  data/detections/<stem>_tags.json     -- decoded tags per view (pipeline/01)
  data/detections/<stem>_poses.json    -- solved camera poses (pipeline/03)

and rewrites viz/field3d.html in place. That file is a self-contained
canvas orbit viewer (drag to orbit, shift/right-drag to pan, scroll to
zoom, hover for tooltips) with no external dependencies -- only the DATA
block at the top is regenerated; the renderer below it is static and
untouched.

A view whose pose didn't solve (too few tags even for the corner
fallback -- see pipeline/03_solve_pose.py) is still included with
x/y/z/yaw/pitch/rms = null and its raw decoded tag list, matching how the
viewer already renders an unsolved camera (grayed out, no position marker).

Usage
-----
  python viz/07_field3d.py --video match.mp4
  python viz/07_field3d.py --video match.mp4 --year 2026 --open

Install: (stdlib only)
"""

import argparse, json, math, pathlib, sys, webbrowser

DATA_DIR      = pathlib.Path(__file__).parent.parent / "data"
FIELD_DIR     = DATA_DIR / "field"
DET_DIR       = DATA_DIR / "detections"
TEMPLATE_PATH = pathlib.Path(__file__).parent / "field3d.html"

CAMERA_COLORS = ["#00ccff", "#39e588", "#ff7744", "#d64fff", "#ffd700"]


# ---------------------------------------------------------------------------
# Field/camera data assembly
# ---------------------------------------------------------------------------

def quat_to_facing_xy(qw, qx, qy, qz):
    """Tag's outward-normal XY components in field coordinates."""
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n < 1e-9:
        return 1.0, 0.0
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    nx = 1 - 2*(qy*qy + qz*qz)
    ny = 2*(qx*qy + qw*qz)
    return round(nx, 3), round(ny, 3)


def build_data(field_data: dict, tags_data: dict, poses_data: dict):
    tags_out = []
    for tid_str, t in sorted(field_data["tags"].items(), key=lambda kv: int(kv[0])):
        nx, ny = quat_to_facing_xy(t["qw"], t["qx"], t["qy"], t["qz"])
        tags_out.append({"id": int(tid_str), "x": round(t["x"], 3), "y": round(t["y"], 3),
                         "z": round(t["z"], 3), "nx": nx, "ny": ny})

    views = tags_data.get("views", {})
    cams_out = []
    for i, vname in enumerate(views):  # detection order: main, bot_left, bot_right, ...
        color       = CAMERA_COLORS[i % len(CAMERA_COLORS)]
        decoded_ids = sorted(int(t) for t in views[vname].get("decoded_tags", {}))
        pose        = poses_data.get(vname)
        if pose:
            residuals = {int(k): {"reproj_px": v["reproj_px"], "dist_m": v["dist_m"]}
                        for k, v in pose.get("tag_residuals", {}).items()}
            p = pose["camera_position_field_m"]
            cams_out.append({
                "name": vname, "x": round(p[0], 3), "y": round(p[1], 3), "z": round(p[2], 3),
                "yaw": pose["yaw_deg"], "pitch": pose["pitch_deg"],
                "rms": pose["rms_reproj_px"], "color": color,
                "detected": sorted(residuals) or decoded_ids,
                "residuals": residuals,
            })
        else:
            cams_out.append({
                "name": vname, "x": None, "y": None, "z": None,
                "yaw": None, "pitch": None, "rms": None, "color": color,
                "detected": decoded_ids, "residuals": {},
            })
    return tags_out, cams_out


# ---------------------------------------------------------------------------
# DATA block rendering -- must stay valid JS, matching field3d.html's format
# ---------------------------------------------------------------------------

def render_data_js(field_data: dict, tags_out: list, cams_out: list) -> str:
    def num(v):
        return "null" if v is None else json.dumps(v)

    tag_lines = ",\n    ".join(
        f'{{id:{t["id"]}, x:{t["x"]},y:{t["y"]},z:{t["z"]},nx:{t["nx"]}, ny:{t["ny"]}}}'
        for t in tags_out)

    cam_lines = []
    for c in cams_out:
        res_parts = ",".join(
            f'{k}:{{reproj_px:{v["reproj_px"]},dist_m:{v["dist_m"]}}}'
            for k, v in c["residuals"].items())
        cam_lines.append(
            f'{{ name:{json.dumps(c["name"])}, x:{num(c["x"])}, y:{num(c["y"])}, z:{num(c["z"])}, '
            f'yaw:{num(c["yaw"])}, pitch:{num(c["pitch"])}, rms:{num(c["rms"])}, '
            f'color:{json.dumps(c["color"])},\n'
            f'      detected:{json.dumps(c["detected"])},\n'
            f'      residuals:{{{res_parts}}} }}'
        )
    cams_js = ",\n    ".join(cam_lines)

    return (
        "const DATA = {\n"
        f'  field: {{ length: {field_data["field_length_m"]}, width: {field_data["field_width_m"]} }},\n'
        "  tags: [\n    " + tag_lines + "\n  ],\n"
        "  cameras: [\n    " + cams_js + "\n  ]\n"
        "};"
    )


def rewrite_template(new_data_js: str, out_path: pathlib.Path):
    if not TEMPLATE_PATH.exists():
        sys.exit(f"[error] template not found: {TEMPLATE_PATH}")
    html = TEMPLATE_PATH.read_text()
    start = html.index("const DATA = {")
    end   = html.index("};", start) + 2
    html  = html[:start] + new_data_js + html[end:]
    out_path.write_text(html)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, metavar="PATH")
    ap.add_argument("--year",  type=int, default=2026)
    ap.add_argument("--tags",  metavar="PATH",
                    help="tags JSON (default: data/detections/<stem>_tags.json)")
    ap.add_argument("--poses", metavar="PATH",
                    help="poses JSON (default: data/detections/<stem>_poses.json)")
    ap.add_argument("--out",   metavar="PATH",
                    help=f"output path (default: {TEMPLATE_PATH}, rewritten in place)")
    ap.add_argument("--open",  action="store_true", help="open in browser after generating")
    args = ap.parse_args()

    stem = pathlib.Path(args.video).stem
    field_path = FIELD_DIR / f"{args.year}_tags.json"
    tags_path  = pathlib.Path(args.tags)  if args.tags  else DET_DIR / f"{stem}_tags.json"
    poses_path = pathlib.Path(args.poses) if args.poses else DET_DIR / f"{stem}_poses.json"

    if not field_path.exists():
        sys.exit(f"[error] field layout not found: {field_path}\n"
                 f"        run pipeline/01_fetch_field_layout.py --year {args.year} first")
    if not tags_path.exists():
        sys.exit(f"[error] tags not found: {tags_path}\n"
                 f"        run pipeline/01_detect_tags.py --video first")
    if not poses_path.exists():
        print(f"[warn] poses not found: {poses_path} -- all cameras will render unsolved",
              file=sys.stderr)

    field_data = json.loads(field_path.read_text())
    tags_data  = json.loads(tags_path.read_text())
    poses_data = json.loads(poses_path.read_text()) if poses_path.exists() else {}

    tags_out, cams_out = build_data(field_data, tags_data, poses_data)
    new_data_js = render_data_js(field_data, tags_out, cams_out)

    out_path = pathlib.Path(args.out) if args.out else TEMPLATE_PATH
    rewrite_template(new_data_js, out_path)

    n_solved = sum(1 for c in cams_out if c["x"] is not None)
    print(f"[out] {out_path}  ({len(tags_out)} tags, {len(cams_out)} view(s), "
          f"{n_solved} solved)", file=sys.stderr)

    if args.open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
