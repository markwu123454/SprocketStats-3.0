#!/usr/bin/env python3
"""
Step 0 -- Split broadcast frame into camera views using temporal analysis.

FRC broadcasts embed 2-3 camera feeds in one frame. CG borders and overlays
are rendered at exact fixed pixel positions by the broadcast graphics engine --
they are bit-identical across every frame. Actual camera content always
changes due to motion, sensor noise, and compression variation.

Algorithm
---------
1. Sample N frames spread across a match.
2. Per pixel: compute the range (max - min) across all N frames.
   Static CG pixels -> range ~ 0.
   Live camera pixels -> range >> 0 (motion, noise, compression).
3. Threshold the range image to get a binary "dynamic" mask.
4. Morphological close to fill small static islands within camera views
   (parked robot, still crowd member, etc.).
5. Find bounding rectangles of connected dynamic regions.
6. Keep only rectangles larger than MIN_AREA_FRACTION of the full frame.
   These are the camera views.
7. Sort and name them: main (largest / topmost), bot_left, bot_right, etc.

Outputs (JSON to stdout)
------------------------
{
  "layout":    "stacked" | "side_by_side" | "single",
  "main":      [x0, y0, x1, y1],
  "bot_left":  [x0, y0, x1, y1],    # stacked layout
  "bot_right": [x0, y0, x1, y1],
  "left":      [x0, y0, x1, y1],    # side_by_side layout
  "center":    [x0, y0, x1, y1],
  "right":     [x0, y0, x1, y1],
  "views":     [ {"name": ..., "box": [x0,y0,x1,y1], "fraction": 0.xx}, ... ]
}

Usage
-----
  python pipeline/00_split_views.py --video match.mp4 --viz
  python pipeline/00_split_views.py --event 2026mabil_qm46 --viz
  python pipeline/00_split_views.py --event 2026mabil_qm46 --frames 1,22,50,100,150,200 --viz

Install: pip install opencv-python requests numpy
"""

import argparse, json, sys, pathlib
import numpy as np
import cv2
import requests

CDN_BASE  = "https://assets.markwu.org/sprocketstats/training_round_1"
DATA_DIR  = pathlib.Path(__file__).parent.parent / "data"
IMG_CACHE = DATA_DIR / "images"
PROF_DIR  = DATA_DIR / "profiles"

# Pixel range below this is considered static CG (0-255 scale).
# Compressed H.264 CG elements: range ~2-5. Camera content: range ~30+.
STATIC_THRESHOLD = 12

# Morphological close kernel size (px). Fills static islands within a view
# (parked robot, stopped game piece) without merging adjacent camera views.
# Camera borders are typically 1-5 px wide, so keep this well below that.
CLOSE_KERNEL_PX = 8

# A bounding rectangle must cover at least this fraction of the frame to
# count as a camera view (filters noise and tiny UI elements).
MIN_VIEW_FRACTION = 0.08


# ---------------------------------------------------------------------------
# Image / frame loading
# ---------------------------------------------------------------------------

def fetch_frame(event_match: str, frame: int) -> pathlib.Path:
    fname = f"frame_{frame:06d}.jpg"
    local = IMG_CACHE / f"{event_match}_{fname}"
    if local.exists() and local.stat().st_size > 0:
        return local
    url = f"{CDN_BASE}/{event_match}/{fname}"
    print(f"[fetch] {url}", file=sys.stderr)
    IMG_CACHE.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    local.write_bytes(r.content)
    return local


def probe_frame_count(event_match: str) -> int:
    """
    Binary-search the CDN to find approximately how many frames exist.

    The intro/outro CG overlays cover the first and last portion of the
    video; we need to know the total count so we can sample from the middle.
    Returns the highest frame index that successfully downloads (+/-a few).
    """
    # First find an upper bound by doubling until we get a 404
    lo, hi = 1, 50
    while True:
        url = f"{CDN_BASE}/{event_match}/frame_{hi:06d}.jpg"
        local = IMG_CACHE / f"{event_match}_frame_{hi:06d}.jpg"
        if local.exists():
            lo = hi
            hi *= 2
            continue
        try:
            r = requests.head(url, timeout=10)
            if r.status_code == 200:
                lo = hi
                hi *= 2
            else:
                break
        except Exception:
            break
        if hi > 10_000:
            break   # sanity cap

    # Binary search between lo and hi
    while hi - lo > 5:
        mid = (lo + hi) // 2
        url = f"{CDN_BASE}/{event_match}/frame_{mid:06d}.jpg"
        local = IMG_CACHE / f"{event_match}_frame_{mid:06d}.jpg"
        exists = local.exists()
        if not exists:
            try:
                r = requests.head(url, timeout=10)
                exists = r.status_code == 200
            except Exception:
                exists = False
        if exists:
            lo = mid
        else:
            hi = mid

    return lo


def middle_frames(event_match: str, n: int = 30,
                  start_frac: float = 0.25, end_frac: float = 0.75) -> list[int]:
    """
    Return n evenly-spaced frame indices from the middle portion of the video.

    Avoids the intro/outro CG overlays which cover the first and last segments.
    """
    total = probe_frame_count(event_match)
    print(f"[probe] ~{total} frames total for {event_match}", file=sys.stderr)
    lo  = max(1, int(total * start_frac))
    hi  = max(lo + 1, int(total * end_frac))
    step = max(1, (hi - lo) // (n - 1))
    return list(range(lo, hi + 1, step))[:n]


def sample_video_frames(video_path: str, n: int = 200,
                        start_frac: float = 0.25,
                        end_frac: float = 0.75) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Sample n evenly-spaced frames from the middle portion of a local video.
    Returns (grays, ref_bgr) where grays are grayscale frames and ref_bgr is
    the middle sampled frame in color (used as the reference image for viz).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[error] cannot open video {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {total} frames total in {video_path}", file=sys.stderr)
    lo  = max(0, int(total * start_frac))
    hi  = min(total - 1, int(total * end_frac))
    step = max(1, (hi - lo) // max(1, n - 1))
    indices = list(range(lo, hi + 1, step))[:n]
    print(f"[video] sampling {len(indices)} frames "
          f"[{indices[0]}..{indices[-1]}]", file=sys.stderr)
    grays = []
    ref_bgr = None
    mid_idx = indices[len(indices) // 2]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"[warn] could not read frame {idx}", file=sys.stderr)
            continue
        if idx == mid_idx:
            ref_bgr = frame
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if ref_bgr is None and grays:
        # fallback: re-read the middle index
        cap2 = cv2.VideoCapture(video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        _, ref_bgr = cap2.read()
        cap2.release()
    return grays, ref_bgr


def load_gray(path: pathlib.Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"[error] cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Temporal accumulation
# ---------------------------------------------------------------------------

def accumulate_range(frames_gray: list[np.ndarray]) -> np.ndarray:
    """
    Per-pixel temporal range (max - min) across N frames.
    Returns a float32 array with values in [0, 255].
    Static pixels -> ~0.   Dynamic pixels -> large positive.
    Uses incremental min/max to avoid stacking all frames into one array
    (200 frames at 1080p would be ~1.7 GB of float32 otherwise).
    """
    mn = frames_gray[0].astype(np.float32)
    mx = mn.copy()
    for f in frames_gray[1:]:
        fa = f.astype(np.float32)
        np.minimum(mn, fa, out=mn)
        np.maximum(mx, fa, out=mx)
    return mx - mn


def save_range_image(range_img: np.ndarray, path: pathlib.Path):
    """Save the range image scaled to [0,255] for inspection."""
    mx = float(range_img.max())
    if mx > 0:
        scaled = np.clip(range_img / mx * 255, 0, 255).astype(np.uint8)
    else:
        scaled = np.zeros_like(range_img, dtype=np.uint8)
    cv2.imwrite(str(path), scaled)


# ---------------------------------------------------------------------------
# Dynamic mask -> camera rectangles
# ---------------------------------------------------------------------------

def _trough_band(profile: np.ndarray, grow_factor: float = 8.0,
                 edge_frac: float = 0.05,
                 max_trough_frac: float = 0.05) -> tuple[int, int] | None:
    """
    Find the deepest trough in `profile` and grow outward until values exceed
    grow_factor * trough_min.

    Returns (start, end) pixel indices, or None if:
      - the minimum is too close to an edge, or
      - trough_min / profile.max() >= max_trough_frac  (not a real static separator,
        just the quietest part of camera content)
    """
    n = len(profile)
    lo, hi = int(n * edge_frac), int(n * (1 - edge_frac))
    if hi <= lo:
        return None
    search = profile[lo:hi]
    idx = int(search.argmin()) + lo
    trough_min = profile[idx]

    # Gate: if the trough is not significantly lower than peak content, skip it
    if profile.max() > 0 and trough_min / profile.max() >= max_trough_frac:
        return None

    threshold = trough_min * grow_factor
    left = idx
    while left > 0 and profile[left - 1] <= threshold:
        left -= 1
    right = idx
    while right < n - 1 and profile[right + 1] <= threshold:
        right += 1

    return (left, right)


def find_camera_rects(range_img: np.ndarray,
                      min_fraction: float = MIN_VIEW_FRACTION) -> list[dict]:
    """
    Hierarchical variance-based camera view detection.

    1. Compute row-wise variance of the range image -> find the horizontal
       separator band as the deepest trough (CG score strip between views).
    2. For each content strip above/below, independently compute column-wise
       variance -> find any vertical separator the same way.
    3. Return bounding rects for all resulting camera-view regions.
    """
    h, w = range_img.shape
    total = h * w

    # --- Step 1: horizontal separator via row_var ---
    row_var = np.var(range_img, axis=1)
    h_band  = _trough_band(row_var)          # (y_sep_start, y_sep_end) or None

    if h_band:
        y_sep_s, y_sep_e = h_band
        row_strips = []
        if y_sep_s > 0:
            row_strips.append((0, y_sep_s))
        if y_sep_e + 1 < h:
            row_strips.append((y_sep_e + 1, h))
    else:
        row_strips = [(0, h)]

    rects = []
    for y0, y1 in row_strips:
        strip = range_img[y0:y1, :]

        # --- Step 2: vertical separator within this strip via col_var ---
        col_var = np.var(strip, axis=0)
        v_band  = _trough_band(col_var)      # (x_sep_start, x_sep_end) or None

        if v_band:
            x_sep_s, x_sep_e = v_band
            col_segs = []
            if x_sep_s > 0:
                col_segs.append((0, x_sep_s))
            if x_sep_e + 1 < w:
                col_segs.append((x_sep_e + 1, w))
        else:
            col_segs = [(0, w)]

        for x0, x1 in col_segs:
            area = (x1 - x0) * (y1 - y0)
            frac = area / total
            if frac >= min_fraction:
                rects.append({"box": [x0, y0, x1, y1],
                               "area": area, "fraction": round(frac, 3)})

    rects.sort(key=lambda r: -r["area"])
    return rects


# ---------------------------------------------------------------------------
# Layout classification and naming
# ---------------------------------------------------------------------------

def classify_and_name(rects: list[dict], img_w: int, img_h: int) -> dict:
    """
    Given sorted (largest-first) camera rectangles, classify the layout and
    assign names.

    Rules (covers observed FRC broadcast styles):
      single       -- 1 rect  -> "main"
      stacked      -- 2+ rects where the largest is significantly taller in its
                     vertical position than the others (top vs bottom)
      side_by_side -- 2+ rects all at roughly the same vertical start position
    """
    if not rects:
        return {"layout": "single",
                "views": [{"name": "main", "box": [0, 0, img_w, img_h],
                            "fraction": 1.0}]}

    if len(rects) == 1:
        r = rects[0]
        return {"layout": "single",
                "views": [{"name": "main", **r}]}

    # Sort by top-left position (top-to-bottom, left-to-right)
    by_pos = sorted(rects, key=lambda r: (r["box"][1], r["box"][0]))

    # Determine if there's a meaningful vertical split:
    # "stacked" if the topmost rect is clearly above the next one.
    top_y_of_second = by_pos[1]["box"][1]
    bot_y_of_first  = by_pos[0]["box"][3]
    has_vertical_split = top_y_of_second > bot_y_of_first * 0.6

    if has_vertical_split:
        # Stacked: first rect = main (top), remaining = bottom cameras
        layout = "stacked"
        named  = []
        named.append({"name": "main", **by_pos[0]})

        # Bottom cameras sorted left -> right
        bottom = sorted(by_pos[1:], key=lambda r: r["box"][0])
        names  = ["bot_left", "bot_right", "bot_extra"]
        for i, r in enumerate(bottom):
            named.append({"name": names[i] if i < len(names) else f"bot_{i}", **r})

    else:
        # Side by side: sort left -> right
        layout = "side_by_side"
        by_x   = sorted(rects, key=lambda r: r["box"][0])
        if len(by_x) == 2:
            names = ["left", "right"]
        elif len(by_x) == 3:
            names = ["left", "center", "right"]
        else:
            names = [f"view_{i}" for i in range(len(by_x))]

        named = [{"name": n, **r} for n, r in zip(names, by_x)]

        # "main" = the largest (most field content)
        main_view = max(named, key=lambda r: r["area"])
        for v in named:
            if v["name"] == main_view["name"]:
                named.append({"name": "main", **{k: v[k] for k in v if k != "name"}})
                break

    return {"layout": layout, "views": named}


def named_crops(classification: dict) -> dict:
    """Flatten views list into a {name: [x0,y0,x1,y1]} dict."""
    return {v["name"]: v["box"] for v in classification["views"]}


# ---------------------------------------------------------------------------
# Profile cache
# ---------------------------------------------------------------------------

def profile_path(event_match: str) -> pathlib.Path:
    PROF_DIR.mkdir(parents=True, exist_ok=True)
    return PROF_DIR / f"{event_match}_layout.json"


def load_profile(event_match: str, img_w: int, img_h: int) -> dict | None:
    p = profile_path(event_match)
    if not p.exists():
        return None
    cached = json.loads(p.read_text())
    if cached.get("img_w") != img_w or cached.get("img_h") != img_h:
        print(f"[profile] dims changed, ignoring cache", file=sys.stderr)
        return None
    print(f"[profile] using cached layout for {event_match}", file=sys.stderr)
    return cached


def save_profile(event_match: str, result: dict, img_w: int, img_h: int):
    payload = {**result, "img_w": img_w, "img_h": img_h}
    profile_path(event_match).write_text(json.dumps(payload, indent=2))
    print(f"[profile] saved -> {profile_path(event_match)}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_COLORS = {
    "main":      (0, 230,  0),
    "bot_left":  (0, 160, 255),
    "bot_right": (0, 160, 255),
    "bot_extra": (0, 100, 200),
    "left":      (0, 160, 255),
    "center":    (255, 200,  0),
    "right":     (0, 160, 255),
}

def visualize(img_bgr: np.ndarray, range_img: np.ndarray,
              classification: dict) -> np.ndarray:
    h, w = img_bgr.shape[:2]

    left = img_bgr.copy()
    for v in classification["views"]:
        x0, y0, x1, y1 = v["box"]
        color = _COLORS.get(v["name"], (200, 200, 200))
        cv2.rectangle(left, (x0, y0), (x1, y1), color, 2)
        cv2.putText(left, f"{v['name']} {v['fraction']:.0%}",
                    (x0 + 6, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    layout = classification["layout"]
    cv2.putText(left, f"layout={layout}  views={len(classification['views'])}",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Right panel: range image scaled for inspection
    scaled = np.clip(range_img / max(range_img.max(), 1) * 255, 0, 255).astype(np.uint8)
    range_bgr = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)

    scale = 0.5
    left_s  = cv2.resize(left,      (int(w * scale), int(h * scale)))
    range_s = cv2.resize(range_bgr, (int(w * scale), int(h * scale)))
    return np.hstack([left_s, range_s])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--event", metavar="EVENT_MATCH",
                     help="event+match key, e.g. 2026mabil_qm46")
    grp.add_argument("--video", metavar="PATH",
                     help="local video file; samples --n-frames frames from the middle")
    ap.add_argument("--n-frames", type=int, default=200,
                    help="frames to sample from --video (default: 200)")
    ap.add_argument("--frames", default="",
                    help="comma-separated frame numbers to accumulate "
                         "(used with --event)")
    ap.add_argument("--range-image", metavar="PATH",
                    help="use a pre-computed range PNG instead of downloading frames "
                         "(used with --event)")
    ap.add_argument("--threshold", type=float, default=STATIC_THRESHOLD,
                    help=f"static/dynamic pixel threshold (default: {STATIC_THRESHOLD})")
    ap.add_argument("--min-fraction", type=float, default=MIN_VIEW_FRACTION,
                    help=f"min view area as fraction of frame (default: {MIN_VIEW_FRACTION})")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--save-crops", action="store_true",
                    help="write cropped images to data/images/")
    ap.add_argument("--save-range", action="store_true",
                    help="save the accumulated range image for inspection")
    ap.add_argument("--viz", action="store_true",
                    help="save side-by-side debug image")
    ap.add_argument("--viz-out", metavar="PATH")
    args = ap.parse_args()

    # ---- resolve event_match and frame list ----
    video_grays = None   # set when --video is used
    if args.video:
        video_path  = args.video
        event_match = pathlib.Path(video_path).stem
        video_grays, ref_img = sample_video_frames(video_path, args.n_frames)
        if ref_img is None:
            sys.exit("[error] could not read any frames from video")
        img_h, img_w = ref_img.shape[:2]
        ref_path = pathlib.Path(video_path)
    else:  # --event
        event_match = args.event
        if args.frames:
            frame_indices = [int(x) for x in args.frames.split(",")]
        else:
            frame_indices = middle_frames(event_match)
            print(f"[frames] sampling {frame_indices}", file=sys.stderr)
        ref_path = fetch_frame(event_match, frame_indices[0])
        ref_img  = cv2.imread(str(ref_path))
        if ref_img is None:
            sys.exit(f"[error] cannot read {ref_path}")
        img_h, img_w = ref_img.shape[:2]

    # ---- load or compute ----
    cached = None if args.no_cache else load_profile(event_match, img_w, img_h)
    if cached:
        classification = {"layout": cached["layout"], "views": cached["views"]}
        crops = named_crops(classification)
        range_img = None
        mask_img  = None
    else:
        # Accumulate frames
        if args.range_image:
            ri = cv2.imread(args.range_image, cv2.IMREAD_GRAYSCALE)
            if ri is None:
                sys.exit(f"[error] cannot read range image {args.range_image}")
            range_img = ri.astype(np.float32) / 255.0 * args.threshold * 10
        elif video_grays is not None:
            print(f"[accum] accumulating {len(video_grays)} video frames ...",
                  file=sys.stderr)
            if not video_grays:
                sys.exit("[error] no frames loaded from video")
            range_img = accumulate_range(video_grays)
            print(f"[accum] range: min={range_img.min():.1f} "
                  f"max={range_img.max():.1f} "
                  f"mean={range_img.mean():.1f}", file=sys.stderr)
        else:
            print(f"[accum] loading {len(frame_indices)} frames ...", file=sys.stderr)
            grays = []
            for fi in frame_indices:
                try:
                    grays.append(load_gray(fetch_frame(event_match, fi)))
                except Exception as e:
                    print(f"[warn] frame {fi} failed: {e}", file=sys.stderr)
            if not grays:
                sys.exit("[error] no frames loaded")
            range_img = accumulate_range(grays)
            print(f"[accum] range: min={range_img.min():.1f} "
                  f"max={range_img.max():.1f} "
                  f"mean={range_img.mean():.1f}", file=sys.stderr)

        if args.save_range:
            rp = DATA_DIR / "images" / f"{event_match}_range.png"
            save_range_image(range_img, rp)
            print(f"[range] saved -> {rp}", file=sys.stderr)

        rects          = find_camera_rects(range_img, args.min_fraction)
        classification = classify_and_name(rects, img_w, img_h)
        mask_img       = None
        crops          = named_crops(classification)

        print(f"[split] layout={classification['layout']}  "
              f"views={[v['name'] for v in classification['views']]}",
              file=sys.stderr)

        save_profile(event_match, classification, img_w, img_h)

    # ---- optional outputs ----
    if args.save_crops:
        out_dir = IMG_CACHE
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = ref_path.stem
        for name, (x0, y0, x1, y1) in crops.items():
            c = ref_img[y0:y1, x0:x1]
            p = out_dir / f"{stem}_{name}.jpg"
            cv2.imwrite(str(p), c)
            print(f"[crop] {name} -> {p}", file=sys.stderr)

    if (args.viz or args.viz_out) and range_img is not None:
        vis = visualize(ref_img, range_img, classification)
        out = args.viz_out or str(ref_path.with_suffix("")) + "_split.jpg"
        cv2.imwrite(out, vis)
        print(f"[viz] {out}", file=sys.stderr)
    elif args.viz and range_img is None:
        print("[viz] skipped (using cached profile, no range image)", file=sys.stderr)


if __name__ == "__main__":
    main()
