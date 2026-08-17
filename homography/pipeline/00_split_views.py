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
3. Row-wise, then column-wise (within each resulting strip), look for
   separators along the temporal-range variance profile. Most districts
   draw a static CG border/scoreboard strip between feeds, which shows up
   as one or more troughs (see _local_minima_bands) -- not just the single
   deepest one, so a row with three items (view | cg | view) still splits
   correctly.
4. Districts that composite feeds edge-to-edge with no static graphic
   between them have no such trough. When none is found, fall back to a
   persistent-edge search (_gradient_ridge_seam): a compositing seam sits
   at the same pixel column/row in every frame, unlike real scene content
   which drifts frame to frame and smears out under averaging. A candidate
   found this way is confirmed via temporal decorrelation
   (_seam_is_independent) so a strong static edge inside one real camera
   view -- a yard line, a scoring-table divider -- isn't mistaken for a
   seam between two different ones.
5. Keep only resulting rectangles larger than MIN_VIEW_FRACTION of the
   full frame. These are the camera views.
6. Sort into a stable reading order (top-to-bottom, left-to-right) and
   assign each an opaque index name (view0, view1, ...) purely so output
   is deterministic across runs -- this is not a claim about which
   physical camera feed a view is; downstream code never inspects the name.

Outputs (JSON to stdout)
------------------------
{
  "views": [ {"name": "view0", "box": [x0,y0,x1,y1], "fraction": 0.xx}, ... ]
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

# A bounding rectangle must cover at least this fraction of the frame to
# count as a camera view (filters noise and tiny UI elements).
MIN_VIEW_FRACTION = 0.08

# --- Borderless-seam fallback (no static CG graphic between views) ---
# Validated against real match footage (see homography/docs) -- a confirmed
# genuine seam came out ~2px wide and present in ~100% of sampled frames; a
# false candidate inside a CG graphic's internal divider came out ~6px wide
# and merely "usually" present. Peak-ratio and correlation thresholds are
# still starting points, not independently validated.
RIDGE_MIN_PEAK_RATIO      = 4.0   # candidate seam's gradient peak vs. neighborhood median
RIDGE_EDGE_FRAC           = 0.05  # ignore this fraction of the strip at each edge
RIDGE_MAX_FWHM_PX         = 4     # max width (px) of the averaged gradient peak
RIDGE_MIN_FRAME_CONSISTENCY = 0.9 # fraction of sampled frames the edge must appear in
SEAM_CORR_MAX             = 0.6   # frame-to-frame delta correlation across the seam must be below this
SEAM_MARGIN_PX            = 4     # strip width sampled on each side of a candidate seam
SEAM_MIN_FRAMES           = 8     # too few frames makes the correlation meaningless


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

def _local_minima_bands(profile: np.ndarray, grow_factor: float = 8.0,
                        edge_frac: float = RIDGE_EDGE_FRAC,
                        max_trough_frac: float = 0.05) -> list[tuple[int, int]]:
    """
    Find every local minimum in `profile` that's a plausible static CG
    separator, and grow each outward the same way a single deepest trough
    would be.

    A layout with more than one separator per axis (e.g. view | cg | view)
    has more than one such minimum, so this returns however many qualify --
    not just the single deepest one. A local minimum only qualifies if it's
    meaningfully below the profile's peak (max_trough_frac gate); that
    filters out the merely-quietest patch of real camera content, which is
    never anywhere near as flat as an actual static CG region.
    """
    n = len(profile)
    lo, hi = int(n * edge_frac), int(n * (1 - edge_frac))
    if hi <= lo:
        return []
    pmax = float(profile.max())
    if pmax <= 0:
        return []

    bands = []
    for idx in range(lo, hi):
        left  = profile[idx - 1] if idx > 0 else profile[idx]
        right = profile[idx + 1] if idx < n - 1 else profile[idx]
        if profile[idx] > left or profile[idx] > right:
            continue  # not a local minimum
        val = profile[idx]
        if val / pmax >= max_trough_frac:
            continue  # not meaningfully below peak content

        threshold = val * grow_factor
        l = idx
        while l > 0 and profile[l - 1] <= threshold:
            l -= 1
        r = idx
        while r < n - 1 and profile[r + 1] <= threshold:
            r += 1
        bands.append((l, r))

    return _merge_bands(bands)


def _merge_bands(bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse overlapping/adjacent (start, end) bands -- e.g. two nearby
    points on the same flat minimum plateau growing into each other."""
    if not bands:
        return []
    bands = sorted(bands)
    merged = [list(bands[0])]
    for s, e in bands[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _segments_from_bands(bands: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    """Turn a sorted list of separator bands into the content segments
    between/around them. No bands -> the whole span is one segment."""
    if not bands:
        return [(0, length)]
    segs = []
    prev_end = 0
    for s, e in bands:
        if s > prev_end:
            segs.append((prev_end, s))
        prev_end = e + 1
    if prev_end < length:
        segs.append((prev_end, length))
    return segs


def _gradient_ridge_seam(frames_gray: list[np.ndarray], axis: int) -> int | None:
    """
    Find a candidate compositing-seam location for a view split that has no
    static CG graphic drawn at the boundary.

    _local_minima_bands finds STATIC pixels; this finds a PERSISTENT EDGE
    instead. A broadcast mixer seam sits at the exact same pixel column/row
    in every frame, so the mean gradient magnitude there, averaged over many
    frames, stays high. A real scene edge doesn't: it drifts by a pixel or
    more frame to frame from compression and stabilization jitter, so
    averaging smears it out. That gap is what separates a seam from ordinary
    in-view content.

    `axis=0` looks for a horizontal separator (top/bottom split, scans rows
    via the vertical gradient); `axis=1` looks for a vertical separator
    (left/right split, scans columns via the horizontal gradient).

    Returns the candidate index, or None if nothing stands out enough to be
    worth testing further with `_seam_is_independent`.
    """
    per_frame_profiles = []
    acc = None
    for g in frames_gray:
        gray = g.astype(np.float32)
        if axis == 1:
            grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        else:
            grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        acc = grad if acc is None else acc + grad
        per_frame_profiles.append(grad.mean(axis=0) if axis == 1 else grad.mean(axis=1))
    mean_grad = acc / len(frames_gray)
    profile = mean_grad.mean(axis=0) if axis == 1 else mean_grad.mean(axis=1)

    n = len(profile)
    lo, hi = int(n * RIDGE_EDGE_FRAC), int(n * (1 - RIDGE_EDGE_FRAC))
    if hi <= lo:
        return None
    idx  = int(profile[lo:hi].argmax()) + lo
    peak = profile[idx]

    mask = np.ones(n, dtype=bool)
    mask[max(0, idx - 3):idx + 4] = False
    neighborhood = profile[mask]
    baseline = float(np.median(neighborhood)) if neighborhood.size else 0.0
    if baseline <= 0 or peak / baseline < RIDGE_MIN_PEAK_RATIO:
        return None

    # A real compositing cut is razor-thin -- a mixer switches source at an
    # exact pixel column. A strong edge that's several pixels wide (a
    # divider bar rendered as part of a graphic, a blurred/antialiased
    # scene edge) is not that, even if its average gradient is high.
    # Measured on real footage: a confirmed genuine seam came out ~2 columns
    # wide (full-width-half-max); a false candidate inside a CG graphic's
    # internal divider came out ~6 columns wide with a second nearby bump.
    half = peak / 2.0
    width = int(np.sum(profile > half))
    if width > RIDGE_MAX_FWHM_PX:
        return None

    # A real seam is there in essentially every sampled frame, not just
    # strong on average -- a few outlier frames dragging the mean up (a
    # robot passing near the boundary, a one-off compression artifact)
    # shouldn't count as "the seam is always there".
    frac_present = float(np.mean([p[idx] > half for p in per_frame_profiles]))
    if frac_present < RIDGE_MIN_FRAME_CONSISTENCY:
        return None

    return idx


def _seam_is_independent(frames_gray: list[np.ndarray], axis: int, idx: int) -> bool:
    """
    Confirm a candidate seam actually separates two independently-sourced
    video feeds, rather than being a strong static edge inside one
    continuous camera view (a yard line, a scoring-table divider, an
    internal feature of a CG graphic).

    Correlate frame-to-frame CHANGES on each side, not raw brightness
    levels -- differencing cancels out anything shared and slow (gradual
    exposure drift, shared camera shake) and isolates genuine independent
    content change. Two different camera feeds change independently frame
    to frame; two samples of one continuous feed generally don't, even
    across a strong internal edge.
    """
    if len(frames_gray) < SEAM_MIN_FRAMES:
        return False
    m = SEAM_MARGIN_PX
    limit = frames_gray[0].shape[1] if axis == 1 else frames_gray[0].shape[0]
    if idx - m < 0 or idx + m >= limit:
        return False

    def strip_means(g):
        if axis == 1:
            return g[:, idx - m:idx].mean(), g[:, idx + 1:idx + 1 + m].mean()
        return g[idx - m:idx, :].mean(), g[idx + 1:idx + 1 + m, :].mean()

    signals = np.array([strip_means(g) for g in frames_gray], dtype=np.float64)
    left_delta  = np.diff(signals[:, 0])
    right_delta = np.diff(signals[:, 1])
    if left_delta.std() < 1e-6 or right_delta.std() < 1e-6:
        return False
    corr = float(np.corrcoef(left_delta, right_delta)[0, 1])
    return corr < SEAM_CORR_MAX


def _find_separators(profile: np.ndarray, frames_gray: list[np.ndarray] | None,
                     axis: int) -> list[tuple[int, int]]:
    """
    Find separator bands along one axis of one content strip.

    Tries the static-CG-border signal first (possibly more than one band --
    see _local_minima_bands). If none qualifies and frame data is available,
    falls back to the borderless-seam search: a persistent-edge candidate,
    confirmed via temporal decorrelation so a strong static edge inside one
    real camera view doesn't get mistaken for a seam between two different
    ones. A given broadcast is consistently one style or the other, so this
    cascade -- run independently per strip -- ends up behaving consistently
    across the whole frame without needing an explicit global decision.
    """
    bands = _local_minima_bands(profile)
    if bands:
        return bands
    if frames_gray:
        idx = _gradient_ridge_seam(frames_gray, axis)
        if idx is not None and _seam_is_independent(frames_gray, axis, idx):
            return [(idx, idx)]
    return []


def find_camera_rects(range_img: np.ndarray,
                      min_fraction: float = MIN_VIEW_FRACTION,
                      frames_gray: list[np.ndarray] | None = None) -> list[dict]:
    """
    Hierarchical variance-based camera view detection.

    1. Compute row-wise variance of the range image -> find horizontal
       separator band(s) splitting the frame into row strips (see
       _find_separators for the CG-border / borderless-seam cascade).
    2. For each row strip, independently compute column-wise variance ->
       find vertical separator band(s) the same way.
    3. Return bounding rects for all resulting camera-view regions.

    `frames_gray`, if given, enables the borderless-seam fallback for
    districts that don't draw a static graphic between camera feeds; without
    it, only the static-CG-trough method runs (e.g. when reusing a
    precomputed --range-image with no frame data behind it).
    """
    h, w = range_img.shape
    total = h * w

    row_var    = np.var(range_img, axis=1)
    h_bands    = _find_separators(row_var, frames_gray, axis=0)
    row_strips = _segments_from_bands(h_bands, h)

    rects = []
    for y0, y1 in row_strips:
        strip        = range_img[y0:y1, :]
        strip_frames = [g[y0:y1, :] for g in frames_gray] if frames_gray else None

        col_var  = np.var(strip, axis=0)
        v_bands  = _find_separators(col_var, strip_frames, axis=1)
        col_segs = _segments_from_bands(v_bands, w)

        for x0, x1 in col_segs:
            area = (x1 - x0) * (y1 - y0)
            frac = area / total
            if frac >= min_fraction:
                rects.append({"box": [x0, y0, x1, y1],
                               "area": area, "fraction": round(frac, 3)})

    rects.sort(key=lambda r: -r["area"])
    return rects


# ---------------------------------------------------------------------------
# View labeling
# ---------------------------------------------------------------------------

def label_views(rects: list[dict], img_w: int, img_h: int) -> list[dict]:
    """
    Assign each detected camera rectangle a stable, opaque name.

    Sorted into reading order (top-to-bottom, then left-to-right) only so
    output is deterministic across runs. No attempt is made to classify the
    broadcast layout or guess which physical camera feed a view is --
    downstream pipeline steps treat the name as nothing more than a unique
    key.
    """
    if not rects:
        rects = [{"box": [0, 0, img_w, img_h], "area": img_w * img_h,
                  "fraction": 1.0}]
    ordered = sorted(rects, key=lambda r: (r["box"][1], r["box"][0]))
    return [{"name": f"view{i}", **r} for i, r in enumerate(ordered)]


def named_crops(views: list[dict]) -> dict:
    """Flatten views list into a {name: [x0,y0,x1,y1]} dict."""
    return {v["name"]: v["box"] for v in views}


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

_COLORS = [
    (0, 230,  0), (0, 160, 255), (255, 200, 0), (0, 100, 200), (200, 100, 255),
]

def visualize(img_bgr: np.ndarray, range_img: np.ndarray,
              views: list[dict]) -> np.ndarray:
    h, w = img_bgr.shape[:2]

    left = img_bgr.copy()
    for i, v in enumerate(views):
        x0, y0, x1, y1 = v["box"]
        color = _COLORS[i % len(_COLORS)]
        cv2.rectangle(left, (x0, y0), (x1, y1), color, 2)
        cv2.putText(left, f"{v['name']} {v['fraction']:.0%}",
                    (x0 + 6, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    cv2.putText(left, f"views={len(views)}",
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
        views     = cached["views"]
        crops     = named_crops(views)
        range_img = None
    else:
        # Accumulate frames
        frames_for_split = None   # None unless we have per-frame pixel data
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
            frames_for_split = video_grays
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
            frames_for_split = grays
            print(f"[accum] range: min={range_img.min():.1f} "
                  f"max={range_img.max():.1f} "
                  f"mean={range_img.mean():.1f}", file=sys.stderr)

        if args.save_range:
            rp = DATA_DIR / "images" / f"{event_match}_range.png"
            save_range_image(range_img, rp)
            print(f"[range] saved -> {rp}", file=sys.stderr)

        rects = find_camera_rects(range_img, args.min_fraction, frames_for_split)
        views = label_views(rects, img_w, img_h)
        crops = named_crops(views)

        print(f"[split] views={[v['name'] for v in views]}", file=sys.stderr)

        save_profile(event_match, {"views": views}, img_w, img_h)

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
        vis = visualize(ref_img, range_img, views)
        out = args.viz_out or str(ref_path.with_suffix("")) + "_split.jpg"
        cv2.imwrite(out, vis)
        print(f"[viz] {out}", file=sys.stderr)
    elif args.viz and range_img is None:
        print("[viz] skipped (using cached profile, no range image)", file=sys.stderr)


if __name__ == "__main__":
    main()
