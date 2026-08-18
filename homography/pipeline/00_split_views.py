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
4. Keep only resulting rectangles larger than MIN_VIEW_FRACTION of the
   full frame.
5. Drop any surviving rectangle whose content looks like a CG graphic
   rather than live camera footage (exclude_cg_regions): a rotating
   sponsor panel or scorebug has a background plus a handful of
   flat-colored logos/text, so its color-histogram entropy stays low in
   every sampled frame regardless of which specific graphic is showing;
   real video doesn't. What's left are the camera views.
6. Sort into a stable reading order (top-to-bottom, left-to-right) and
   assign each an opaque index name (view0, view1, ...) purely so output
   is deterministic across runs -- this is not a claim about which
   physical camera feed a view is; downstream code never inspects the name.

find_camera_rects only uses static-CG-band detection (_local_minima_bands).
A separate, NOT WIRED IN set of functions further down the file
(_activity_seam_segments / _cluster_seam_segments / _filter_seam_lines /
_connect_regions) is an in-progress replacement aimed at districts that
composite feeds edge-to-edge with no static graphic between them, or whose
seams don't span the full frame width/height -- see the comment above
_connect_regions for where that stands.

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

# Ignore this fraction of a strip at each edge when scanning for static-CG
# troughs (_local_minima_bands) -- broadcast letterboxing/edge artifacts live
# right at the frame border and shouldn't be mistaken for a real separator.
BAND_EDGE_FRAC = 0.05

# A local minimum only qualifies as a static-CG trough if it's below this
# fraction of the profile's peak. Measured on match1: the true separator
# bottoms out at ~0.08-0.1% of peak, but a scoreboard's score/clock digits
# -- which change value over the match even though the bar they sit in is
# positionally static -- locally spike row_var, splitting the bar into two
# spurious troughs at ~1.3% and ~3.4% of peak. 0.05 let both through; 0.02
# sits with comfortable margin below the false troughs and >10x above the
# real one.
BAND_MAX_TROUGH_FRAC = 0.02

# --- Seam content/independence checks (used by _filter_seam_lines) ---
# A candidate edge is only a compositing cut if the two sides are genuinely
# different footage. A strong line INSIDE one continuous camera view -- a field
# center line, a guardrail -- produces an identical edge, and on a locked wide
# shot temporal independence can't reject it either (the two halves of one field
# have independent local motion). The static scene is what tells them apart:
# across a real cut the two sides are unrelated, across a field line the SAME
# scene continues (carpet/brightness match). Measured on real footage a field
# center line scored ~0.03-0.09; genuine borderless seams scored ~0.39-0.68.
# Known limit: two adjacent feeds of near-identical content (e.g. two field
# cams) would also score low -- not seen in practice, and inherently ambiguous.
SEAM_CORR_MAX       = 0.6   # frame-to-frame delta correlation across the seam must be below this
SEAM_MARGIN_PX      = 4     # strip width sampled on each side of a candidate seam
SEAM_MIN_FRAMES     = 8     # too few frames makes the correlation meaningless
CONTENT_DISSIM_MIN  = 0.25  # 1 - corr(mean-image strips across the seam)
CONTENT_DISSIM_HALF = 20    # px of static content sampled each side
CONTENT_DISSIM_GAP  = 4     # px skipped at the seam so a painted line can't inflate it

# --- CG-region classification (color concentration) ---
# Below this per-frame color-histogram entropy (bits), a region is treated
# as a CG graphic rather than live camera content. Validated against one
# real match: a confirmed CG panel measured ~3.6 bits, confirmed camera
# views measured ~9.5-9.8 bits, and a real camera view carrying a partial
# CG overlay (correctly NOT excluded) measured ~7.9-8.0 bits. This
# threshold sits comfortably below all three seen so far; it hasn't been
# validated against a genuinely rotating CG panel, where averaging color
# across several different graphic states per sampled frame could
# plausibly land higher than a single static graphic would.
CG_ENTROPY_THRESHOLD = 6.0
CG_HIST_BINS         = 24


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
                        end_frac: float = 0.75
                        ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    Sample n evenly-spaced frames from the middle portion of a local video.
    Returns (grays, bgrs, ref_bgr): grayscale frames for the temporal/edge
    detection, the same frames in color for CG-region classification, and
    the middle sampled frame (again in color) for crops/visualization.
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
    grays, bgrs = [], []
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
        bgrs.append(frame)
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if ref_bgr is None and grays:
        # fallback: re-read the middle index
        cap2 = cv2.VideoCapture(video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        _, ref_bgr = cap2.read()
        cap2.release()
    return grays, bgrs, ref_bgr


def load_gray_and_bgr(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"[error] cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img


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
                        edge_frac: float = BAND_EDGE_FRAC,
                        max_trough_frac: float = BAND_MAX_TROUGH_FRAC) -> list[tuple[int, int]]:
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


def find_camera_rects(range_img: np.ndarray,
                      min_fraction: float = MIN_VIEW_FRACTION,
                      frames_gray: list[np.ndarray] | None = None) -> list[dict]:
    """
    Hierarchical variance-based camera view detection, using static-CG-band
    detection only (_local_minima_bands) -- no seam/ridge search involved.

    1. Compute row-wise variance of the range image -> find horizontal
       separator band(s) (troughs) splitting the frame into row strips.
    2. For each row strip, independently compute column-wise variance ->
       find vertical separator band(s) the same way.
    3. Return bounding rects for all resulting camera-view regions.

    `frames_gray` is accepted for call-site compatibility but unused here --
    band detection needs only the accumulated range image.

    LIMITATION: this is a guillotine and every separator must span the full
    width/height of its strip. A layout whose seams don't (e.g. a bottom row
    with a center CG panel vertically offset from its neighbours -- match8),
    or that composites feeds edge-to-edge with no static graphic between them
    at all, has no full-span trough, so the affected split is missed and the
    frame under-segments. The DETECT/CLUSTER/FILTER/CONNECT functions further
    below are an unwired experimental replacement aimed at that case.
    """
    h, w = range_img.shape
    total = h * w

    row_var    = np.var(range_img, axis=1)
    h_bands    = _local_minima_bands(row_var)
    row_strips = _segments_from_bands(h_bands, h)

    rects = []
    for y0, y1 in row_strips:
        strip = range_img[y0:y1, :]

        col_var  = np.var(strip, axis=0)
        v_bands  = _local_minima_bands(col_var)
        col_segs = _segments_from_bands(v_bands, w)

        for x0, x1 in col_segs:
            area = (x1 - x0) * (y1 - y0)
            frac = area / total
            if frac >= min_fraction:
                rects.append({"box": [x0, y0, x1, y1],
                               "area": area, "fraction": round(frac, 3)})

    rects.sort(key=lambda r: -r["area"])
    return rects


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


def _mean_image(frames_gray: list[np.ndarray]) -> np.ndarray:
    """Per-pixel temporal mean -- the static scene with moving content (balls,
    robots, the switcher's live feed) averaged away. Accumulated incrementally
    to avoid stacking every frame in memory at once."""
    acc = frames_gray[0].astype(np.float32).copy()
    for f in frames_gray[1:]:
        acc += f
    return acc / len(frames_gray)


def _seam_content_dissimilarity(mean_gray: np.ndarray, axis: int, idx: int,
                                half: int = CONTENT_DISSIM_HALF,
                                gap: int = CONTENT_DISSIM_GAP) -> float:
    """
    1 - correlation between the static content just past each side of a
    candidate seam. High -> the two sides are unrelated footage (a real cut).
    Low -> the same continuous scene spans the seam, i.e. the "seam" is a field
    line/guardrail painted on one camera's view (see CONTENT_DISSIM_MIN).

    The +-gap skip excludes the edge pixels themselves, so a bright painted line
    can't inflate the score; the sides are compared as 1-D profiles along the
    seam, which stays discriminative even when both sides are similarly lit.
    """
    h, w = mean_gray.shape
    a0, a1, b0, b1 = idx - gap - half, idx - gap, idx + gap, idx + gap + half
    if axis == 0:
        if a0 < 0 or b1 > h:
            return 0.0
        A = mean_gray[a0:a1, :].mean(axis=0)
        B = mean_gray[b0:b1, :].mean(axis=0)
    else:
        if a0 < 0 or b1 > w:
            return 0.0
        A = mean_gray[:, a0:a1].mean(axis=1)
        B = mean_gray[:, b0:b1].mean(axis=1)
    if A.std() < 1e-6 or B.std() < 1e-6:
        return 0.0
    return 1.0 - float(np.corrcoef(A, B)[0, 1])


# --- FILTER: reject seam-line candidates that aren't real compositing cuts ---
# _activity_seam_segments finds every persistent axis-aligned edge, including
# false positives it can't itself tell apart from a real seam: a strong
# static edge inside one continuous view (a guardrail, a scoring-structure
# edge, a field line) or a CG graphic's internal divider. _seam_is_independent
# and _seam_content_dissimilarity already discriminate a real compositing cut
# from a strong-but-internal edge -- they were built for the old full-span
# ridge detector -- and are reused here unchanged, but evaluated only across
# each line's own covered interval(s), stitched together, rather than the
# whole frame: a match8-style partial seam has no meaningful signal outside
# where it's actually present.
SEAM_MIN_STITCHED_LEN = 20  # px of covered interval needed to trust the stats


def _stitch_along_intervals(arr: np.ndarray, axis: int,
                            intervals: list[tuple[int, int]]) -> np.ndarray:
    """Concatenate the covered slices of `arr` along the seam direction,
    dropping uncovered gaps (a CG graphic mid-line, an NMS hole) so the
    content checks only see genuine seam-adjacent pixels."""
    if axis == 0:
        parts = [arr[:, s:e] for s, e in intervals if e > s]
        return np.concatenate(parts, axis=1) if parts else arr[:, :0]
    parts = [arr[s:e, :] for s, e in intervals if e > s]
    return np.concatenate(parts, axis=0) if parts else arr[:0, :]


def _filter_seam_lines(lines: list[dict], frames_gray: list[np.ndarray],
                       mean_gray: np.ndarray, axis: int) -> list[dict]:
    """Keep only lines that pass the real-seam content checks, evaluated over
    each line's own covered span rather than assuming a full-frame span."""
    kept = []
    for line in lines:
        if line["coverage"] < SEAM_MIN_STITCHED_LEN:
            continue
        idx = line["pos"]
        mean_strip = _stitch_along_intervals(mean_gray, axis, line["intervals"])
        if _seam_content_dissimilarity(mean_strip, axis, idx) < CONTENT_DISSIM_MIN:
            continue
        frame_strips = [_stitch_along_intervals(g, axis, line["intervals"])
                        for g in frames_gray]
        if not _seam_is_independent(frame_strips, axis, idx):
            continue
        kept.append(line)
    return kept


# --- CONNECT: assemble surviving lines into camera-view rectangles ---
# A surviving line may only wall off PART of the frame width/height (match8's
# vertically-offset center panel), so a guillotine cut -- which must span the
# full strip -- can't place it. Instead: lay a grid over the frame at every
# surviving line's position on both axes, then merge adjacent grid cells back
# together wherever no surviving line actually covers their shared boundary --
# a wall only separates the cells it's actually documented to run between.
WALL_POS_MERGE_PX  = 6    # candidate positions this close are one wall, not two
WALL_MIN_COVER_FRAC = 0.6  # fraction of a cell boundary a wall must cover to separate the cells


def _grid_positions(line_positions: list[int], length: int,
                    tol: int = WALL_POS_MERGE_PX) -> list[int]:
    """Sorted, deduped interior grid lines for one axis, plus the frame's own
    two edges. Interior positions within `tol` of an edge are dropped rather
    than merged into it, so the true 0/length endpoints are never lost to a
    nearby line snapping onto them."""
    interior = sorted(p for p in line_positions if tol < p < length - tol)
    merged = []
    for p in interior:
        if not merged or p - merged[-1] > tol:
            merged.append(p)
    return [0] + merged + [length]


def _wall_covers(line_group: list[dict], span: tuple[int, int],
                 min_frac: float = WALL_MIN_COVER_FRAC) -> bool:
    """True if the combined covered interval(s) of every line in
    `line_group` span at least `min_frac` of a grid cell boundary."""
    s0, s1 = span
    span_len = s1 - s0
    if span_len <= 0:
        return False
    all_intervals = [iv for line in line_group for iv in line["intervals"]]
    if not all_intervals:
        return False
    covered = sum(max(0, min(e, s1) - max(s, s0)) for s, e in _merge_bands(all_intervals))
    return covered / span_len >= min_frac


def _connect_regions(h_lines: list[dict], v_lines: list[dict],
                     img_w: int, img_h: int) -> list[tuple[int, int, int, int]]:
    """
    Grid + union-find: cut the frame into a grid at every surviving line's
    position on both axes, then union adjacent cells whenever no surviving
    line actually walls off their shared boundary. Returns bounding boxes of
    the resulting cell groups -- the camera-view rectangles.
    """
    ys = _grid_positions([l["pos"] for l in h_lines], img_h)
    xs = _grid_positions([l["pos"] for l in v_lines], img_w)
    nrows, ncols = len(ys) - 1, len(xs) - 1

    parent = list(range(nrows * ncols))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    def cell(r, c):
        return r * ncols + c

    def lines_at(lines, pos):
        return [l for l in lines if abs(l["pos"] - pos) <= WALL_POS_MERGE_PX]

    for r in range(nrows):
        for c in range(ncols - 1):
            group = lines_at(v_lines, xs[c + 1])
            if not _wall_covers(group, (ys[r], ys[r + 1])):
                union(cell(r, c), cell(r, c + 1))
    for c in range(ncols):
        for r in range(nrows - 1):
            group = lines_at(h_lines, ys[r + 1])
            if not _wall_covers(group, (xs[c], xs[c + 1])):
                union(cell(r, c), cell(r + 1, c))

    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(nrows):
        for c in range(ncols):
            groups.setdefault(find(cell(r, c)), []).append((r, c))

    boxes = []
    for cells in groups.values():
        r0 = min(r for r, c in cells); r1 = max(r for r, c in cells) + 1
        c0 = min(c for r, c in cells); c1 = max(c for r, c in cells) + 1
        boxes.append((xs[c0], ys[r0], xs[c1], ys[r1]))
    return boxes


# NOT WIRED IN, and parked for now -- see "Status" below before picking this
# back up. find_camera_rects (above, near the top of the file) is the active
# detector and uses only _local_minima_bands, which requires a static CG
# border between camera feeds. That's the official FIRST guideline and
# essentially every district/event follows it -- Texas is the one exception
# seen so far, which is the whole reason this experimental path exists. This
# is therefore a minority-of-events problem, not a general fix; time-box any
# further work on it accordingly.
#
# Pipeline: DETECT (_activity_seam_segments) / CLUSTER (_cluster_seam_segments)
# / FILTER (_filter_seam_lines) / CONNECT (_connect_regions). DETECT+CLUSTER
# reliably recover every real seam, including match8-style seams that don't
# span the full frame (a CG panel vertically offset from the two camera feeds
# flanking it -- see ground-truth pixel coords for match7/match8 in git log /
# conversation history if picking this back up). FILTER and CONNECT do not
# yet reliably turn that into correct rectangles. Concretely tried and ruled
# out:
#
#   1. _filter_seam_lines as currently written (content-dissimilarity +
#      temporal-independence, evaluated over each line's own stitched
#      covered interval) keeps far too many false positives once you look at
#      the FULL candidate set rather than a few hand-picked examples: on
#      match7/match8, every ground-truth-real line does survive, but so do
#      roughly half of everything else (e.g. match8 h-axis: 23 candidates,
#      14 survive FILTER, only 3 are real). Neither signal separates real
#      from false on its own -- false lines routinely score HIGHER on
#      content-dissimilarity than real ones (match8: false pos=280 scores
#      1.211 vs real pos=713's 1.062), and temporal-independence reads True
#      for most lines regardless of truth (15/23 on match8 h-axis). Raw
#      `coverage` doesn't separate them either -- a false line can have more
#      covered length than a real one (match8 v-axis: false pos=1032 at 404px
#      vs real pos=1266 at 325px). The specific failure mode confirmed on
#      match8: a physical arena guardrail crossing most of the frame width
#      (a static, near-motionless object, same class of false positive as
#      the guardrail/rope found in match1/match4 -- see conversation history)
#      registers as pos=657, coverage=850, dissim=0.825 (high!) -- only
#      _seam_is_independent (indep=False) correctly rejects it, and that's
#      not something you can rely on in general per the point above.
#
#   2. Region-level content comparison as a replacement for per-line
#      filtering (build the grid from ALL candidate lines, unfiltered, then
#      decide cell-to-cell merges via _seam_is_independent/
#      _seam_content_dissimilarity-style checks applied to the two whole
#      neighboring grid cells instead of a thin strip around one line) --
#      the idea being that a real seam's two sides should be robustly
#      distinguishable using their full area, not just a narrow margin.
#      Tested directly on hand-picked large boxes (e.g. match7's main view
#      vs. bot_left cam) and the checks work great in isolation (dissim=0.94,
#      clearly different). But wired into the full grid+union-find CONNECT,
#      the result collapsed to just 1-2 regions instead of 3-4: with dozens
#      of candidate lines the grid fragments into many cells, several
#      inevitably thin (<10-15px) on one side, too thin to compare reliably;
#      auto-merging those (the only sane default) plus even one or two
#      marginal same-content calls elsewhere is enough for union-find
#      transitivity to silently bridge two genuinely different regions
#      together through a chain of "uncertain, so merge" steps -- one bad
#      link anywhere in the chain glues everything past it into one blob.
#      This is a real structural problem with cell-by-cell grid merging, not
#      a threshold-tuning issue.
#
#   3. Hierarchical linear strip-merge (same region-comparison idea, but
#      row-strips using the FULL frame width, then column-strips using the
#      full height of each surviving row-strip -- i.e. avoid 2D grid cells
#      and their thin-sliver/transitivity problem entirely by only ever
#      comparing large, full-span strips, closer in spirit to the original
#      guillotine). This avoids the thin-cell problem by construction, but
#      initial testing on match7 showed it *still* failed to merge adjacent
#      strips that are obviously part of the same real camera view (e.g.
#      row(0,129) vs row(129,184), both well within the top scoreboard/main
#      view before the real y=254 cut) -- investigation of why was cut short
#      here. Worth checking first if this gets picked back up: whether
#      SEAM_CORR_MAX/CONTENT_DISSIM_MIN (tuned for thin strips beside a line)
#      are simply too permissive at this coarser, full-strip scale, before
#      concluding the approach itself doesn't work.
#
# Until FILTER/CONNECT are fixed, call the individual DETECT/CLUSTER
# functions directly if needed (see visualize_seam_lines / --viz-seams for a
# debug view of what they currently detect).


# ---------------------------------------------------------------------------
# CG-region classification (color concentration)
# ---------------------------------------------------------------------------

def _color_entropy(frames_bgr: list[np.ndarray], box: list[int]) -> float:
    """
    Mean per-frame color-histogram entropy (bits) over a region.

    Low entropy = a few dominant colors (CG-like -- a graphic's flat-color
    background plus a handful of logo/text colors). High entropy = broadly
    distributed colors (camera-like -- a real scene has no such small
    palette). Computed per frame and averaged, not on an accumulated image,
    so a ROTATING panel still reads as low-entropy: whichever specific
    graphic is showing in a given frame is itself low-entropy, even though
    the graphic differs frame to frame.
    """
    x0, y0, x1, y1 = box
    entropies = []
    for f in frames_bgr:
        crop = f[y0:y1, x0:x1]
        hist = cv2.calcHist([crop], [0, 1, 2], None,
                            [CG_HIST_BINS] * 3, [0, 256] * 3).flatten()
        total = hist.sum()
        if total == 0:
            continue
        p = hist / total
        p_nz = p[p > 0]
        entropies.append(float(-np.sum(p_nz * np.log2(p_nz))))
    return float(np.mean(entropies)) if entropies else 0.0


def exclude_cg_regions(rects: list[dict],
                       frames_bgr: list[np.ndarray] | None) -> tuple[list[dict], list[dict]]:
    """
    Split detected rectangles into (camera views, CG-like regions).

    Runs after find_camera_rects -- a region reaching here already passed
    the geometric/temporal detection, so this is purely about content: does
    it look like a graphic (a rotating sponsor panel, a scorebug) rather
    than live footage? See _color_entropy. This is the mechanism that
    excludes a rotating CG panel the geometric detectors can't tell apart
    from a real view any other way.

    `frames_bgr` is required for this; without it (e.g. reusing a cached
    profile with no frame data), nothing gets excluded here.
    """
    if not frames_bgr:
        return rects, []
    kept, excluded = [], []
    for r in rects:
        ent = _color_entropy(frames_bgr, r["box"])
        if ent < CG_ENTROPY_THRESHOLD:
            excluded.append({**r, "entropy": round(ent, 2)})
        else:
            kept.append(r)
    return kept, excluded


# ---------------------------------------------------------------------------
# Seam-segment DETECT + CLUSTER -- used by find_camera_rects above
# ---------------------------------------------------------------------------
#
# The old approach was a two-level guillotine: a full-width horizontal dip ->
# row strips, then a full-height dip within each strip, plus a separate
# full-span ridge search for borderless cuts. It assumed every seam runs
# edge-to-edge of the frame (true for most broadcasts) and so could not
# represent a layout whose seams don't span the frame -- e.g. a bottom row of
# feeds with a center CG panel offset vertically from its neighbours
# (match8): there is no single horizontal line across the full width, so the
# row pass found nothing there and the frame under-segmented.
#
# The functions below treat the temporal range image as an "activity graph"
# and find seams as axis-aligned edge SEGMENTS in it, so a seam is allowed to
# stop partway across the frame (it just ends where it meets a perpendicular
# seam). One detector recovers both a static-CG-band border and a borderless
# compositing cut -- both produce a persistent edge in the activity image --
# where the old code needed two separate mechanisms. Validated on match8:
# recovers every real cut (including the interrupted main/bottom border the
# old guillotine missed), plus noise that _filter_seam_lines rejects.

def _contiguous_runs(indices: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Group sorted indices into (start, end) runs, bridging holes <= `gap` --
    a ball crossing a seam blanks a few pixels of it and shouldn't split it."""
    if len(indices) == 0:
        return []
    runs = []
    start = prev = int(indices[0])
    for i in indices[1:]:
        i = int(i)
        if i - prev <= gap:
            prev = i
        else:
            runs.append((start, prev)); start = prev = i
    runs.append((start, prev))
    return runs


def _activity_seam_segments(range_img: np.ndarray, axis: int,
                            smooth: int = 121, min_len: int = 140,
                            thr_pct: float = 97.0, nms: int = 11,
                            gap: int = 40) -> list[tuple[int, int, int, int]]:
    """
    Detect axis-aligned seam SEGMENTS in the activity (temporal-range) image.
    `axis=0` finds horizontal seams, `axis=1` vertical. Returns (x0,y0,x1,y1).

    The trick that makes this work on real footage is directional smoothing:
    average the activity ALONG the seam direction first. A composite seam is
    constant along its length, so it survives the averaging; ball/robot texture
    is random and cancels. Only then take the perpendicular gradient, so the
    strong responses are the seams rather than the thousands of moving balls.
    Then non-max-suppress across the perpendicular axis to thin each seam to one
    line, and keep contiguous runs longer than `min_len` (partial seams are
    fine -- that is the whole point vs. the full-span guillotine).

    Validated on match8: recovers every real cut, but also picks up scorebug
    text, arena rails, etc. -- filtering those out is _filter_seam_lines's job.
    """
    A = range_img
    if axis == 0:
        blurred = cv2.blur(A, (smooth, 1))                       # cancel texture along x
        edge = cv2.blur(np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)), (smooth, 1))
        keep = (edge > np.percentile(edge, thr_pct)) & \
               (edge >= cv2.dilate(edge, np.ones((nms, 1), np.uint8)) - 1e-3)
        segs = []
        for y in range(A.shape[0]):
            for s, e in _contiguous_runs(np.where(keep[y])[0], gap):
                if e - s >= min_len:
                    segs.append((s, y, e, y))
        return segs
    blurred = cv2.blur(A, (1, smooth))                           # cancel texture along y
    edge = cv2.blur(np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)), (1, smooth))
    keep = (edge > np.percentile(edge, thr_pct)) & \
           (edge >= cv2.dilate(edge, np.ones((1, nms), np.uint8)) - 1e-3)
    segs = []
    for x in range(A.shape[1]):
        for s, e in _contiguous_runs(np.where(keep[:, x])[0], gap):
            if e - s >= min_len:
                segs.append((x, s, x, e))
    return segs


def _cluster_seam_segments(segments: list[tuple[int, int, int, int]], axis: int,
                           tol: int = 12, merge_gap: int = 30) -> list[dict]:
    """
    Group segments sharing a row (axis=0) or column (axis=1) into cut LINES.
    Each line carries its position, the merged coverage intervals along the
    seam, and the total covered length. A real cut is a long, well-covered line;
    scene texture is many short scattered ones -- but a full-width scorebug
    also produces long, well-covered INTERNAL lines, so coverage alone isn't
    enough to filter; that's what _filter_seam_lines's content checks are for.
    """
    pos_i = 1 if axis == 0 else 0
    a_i, b_i = (0, 2) if axis == 0 else (1, 3)
    lines: list[dict] = []
    for seg in sorted(segments, key=lambda s: s[pos_i]):
        for L in lines:
            if abs(L["pos"] - seg[pos_i]) <= tol:
                L["intervals"].append((seg[a_i], seg[b_i]))
                L["pos"] = (L["pos"] * L["n"] + seg[pos_i]) / (L["n"] + 1)
                L["n"] += 1
                break
        else:
            lines.append({"pos": float(seg[pos_i]),
                          "intervals": [(seg[a_i], seg[b_i])], "n": 1})
    for L in lines:
        L["pos"] = int(round(L["pos"]))
        merged: list[list[int]] = []
        for s, e in sorted(L["intervals"]):
            if merged and s <= merged[-1][1] + merge_gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        L["intervals"] = [(s, e) for s, e in merged]
        L["coverage"] = sum(e - s for s, e in L["intervals"])
    return lines


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


def visualize_seam_lines(img_bgr: np.ndarray,
                         h_lines: list[dict], v_lines: list[dict],
                         h_kept: list[dict], v_kept: list[dict]) -> np.ndarray:
    """
    Debug view of the DETECT+CLUSTER+FILTER stages: every clustered seam line
    drawn over its own covered interval(s) -- not a full-width/height line --
    so a partial seam's true extent is visible. Green = survived
    _filter_seam_lines; red = rejected. Thickness scales with `coverage` so
    long, confident candidates are easy to pick out from short, noisy ones.
    """
    vis = img_bgr.copy()
    kept_h_ids = {id(l) for l in h_kept}
    kept_v_ids = {id(l) for l in v_kept}
    for lines, kept_ids, axis in [(h_lines, kept_h_ids, 0), (v_lines, kept_v_ids, 1)]:
        for line in lines:
            passed = id(line) in kept_ids
            color = (0, 220, 0) if passed else (0, 0, 230)
            thickness = 1 + min(6, line["coverage"] // 200)
            pos = line["pos"]
            for s, e in line["intervals"]:
                pt1, pt2 = ((s, pos), (e, pos)) if axis == 0 else ((pos, s), (pos, e))
                cv2.line(vis, pt1, pt2, color, thickness)
            label_at = (line["intervals"][0][0] + 4, pos - 4) if axis == 0 \
                       else (pos + 4, line["intervals"][0][0] + 12)
            cv2.putText(vis, f"{pos}|{line['coverage']}", label_at,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return vis


def _plot_profile_vertical(profile: np.ndarray, plot_h: int, width: int,
                           bands: list[tuple[int, int]], label: str) -> np.ndarray:
    """
    Render a 1D per-row profile (e.g. row_var) as a vertical line plot --
    row index on the vertical axis, activity value on the horizontal axis --
    resampled to `plot_h` so it lines up row-for-row with the frame/activity
    panels once those are scaled to the same height. Detected bands are
    shaded behind the trace so a trough's position is visible relative to
    what _local_minima_bands actually grew it into.
    """
    idx = (np.linspace(0, len(profile) - 1, plot_h)).astype(np.int32)
    resampled = profile[idx]
    plot = np.full((plot_h, width, 3), 255, dtype=np.uint8)
    for s, e in bands:
        s2, e2 = int(s / len(profile) * plot_h), int(e / len(profile) * plot_h)
        cv2.rectangle(plot, (0, s2), (width - 1, min(e2, plot_h - 1)), (225, 225, 255), -1)
    pmax = float(resampled.max()) or 1.0
    xs = np.clip((resampled / pmax * (width - 12)).astype(np.int32), 0, width - 1)
    pts = np.column_stack([xs, np.arange(plot_h)]).astype(np.int32)
    cv2.polylines(plot, [pts], isClosed=False, color=(180, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(plot, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return plot


def visualize_bands(img_bgr: np.ndarray, range_img: np.ndarray) -> np.ndarray:
    """
    Debug view of the static-CG-band detection (_local_minima_bands) that
    find_camera_rects actually uses: the live frame, the activity (temporal-
    range) image, and a plot of the global row-activity profile (row_var),
    all side by side. Unlike seam lines, a band is a guillotine cut -- it
    always spans the FULL width (h_bands, red) or the full height of the row
    strip it was found within (v_bands, magenta), never a partial interval,
    so each is drawn as a translucent strip rather than a line. The plot
    panel shows the exact per-row signal _local_minima_bands scans for
    troughs in, with the same bands shaded behind the trace.
    """
    h, w = range_img.shape

    scaled = np.clip(range_img / max(range_img.max(), 1) * 255, 0, 255).astype(np.uint8)
    frame_vis = img_bgr.copy()
    range_vis = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)

    def draw_band(y0: int, y1: int, x0: int, x1: int, color: tuple):
        for vis in (frame_vis, range_vis):
            overlay = vis.copy()
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, -1)
            vis[:] = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    row_var = np.var(range_img, axis=1)
    h_bands = _local_minima_bands(row_var)
    for s, e in h_bands:
        draw_band(s, e + 1, 0, w, (0, 0, 255))            # red: row-axis bands

    row_strips = _segments_from_bands(h_bands, h)
    total_v_bands = 0
    for y0, y1 in row_strips:
        strip   = range_img[y0:y1, :]
        col_var = np.var(strip, axis=0)
        v_bands = _local_minima_bands(col_var)
        total_v_bands += len(v_bands)
        for s, e in v_bands:
            draw_band(y0, y1, s, e + 1, (255, 0, 255))    # magenta: col-axis bands within this strip

    for vis in (frame_vis, range_vis):
        cv2.putText(vis, f"h_bands={len(h_bands)} v_bands={total_v_bands}",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    scale  = 0.5
    plot_w = 260
    frame_s = cv2.resize(frame_vis, (int(w * scale), int(h * scale)))
    range_s = cv2.resize(range_vis, (int(w * scale), int(h * scale)))
    plot_s  = _plot_profile_vertical(row_var, int(h * scale), plot_w, h_bands,
                                     "row activity (row_var)")
    return np.hstack([frame_s, range_s, plot_s])


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
    ap.add_argument("--viz-seams", action="store_true",
                    help="save a debug image of every DETECT+CLUSTER seam line "
                         "over its covered interval(s), green=passed FILTER / "
                         "red=rejected, instead of running full view detection")
    ap.add_argument("--viz-bands", action="store_true",
                    help="save a debug image of the static-CG-band detection "
                         "(_local_minima_bands) find_camera_rects actually uses, "
                         "on both the frame and the activity image, instead of "
                         "running full view detection")
    ap.add_argument("--viz-out", metavar="PATH")
    args = ap.parse_args()

    # ---- resolve event_match and frame list ----
    video_grays = None   # set when --video is used
    video_bgrs  = None
    if args.video:
        video_path  = args.video
        event_match = pathlib.Path(video_path).stem
        video_grays, video_bgrs, ref_img = sample_video_frames(video_path, args.n_frames)
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
        frames_bgr       = None   # None unless we have per-frame color data
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
            frames_bgr       = video_bgrs
            print(f"[accum] range: min={range_img.min():.1f} "
                  f"max={range_img.max():.1f} "
                  f"mean={range_img.mean():.1f}", file=sys.stderr)
        else:
            print(f"[accum] loading {len(frame_indices)} frames ...", file=sys.stderr)
            grays, bgrs = [], []
            for fi in frame_indices:
                try:
                    g, b = load_gray_and_bgr(fetch_frame(event_match, fi))
                    grays.append(g)
                    bgrs.append(b)
                except Exception as e:
                    print(f"[warn] frame {fi} failed: {e}", file=sys.stderr)
            if not grays:
                sys.exit("[error] no frames loaded")
            range_img = accumulate_range(grays)
            frames_for_split = grays
            frames_bgr       = bgrs
            print(f"[accum] range: min={range_img.min():.1f} "
                  f"max={range_img.max():.1f} "
                  f"mean={range_img.mean():.1f}", file=sys.stderr)

        if args.save_range:
            rp = DATA_DIR / "images" / f"{event_match}_range.png"
            save_range_image(range_img, rp)
            print(f"[range] saved -> {rp}", file=sys.stderr)

        if args.viz_seams:
            h_lines = _cluster_seam_segments(_activity_seam_segments(range_img, axis=0), axis=0)
            v_lines = _cluster_seam_segments(_activity_seam_segments(range_img, axis=1), axis=1)
            if frames_for_split:
                mean_gray = _mean_image(frames_for_split)
                h_kept = _filter_seam_lines(h_lines, frames_for_split, mean_gray, axis=0)
                v_kept = _filter_seam_lines(v_lines, frames_for_split, mean_gray, axis=1)
            else:
                h_kept, v_kept = h_lines, v_lines
            print(f"[seams] h_lines={len(h_lines)} (kept {len(h_kept)})  "
                  f"v_lines={len(v_lines)} (kept {len(v_kept)})", file=sys.stderr)
            vis = visualize_seam_lines(ref_img, h_lines, v_lines, h_kept, v_kept)
            out = args.viz_out or str(ref_path.with_suffix("")) + "_seams.jpg"
            cv2.imwrite(out, vis)
            print(f"[viz-seams] {out}", file=sys.stderr)
            return

        if args.viz_bands:
            vis = visualize_bands(ref_img, range_img)
            out = args.viz_out or str(ref_path.with_suffix("")) + "_bands.jpg"
            cv2.imwrite(out, vis)
            print(f"[viz-bands] {out}", file=sys.stderr)
            return

        rects = find_camera_rects(range_img, args.min_fraction, frames_for_split)
        rects, excluded_cg = exclude_cg_regions(rects, frames_bgr)
        if excluded_cg:
            print(f"[cg] excluded {len(excluded_cg)} CG-like region(s): "
                  + ", ".join(f"{r['box']} entropy={r['entropy']}" for r in excluded_cg),
                  file=sys.stderr)

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
