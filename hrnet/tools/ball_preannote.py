#!/usr/bin/env python3
"""
Yellow-ball heuristic pre-annotation — project 272005.

Lists R2 objects directly, samples randomly, fetches public images from
assets.markwu.org, runs an HSV yellow-ball detector, and imports tasks
with embedded predictions in one shot (no task IDs needed).

Usage:
  python ball_preannote.py                  # 200 random frames
  python ball_preannote.py -n 50            # fewer frames
  python ball_preannote.py -n 20 --dry-run  # preview detections, no upload
"""
import os, pathlib, argparse, itertools
import numpy as np
import cv2
import requests
import sys as _sys
_MODULE_ROOT = pathlib.Path(__file__).parent.parent
_sys.path.insert(0, str(_MODULE_ROOT))

from dotenv import load_dotenv
from ls_ext import LabelStudioClient

load_dotenv(_MODULE_ROOT / ".env")

# ── Label Studio ───────────────────────────────────────────────────────────
PROJECT_ID = 272005
LS_URL     = "https://app.humansignal.com"
MODEL_VER  = "yellow-hsv-v2"
LABEL_NAME = "Fuel"
FROM_NAME  = "label"
TO_NAME    = "image"

# ── public URL base (tasks already exist in LS) ────────────────────────────
PUBLIC_BASE = "https://assets.markwu.org"

# ── HSV yellow (OpenCV H 0–179, S/V 0–255) ────────────────────────────────
HSV_LO = np.array([ 18, 160, 140], dtype=np.uint8)   # high S+V: real balls are fluorescent
HSV_HI = np.array([ 42, 255, 255], dtype=np.uint8)

# ── Ball size gates (pixels, after squaring) ───────────────────────────────
# From labeled data on 1280×720: isolated balls are ~5–30 px on a side
BALL_MIN_PX     = 5     # smaller = noise
BALL_MAX_PX     = 35    # larger  = cluster or field element
MAX_ASPECT      = 1.6   # h/w in pixels after squaring; blobs taller than this are sus
CIRCULARITY_MIN = 0.70  # isolated ball caps score ~0.75+; touching blobs drop below this
SIZE_IQR_FACTOR = 1.8   # keep only blobs within IQR*factor of Q1-Q3 per frame

# Crop scoreboard strip at top
SCOREBOARD_FRAC = 0.17

DEFAULT_N = 200
# ──────────────────────────────────────────────────────────────────────────


def fetch_image(url: str):
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        arr = np.frombuffer(r.content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  [fetch] {url}: {e}")
        return None


def detect_balls(img_bgr: np.ndarray) -> list[tuple]:
    """Return (x_pct, y_pct, w_pct, h_pct) for each isolated ball."""
    H, W = img_bgr.shape[:2]
    crop_top = int(H * SCOREBOARD_FRAC)
    field = img_bgr[crop_top:, :]

    hsv  = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, closed=True)
        if perim == 0:
            continue
        # Isolated ball top is a bright circular cap — high circularity.
        # Clusters merge blobs into lumpy shapes that score much lower.
        if 4 * np.pi * area / (perim ** 2) < CIRCULARITY_MIN:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        side = max(w, h)

        # Absolute size gate before any squaring
        if not (BALL_MIN_PX <= side <= BALL_MAX_PX):
            continue

        # Pre-squaring shape check on the raw contour bounding rect.
        # An isolated ball cap is roughly as wide as it is tall (w/h ≈ 1).
        # A horizontal cluster is wide and flat (w >> h); reject those.
        if w > 0 and h > 0 and (w / h > 1.5 or h / w > 2.0):
            continue

        # Dark-crescent check: the ball's shadow occupies the bottom ~30%
        # of the extended box. If that region is also bright, this is a
        # field marking or graphic rather than a ball.
        crescent_y0 = y + side          # bottom of bright blob
        crescent_y1 = min(crescent_y0 + max(side // 3, 2), field.shape[0])
        if crescent_y1 > crescent_y0:
            crescent_roi = field[crescent_y0:crescent_y1, x:x + w]
            if crescent_roi.size > 0:
                mean_v = cv2.cvtColor(crescent_roi, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
                if mean_v > 160:   # too bright below → not a ball shadow
                    continue

        # Square the box + small upward pad for the missed bright top
        pad_top = max(2, side // 8)   # slightly larger fraction catches small-ball tops
        y_abs = max(0, y + crop_top - pad_top)
        box_h = min(side + pad_top, H - y_abs)

        raw.append((x, y_abs, side, box_h, round(4 * np.pi * area / perim ** 2, 3)))

    if not raw:
        return []

    # Per-frame size-consistency filter using IQR.
    sizes = np.array([w for x, y, w, h, _ in raw], dtype=float)
    q1, q3 = np.percentile(sizes, [25, 75])
    iqr = q3 - q1
    lo = max(BALL_MIN_PX, q1 - SIZE_IQR_FACTOR * iqr)
    hi = min(BALL_MAX_PX, q3 + SIZE_IQR_FACTOR * iqr)

    balls = []
    for x, y_abs, w, h, circ in raw:
        if not (lo <= w <= hi):
            continue
        if h > 0 and w / h < (1 / MAX_ASPECT):
            continue
        balls.append({
            "x": x / W * 100, "y": y_abs / H * 100,
            "w": w / W * 100, "h": h / H * 100,
            "side_px": w, "circ": circ,
        })
    return balls


def to_ls_results(balls, orig_w, orig_h):
    return [
        {
            "from_name": FROM_NAME,
            "to_name":   TO_NAME,
            "type":      "rectanglelabels",
            "value": {
                "x": b["x"], "y": b["y"], "width": b["w"], "height": b["h"],
                "rotation": 0,
                "rectanglelabels": [LABEL_NAME],
            },
            "original_width":  orig_w,
            "original_height": orig_h,
        }
        for b in balls
    ]


def draw_detections(img: np.ndarray, balls: list) -> np.ndarray:
    vis = img.copy()
    H, W = vis.shape[:2]
    sizes  = [b["side_px"] for b in balls]
    circs  = [b["circ"]    for b in balls]
    for b in balls:
        x1 = int(b["x"] / 100 * W)
        y1 = int(b["y"] / 100 * H)
        x2 = x1 + int(b["w"] / 100 * W)
        y2 = y1 + int(b["h"] / 100 * H)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=DEFAULT_N,
                    help="tasks to process (default %(default)s)")
    ap.add_argument("--only-unpredicted", action="store_true",
                    help="skip tasks that already have a prediction")
    ap.add_argument("--dry-run", action="store_true",
                    help="detect but don't push to LS")
    ap.add_argument("--show", type=int, default=0, metavar="N",
                    help="display detections on the first N images (press any key to advance)")
    args = ap.parse_args()

    ls = LabelStudioClient(base_url=LS_URL, api_key=os.environ["LS_API_KEY"])

    task_iter = itertools.islice(
        ls.tasks.list(project=PROJECT_ID, only_annotated=False, page_size=200),
        args.n,
    )
    pushed = skip_predicted = skip_fetch = skip_no_balls = 0
    # buffer = list of (tid, img, balls) — kept only when --show is active
    buffer = []

    for task in task_iter:
        tid = task.id

        if args.only_unpredicted and getattr(task, "total_predictions", 0):
            skip_predicted += 1
            continue

        url = task.data["image"] if isinstance(task.data, dict) else task.data.image
        img = fetch_image(url)
        if img is None:
            skip_fetch += 1
            continue

        orig_h, orig_w = img.shape[:2]
        balls = detect_balls(img)
        print(f"  task {tid}: {len(balls)} ball(s)  {'[DRY RUN]' if args.dry_run else ''}")

        if args.show:
            buffer.append((tid, img, balls))

        if not balls:
            skip_no_balls += 1
            continue

        if args.dry_run:
            continue

        try:
            ls.predictions.create(
                task=tid,
                model_version=MODEL_VER,
                score=0.5,
                result=to_ls_results(balls, orig_w, orig_h),
            )
            pushed += 1
        except Exception as e:
            print(f"  [push] task {tid} FAILED: {e}")

    # --show: pick frames with most balls, save previews + print stats
    if args.show and buffer:
        top = sorted(buffer, key=lambda t: len(t[2]), reverse=True)[:args.show]
        for tid, img, balls in top:
            vis = draw_detections(img, balls)
            out = pathlib.Path(__file__).parent / f"_preview_{tid}.jpg"
            cv2.imwrite(str(out), vis)
            print(f"\n  [preview] task {tid}  ({len(balls)} balls) -> {out}")
            if balls:
                sizes = [b["side_px"] for b in balls]
                circs = [b["circ"]    for b in balls]
                print(f"    size : min={min(sizes)}  max={max(sizes)}  med={int(np.median(sizes))}  mean={np.mean(sizes):.1f}")
                print(f"    circ : min={min(circs):.2f}  max={max(circs):.2f}  mean={np.mean(circs):.2f}")

    print(f"\n[done] pushed={pushed}  no_balls={skip_no_balls}  "
          f"already_predicted={skip_predicted}  fetch_fail={skip_fetch}")


if __name__ == "__main__":
    main()
