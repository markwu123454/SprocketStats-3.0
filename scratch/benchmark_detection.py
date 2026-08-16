#!/usr/bin/env python3
"""
Benchmark new AT3 image-preprocessing variants against the current 4-combo
baseline across multiple match videos.

Baseline (current ensemble):
  sharp+native, sharp+upscale2x, sharp+clahe, soft+native

Candidates:
  sharp+unsharp   -- unsharp mask pre-filter (sharpens before the quad detector,
                     unlike AT3's decode_sharpening which acts after the quad is found)
  sharp+upscale3x -- 3x bicubic upscale (for very small/distant tags)
  sharp+denoise   -- bilateral filter to suppress H.264 block-boundary gradients
  sharp+gamma     -- gamma=0.5 brighten (for underexposed broadcast feeds)

Metric: for each candidate, count (view, frame, tag_id) triples where it found
a tag the baseline union completely missed. This directly answers "would this
technique find tags we'd otherwise lose?"

Usage
-----
  python scratch/benchmark_detection.py match.mp4 match2.mp4 match3.mp4 match4.mp4
  python scratch/benchmark_detection.py match2.mp4 --n-frames 20
"""

import argparse, json, pathlib, sys, time
import numpy as np
import cv2
from pupil_apriltags import Detector as AT3Detector

PROF_DIR = pathlib.Path(__file__).parent.parent / "homography" / "data" / "profiles"

SAMPLE_START = 0.05
SAMPLE_END   = 0.95

BASELINE_COMBOS  = ["sharp+native", "sharp+upscale2x", "sharp+clahe", "soft+native"]
CANDIDATE_COMBOS = ["sharp+unsharp", "sharp+upscale3x", "sharp+denoise", "sharp+gamma"]

_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def _make_detector(quad_sigma: float, decode_sharpening: float) -> AT3Detector:
    return AT3Detector(
        families="tag36h11", nthreads=4,
        quad_decimate=1.0, quad_sigma=quad_sigma,
        refine_edges=1, decode_sharpening=decode_sharpening,
    )


# ---------------------------------------------------------------------------
# Image variants
# ---------------------------------------------------------------------------

def _unsharp(gray: np.ndarray, strength: float = 1.5, sigma: float = 2.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return np.clip(cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0),
                   0, 255).astype(np.uint8)


def _upscale(gray: np.ndarray, factor: float) -> np.ndarray:
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def _denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)


_GAMMA_TABLE = np.array(
    [(i / 255.0) ** 0.5 * 255 for i in range(256)], dtype=np.uint8
)

def _gamma(gray: np.ndarray) -> np.ndarray:
    return cv2.LUT(gray, _GAMMA_TABLE)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _detect(img: np.ndarray, det: AT3Detector, scale: float = 1.0) -> dict[int, float]:
    """Run detector, return {tag_id: decision_margin} (best margin per tag)."""
    found: dict[int, float] = {}
    for d in det.detect(img):
        m = float(d.decision_margin)
        if d.tag_id not in found or m > found[d.tag_id]:
            found[d.tag_id] = m
    return found


def run_all_combos(gray: np.ndarray,
                   dets: dict[str, AT3Detector]) -> dict[str, dict[int, float]]:
    """Run every combo on `gray`. Returns {combo_name: {tag_id: margin}}."""
    u2 = _upscale(gray, 2.0)
    u3 = _upscale(gray, 3.0)
    return {
        # baseline
        "sharp+native":   _detect(gray,           dets["sharp"]),
        "sharp+upscale2x":_detect(u2,             dets["sharp"], 2.0),
        "sharp+clahe":    _detect(_CLAHE.apply(gray), dets["sharp"]),
        "soft+native":    _detect(gray,           dets["soft"]),
        # candidates
        "sharp+unsharp":  _detect(_unsharp(gray), dets["sharp"]),
        "sharp+upscale3x":_detect(u3,             dets["sharp"], 3.0),
        "sharp+denoise":  _detect(_denoise(gray), dets["sharp"]),
        "sharp+gamma":    _detect(_gamma(gray),   dets["sharp"]),
    }


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def sample_frames(video_path: str, n: int,
                  start_frac: float, end_frac: float) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[error] cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo    = max(0, int(total * start_frac))
    hi    = min(total - 1, int(total * end_frac))
    step  = max(1, (hi - lo) // max(1, n - 1))
    idxs  = list(range(lo, hi + 1, step))[:n]
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            frames.append((idx, f))
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--n-frames", type=int, default=40,
                    help="frames to sample per match (default: 40)")
    args = ap.parse_args()

    dets = {
        "sharp": _make_detector(0.0, 1.25),
        "soft":  _make_detector(0.6, 0.5),
    }

    # Global accumulators
    # new_finds[combo]  = list of (stem, view, tag_id) found exclusively by this candidate
    # new_frames[combo] = number of (frame, view, tag) triples baseline missed
    new_finds:  dict[str, list] = {c: [] for c in CANDIDATE_COMBOS}
    new_frames: dict[str, int]  = {c: 0  for c in CANDIDATE_COMBOS}
    baseline_total = 0   # total (view × frame × tag_id) detections by baseline

    for video_str in args.videos:
        video_path = pathlib.Path(video_str)
        stem = video_path.stem
        print(f"\n{'='*60}")
        print(f"  {stem}  ({video_path.stat().st_size // 1_000_000} MB)")
        print(f"{'='*60}")

        layout_path = PROF_DIR / f"{stem}_layout.json"
        if not layout_path.exists():
            print(f"  [skip] no layout profile -- run 00_split_views.py --video {video_str} first")
            continue
        layout = json.loads(layout_path.read_text())

        # Deduplicate views (layout may have a "main" alias sharing a box)
        seen_boxes, unique_views = set(), []
        for v in layout.get("views", []):
            key = tuple(v["box"])
            if key not in seen_boxes:
                seen_boxes.add(key)
                unique_views.append(v)

        t0 = time.time()
        frames = sample_frames(str(video_path), args.n_frames, SAMPLE_START, SAMPLE_END)
        print(f"  sampled {len(frames)} frames  "
              f"[{frames[0][0]}..{frames[-1][0]}]  ({time.time()-t0:.1f}s load)")

        for view in unique_views:
            vname        = view["name"]
            x0,y0,x1,y1 = view["box"]

            # Per-(tag_id): frames found by baseline vs each candidate
            baseline_tag_frames:  dict[int, int] = {}   # tag_id -> n frames baseline found it
            cand_exclusive_frames: dict[str, dict[int, int]] = {
                c: {} for c in CANDIDATE_COMBOS
            }

            t1 = time.time()
            for _fi, (_idx, frame_bgr) in enumerate(frames):
                crop = frame_bgr[y0:y1, x0:x1]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                results = run_all_combos(gray, dets)

                # Baseline union
                baseline_found: set[int] = set()
                for c in BASELINE_COMBOS:
                    baseline_found |= results[c].keys()

                baseline_total += len(baseline_found)
                for tid in baseline_found:
                    baseline_tag_frames[tid] = baseline_tag_frames.get(tid, 0) + 1

                # Candidate exclusives
                for c in CANDIDATE_COMBOS:
                    cand_found  = set(results[c].keys())
                    exclusive   = cand_found - baseline_found
                    for tid in exclusive:
                        cand_exclusive_frames[c][tid] = \
                            cand_exclusive_frames[c].get(tid, 0) + 1
                        new_frames[c] += 1

            elapsed = time.time() - t1
            print(f"\n  [{vname}]  {elapsed:.1f}s")
            print(f"    baseline: {len(baseline_tag_frames)} unique tag(s) "
                  f"found across {len(frames)} frames  "
                  f"ids={sorted(baseline_tag_frames.keys())}")

            for c in CANDIDATE_COMBOS:
                excl = cand_exclusive_frames[c]
                if excl:
                    # Track globally: (stem, vname, tag_id) tuples
                    for tid in excl:
                        new_finds[c].append((stem, vname, tid))
                    tag_summary = {tid: f"{n}fr" for tid, n in sorted(excl.items())}
                    print(f"    {c}: +{sum(excl.values())} detections  "
                          f"exclusive tag(s): {tag_summary}")
                else:
                    print(f"    {c}: (no new tags)")

    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  SUMMARY  (baseline total detections: {baseline_total})")
    print(f"{'='*60}")
    for c in CANDIDATE_COMBOS:
        n     = new_frames[c]
        finds = new_finds[c]
        pct   = 100 * n / max(1, baseline_total)
        unique_combos = len(set(finds))
        print(f"  {c:<22}  +{n:4d} frame-detections  ({pct:5.2f}%)  "
              f"{unique_combos} unique (video,view,tag) combos")
        for stem, vname, tid in sorted(set(finds)):
            count = sum(1 for x in finds if x == (stem, vname, tid))
            print(f"              -> {stem}/{vname}/tag{tid}  ({count} frame(s))")

    print()


if __name__ == "__main__":
    main()
