#!/usr/bin/env python3
"""
Step 1 -- Detect AprilTags per camera view using AprilTag 3 (pupil-apriltags).

Replaces the previous ArUco-based approach. ArUco uses adaptive thresholding
to find quads, which produces ~800 false-positive rejected candidates per frame
on broadcast footage and misses many real tags. AprilTag 3 uses gradient-based
quad detection (clustering by gradient direction, union-find region fitting)
which is fundamentally more robust to JPEG compression artifacts, motion blur,
and distant/small tags. On the same broadcast frames, AT3 decodes 5-7 tags per
frame where ArUco decoded 0-4, and never does worse.

The old rejected-candidates pipeline (persistence clustering + parallelogram
scoring) is dropped because AT3 does not expose intermediate quad candidates
through its Python API -- only successfully decoded detections are returned.

Optimal single-shot parameters discovered empirically on broadcast FRC footage:
  quad_sigma=0.0       -- no pre-blur; AT3's gradient approach doesn't need it
                          and blur smears the bit cells before decode
  decode_sharpening=1.25 -- aggressive sharpening of the tag interior before
                            bit decoding recovers compressed tags
  quad_decimate=1.0    -- no subsampling; most thorough, needed for small tags
  refine_edges=1       -- subpixel corner refinement

Ensemble mode (default, --no-ensemble to disable)
--------------------------------------------------
A single detector config is a compromise: aggressive sharpening recovers
compressed tags but a tag with real motion blur wants mild blur instead, and
a tag that's simply too small for the bit decoder wants more pixels, not
different preprocessing. So each frame is run through multiple independent
(image variant x detector config) combos and results are merged per tag_id,
keeping the highest decision_margin on overlap. This only adds *unique*
detections; it never removes one the single-shot pass would have found.

  variants:
    native      -- unmodified crop
    upscale2x   -- 2x bicubic upscale; gives the bit-cell decoder more
                   samples per cell for small/distant tags (does nothing for
                   tags that were already large enough, costs nothing there)
    clahe       -- adaptive histogram equalization; recovers tags in glare
                   or shadow (stadium lighting is uneven)

  detector configs:
    sharp -- the tuned defaults above; best for compression artifacts
    soft  -- quad_sigma=0.6, decode_sharpening=0.5; best for motion blur
             (sharpening a blurred tag amplifies the blur, not the signal)

  combos run: sharp x native, sharp x upscale2x, sharp x upscale3x,
              sharp x upscale4x, sharp x clahe, sharp x unsharp,
              sharp x gamma, sharp x bilateral,
              soft x native, soft x upscale2x
  (soft is skipped on clahe/unsharp/gamma/bilateral -- motion blur isn't fixed
  by contrast/sharpening preprocessing, so those combos rarely earn the cost)

  upscale3x: benchmarked across 4 matches (+6.5% frame-detections vs baseline).
  unsharp: largest contributor (+14.9%) -- sharpening before the quad detector
  helps find tag borders the baseline misses entirely (tag 6 in match3/main
  was found in 23/40 sampled frames exclusively by this combo).
  upscale4x: extends upscale3x for tags at the detection limit (~14px, e.g.,
  wall tags 13/14); at 4x they are ~56px giving the bit-cell decoder margin.
  gamma: global gamma=0.5 brightens shadowed tags; complementary to CLAHE
  which only does local contrast -- a tag in deep shadow benefits from both.
  bilateral: bilateral filter (d=9, sigmaColor=75) removes H.264 DCT block
  artifacts from flat regions before the gradient quad detector runs; those
  blocks create spurious high-frequency gradients that can mask tag borders.
  soft+upscale2x: motion blur + small tag is a real failure combo; the soft
  config avoids amplifying the blur while upscale gives the decoder more pixels.

Outputs
-------
  data/detections/<stem>_tags.json
    { "video": ..., "n_frames_sampled": ..., "frame_indices": [...],
      "params": {...},
      "views": {
        "main": {
          "box": [...],
          "n_frames_sampled": N,
          "decoded_tags": {
            "5": {
              "n_frames_detected": N,
              "mean_size_px": ...,
              "mean_center_px": [...],
              "mean_corners": [[x,y], ...],
              "mean_decision_margin": ...,
              "observations": [
                {"frame_idx": N, "corners": [...], "size_px": ...,
                 "decision_margin": ..., "source": "sharp+upscale2x"},
                ...
              ]
            }, ...
          }
        }, ...
      }
    }

Usage
-----
  python pipeline/01_detect_tags.py --video match.mp4
  python pipeline/01_detect_tags.py --video match.mp4 --view main --n-frames 200
  python pipeline/01_detect_tags.py --video match.mp4 --no-ensemble   # single-pass, faster

Install: pip install opencv-python numpy pupil-apriltags
"""

import argparse, json, pathlib, sys
import numpy as np
import cv2
from pupil_apriltags import Detector as AT3Detector

DATA_DIR       = pathlib.Path(__file__).parent.parent / "data"
PROF_DIR       = DATA_DIR / "profiles"
DETECTIONS_DIR = DATA_DIR / "detections"

SAMPLE_START = 0.05
SAMPLE_END   = 0.95

DEFAULT_N_FRAMES         = 150
DEFAULT_QUAD_DECIMATE    = 1.0
DEFAULT_QUAD_SIGMA       = 0.0
DEFAULT_DECODE_SHARPENING = 1.25

# Second detector config for ensemble mode -- tuned for motion blur, the
# opposite failure mode from the compression artifacts DEFAULT_* targets.
SOFT_QUAD_SIGMA        = 0.6
SOFT_DECODE_SHARPENING = 0.5

UPSCALE_FACTOR   = 2.0
UPSCALE_FACTOR_3 = 3.0
UPSCALE_FACTOR_4 = 4.0

GAMMA_VALUE = 0.5  # < 1 boosts shadows; complements CLAHE's local contrast


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def _make_detector(quad_decimate: float, quad_sigma: float,
                   decode_sharpening: float) -> AT3Detector:
    return AT3Detector(
        families          = "tag36h11",
        nthreads          = 4,
        quad_decimate     = quad_decimate,
        quad_sigma        = quad_sigma,
        refine_edges      = 1,
        decode_sharpening = decode_sharpening,
    )


def _make_ensemble_detectors(quad_decimate: float,
                             sharp_quad_sigma: float = DEFAULT_QUAD_SIGMA,
                             sharp_decode_sharpening: float = DEFAULT_DECODE_SHARPENING
                             ) -> dict[str, AT3Detector]:
    """sharp = tuned defaults (compression artifacts); soft = motion blur."""
    return {
        "sharp": _make_detector(quad_decimate, sharp_quad_sigma,
                                sharp_decode_sharpening),
        "soft":  _make_detector(quad_decimate, SOFT_QUAD_SIGMA,
                                SOFT_DECODE_SHARPENING),
    }


def _quad_size(corners: np.ndarray) -> float:
    edges = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i]))
             for i in range(4)]
    return float(np.mean(edges))


# ---------------------------------------------------------------------------
# Ensemble image variants
# ---------------------------------------------------------------------------

_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def _unsharp(gray: np.ndarray, strength: float = 1.5, sigma: float = 2.0) -> np.ndarray:
    """Unsharp mask: amplify high-frequency edges before the quad detector runs.

    AT3's decode_sharpening only acts on the tag interior after the quad is
    already found -- it can't help the quad detector find the border in the
    first place. Pre-sharpening the whole crop does. Benchmarked: +14.9%
    frame-detections vs the 4-combo baseline across 4 matches.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return np.clip(cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0),
                   0, 255).astype(np.uint8)


_GAMMA_LUT = np.array(
    [int((i / 255.0) ** GAMMA_VALUE * 255 + 0.5) for i in range(256)],
    dtype=np.uint8,
)


def _gamma_boost(gray: np.ndarray) -> np.ndarray:
    """Global power-law gamma < 1: lifts shadows without touching local contrast.

    CLAHE only adjusts local contrast tiles; a tag in a uniformly dark region
    (deep shadow, underexposed end wall) benefits from a global brightness lift
    first. gamma=0.5 doubles perceived brightness in mid-tones.
    """
    return cv2.LUT(gray, _GAMMA_LUT)


def _bilateral(gray: np.ndarray) -> np.ndarray:
    """Edge-preserving bilateral filter to remove H.264 DCT block noise.

    H.264/H.265 compression of broadcast footage creates 8x8 or 16x16 block
    artifacts in flat regions (walls, arena floor). Those block boundaries are
    spurious high-frequency gradients that can mask or split tag borders when
    AT3's gradient-based quad detector runs. Bilateral filter suppresses them
    while leaving the real sharp tag edges intact (large sigmaColor step in
    intensity is treated as an edge and not smoothed across).
    """
    return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)


def _ensemble_combos(gray: np.ndarray,
                     detectors: dict[str, AT3Detector]) -> list[tuple[str, AT3Detector, np.ndarray, float]]:
    """
    Return [(source_label, detector, image, scale), ...] to run for one frame.

    `scale` is the factor the image was upscaled by, so detected corners can
    be divided back down to native crop coordinates.
    """
    upscaled2  = cv2.resize(gray, None, fx=UPSCALE_FACTOR,   fy=UPSCALE_FACTOR,
                            interpolation=cv2.INTER_CUBIC)
    upscaled3  = cv2.resize(gray, None, fx=UPSCALE_FACTOR_3, fy=UPSCALE_FACTOR_3,
                            interpolation=cv2.INTER_CUBIC)
    upscaled4  = cv2.resize(gray, None, fx=UPSCALE_FACTOR_4, fy=UPSCALE_FACTOR_4,
                            interpolation=cv2.INTER_CUBIC)
    enhanced   = _CLAHE.apply(gray)
    sharpened  = _unsharp(gray)
    gamma_img  = _gamma_boost(gray)
    bilateral_ = _bilateral(gray)
    return [
        ("sharp+native",    detectors["sharp"], gray,       1.0),
        ("sharp+upscale2x", detectors["sharp"], upscaled2,  UPSCALE_FACTOR),
        ("sharp+upscale3x", detectors["sharp"], upscaled3,  UPSCALE_FACTOR_3),
        ("sharp+upscale4x", detectors["sharp"], upscaled4,  UPSCALE_FACTOR_4),
        ("sharp+clahe",     detectors["sharp"], enhanced,   1.0),
        ("sharp+unsharp",   detectors["sharp"], sharpened,  1.0),
        ("sharp+gamma",     detectors["sharp"], gamma_img,  1.0),
        ("sharp+bilateral", detectors["sharp"], bilateral_, 1.0),
        ("soft+native",     detectors["soft"],  gray,       1.0),
        ("soft+upscale2x",  detectors["soft"],  upscaled2,  UPSCALE_FACTOR),
    ]


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def sample_frames(video_path: str, n: int, start_frac: float,
                  end_frac: float) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[error] cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo    = max(0, int(total * start_frac))
    hi    = min(total - 1, int(total * end_frac))
    step  = max(1, (hi - lo) // max(1, n - 1))
    indices = list(range(lo, hi + 1, step))[:n]
    print(f"[video] {total} total frames; sampling {len(indices)} "
          f"[{indices[0]}..{indices[-1]}]", file=sys.stderr)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx, frame))
        else:
            print(f"[warn] could not read frame {idx}", file=sys.stderr)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Per-view detection
# ---------------------------------------------------------------------------

def process_view(view: dict, frames: list[tuple[int, np.ndarray]],
                 detectors: dict[str, AT3Detector], ensemble: bool = True) -> dict:
    x0, y0, x1, y1 = view["box"]
    decoded_by_id: dict[int, list[dict]] = {}
    source_counts: dict[str, int] = {}

    for fi, (idx, frame_bgr) in enumerate(frames):
        print(f"\r[{view['name']}] frame {fi + 1}/{len(frames)} (idx={idx})",
              end="", file=sys.stderr, flush=True)
        crop = frame_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        combos = (_ensemble_combos(gray, detectors) if ensemble
                 else [("sharp+native", detectors["sharp"], gray, 1.0)])

        # Multiple combos can decode the same tag in the same frame; keep
        # only the highest-decision_margin hit per tag_id so a frame never
        # contributes more than one observation per tag.
        frame_best: dict[int, dict] = {}
        for source, det, img, scale in combos:
            for d in det.detect(img):
                corners = d.corners.astype(np.float64) / scale  # back to crop coords
                candidate = {
                    "frame_idx":       idx,
                    "corners":         corners,
                    "size_px":         _quad_size(corners),
                    "center_px":       corners.mean(axis=0),
                    "decision_margin": float(d.decision_margin),
                    "source":          source,
                }
                best = frame_best.get(d.tag_id)
                if best is None or candidate["decision_margin"] > best["decision_margin"]:
                    frame_best[d.tag_id] = candidate

        for tid, cand in frame_best.items():
            decoded_by_id.setdefault(tid, []).append(cand)
            source_counts[cand["source"]] = source_counts.get(cand["source"], 0) + 1

    print(f"\r[{view['name']}] done -- "
          f"{len(decoded_by_id)} unique tag id(s) across {len(frames)} frames",
          file=sys.stderr)
    if ensemble:
        breakdown = ", ".join(f"{k}={v}" for k, v in
                              sorted(source_counts.items(), key=lambda kv: -kv[1]))
        print(f"[{view['name']}] winning source breakdown: {breakdown}",
              file=sys.stderr)

    decoded_out = {}
    for tid, obs in sorted(decoded_by_id.items()):
        decoded_out[str(tid)] = {
            "n_frames_detected":    len(obs),
            "mean_size_px":         round(float(np.mean([o["size_px"] for o in obs])), 1),
            "mean_center_px":       [round(v, 1) for v in
                                     np.mean([o["center_px"] for o in obs], axis=0)],
            "mean_corners":         np.mean([o["corners"] for o in obs],
                                            axis=0).round(1).tolist(),
            "mean_decision_margin": round(float(np.mean(
                                        [o["decision_margin"] for o in obs])), 1),
            "observations": [
                {"frame_idx":       o["frame_idx"],
                 "corners":         o["corners"].round(2).tolist(),
                 "size_px":         round(o["size_px"], 1),
                 "decision_margin": round(o["decision_margin"], 1),
                 "source":          o["source"]}
                for o in obs
            ],
        }

    return {
        "box":              view["box"],
        "n_frames_sampled": len(frames),
        "decoded_tags":     decoded_out,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, metavar="PATH")
    ap.add_argument("--view", metavar="NAME",
                    help="process only this camera view (default: all views)")
    ap.add_argument("--n-frames", type=int, default=DEFAULT_N_FRAMES,
                    help="frames to sample (default: %(default)s)")
    ap.add_argument("--sample-start", type=float, default=SAMPLE_START)
    ap.add_argument("--sample-end",   type=float, default=SAMPLE_END)
    ap.add_argument("--quad-decimate", type=float, default=DEFAULT_QUAD_DECIMATE,
                    help="AT3 image subsampling before quad detection; "
                         "1.0 = none / most thorough (default: %(default)s)")
    ap.add_argument("--quad-sigma", type=float, default=DEFAULT_QUAD_SIGMA,
                    help="AT3 gaussian blur before quad detection; "
                         "0.0 = off -- best for small compressed broadcast tags "
                         "(default: %(default)s)")
    ap.add_argument("--decode-sharpening", type=float, default=DEFAULT_DECODE_SHARPENING,
                    help="AT3 sharpening of tag interior before bit decode; "
                         "higher recovers more compressed/blurry tags "
                         "(default: %(default)s)")
    ap.add_argument("--out", metavar="PATH",
                    help="output path (default: data/detections/<stem>_tags.json)")
    ap.add_argument("--no-ensemble", action="store_true",
                    help="disable the multi-variant/multi-config ensemble and "
                         "run a single detector pass (original behavior, ~4x faster)")
    args = ap.parse_args()
    ensemble = not args.no_ensemble

    stem = pathlib.Path(args.video).stem
    layout_path = PROF_DIR / f"{stem}_layout.json"
    if not layout_path.exists():
        sys.exit(f"[error] view profile not found: {layout_path}\n"
                 f"        run pipeline/00_split_views.py --video first")
    layout = json.loads(layout_path.read_text())

    views = layout.get("views", [])
    if args.view:
        views = [v for v in views if v["name"] == args.view]
        if not views:
            sys.exit(f"[error] view {args.view!r} not in profile")

    frames = sample_frames(args.video, args.n_frames,
                           args.sample_start, args.sample_end)
    if not frames:
        sys.exit("[error] no frames could be read from video")

    detectors = _make_ensemble_detectors(args.quad_decimate, args.quad_sigma,
                                         args.decode_sharpening)

    views_out = {}
    for view in views:
        views_out[view["name"]] = process_view(view, frames, detectors, ensemble)

    total = sum(len(v["decoded_tags"]) for v in views_out.values())
    print(f"[summary] {total} unique tag id(s) across {len(views_out)} view(s)",
          file=sys.stderr)

    out = {
        "video":            args.video,
        "n_frames_sampled": len(frames),
        "frame_indices":    [idx for idx, _ in frames],
        "params": {
            "detector":                "apriltag3",
            "ensemble":                ensemble,
            "quad_decimate":           args.quad_decimate,
            "sharp_quad_sigma":        args.quad_sigma,
            "sharp_decode_sharpening": args.decode_sharpening,
            "soft_quad_sigma":         SOFT_QUAD_SIGMA if ensemble else None,
            "soft_decode_sharpening":  SOFT_DECODE_SHARPENING if ensemble else None,
            "upscale_factor_2x":       UPSCALE_FACTOR   if ensemble else None,
            "upscale_factor_3x":       UPSCALE_FACTOR_3 if ensemble else None,
            "upscale_factor_4x":       UPSCALE_FACTOR_4 if ensemble else None,
            "unsharp_strength":        1.5 if ensemble else None,
            "gamma_value":             GAMMA_VALUE if ensemble else None,
            "bilateral_d":             9 if ensemble else None,
            "bilateral_sigma_color":   75 if ensemble else None,
            "refine_edges":            1,
        },
        "views": views_out,
    }

    out_path = (pathlib.Path(args.out) if args.out
                else DETECTIONS_DIR / f"{stem}_tags.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[out] {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
