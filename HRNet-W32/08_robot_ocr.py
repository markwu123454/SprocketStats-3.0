#!/usr/bin/env python3
"""
Step 8 — Identify which robot is which team by reading bumper numbers.

Given:
  - A frame image (BGR)
  - Detected robot positions from the HRNet model [x, y, cls, score]
  - The 6 known team numbers for the match (3 red, 3 blue)

Returns per-robot team assignments by:
  1. Cropping a bumper-focused region around each detected center
  2. Running EasyOCR (digit-only) on each crop
  3. Building a similarity cost matrix vs. the known candidates
  4. Solving a 1-to-1 assignment with the Hungarian algorithm

Why constrained matching works: even if OCR reads "1b78" instead of "1678",
the Levenshtein distance to the correct candidate will be much smaller than to
any other 4-digit number in the match — especially when the candidate set has
only 3 options per alliance.

Usage (standalone):
  python 08_robot_ocr.py path/to/frame.jpg \
      --red 1678 2056 971 \
      --blue 254 1 4414 \
      [--ckpt data/frc/checkpoints/best.pt]  # optional: skip if passing --no-detect

  python 08_robot_ocr.py path/to/frame.jpg \
      --red 1678 2056 971 \
      --blue 254 1 4414 \
      --no-detect   # skip HRNet; scan full image for any of the 6 numbers
"""
import argparse
import pathlib
import sys
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Levenshtein (pure Python, no extra dep) for OCR → candidate scoring
# ─────────────────────────────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _similarity(ocr_text: str, candidate: int) -> float:
    """
    Score how well an OCR string matches a team number (0..1, 1=perfect).

    Strategy (in priority order):
      1. Exact match after stripping non-digits               → 1.0
      2. Candidate string is a substring of cleaned OCR text  → 0.85
      3. Normalized Levenshtein on digit-only strings         → decreasing
    """
    cand_str = str(candidate)
    # keep only digits from ocr
    ocr_digits = "".join(ch for ch in ocr_text if ch.isdigit())
    if not ocr_digits:
        return 0.0
    if ocr_digits == cand_str:
        return 1.0
    if cand_str in ocr_digits:
        return 0.85
    dist = _levenshtein(ocr_digits, cand_str)
    max_len = max(len(ocr_digits), len(cand_str))
    return max(0.0, 1.0 - dist / max_len)


# ─────────────────────────────────────────────────────────────────────────────
# Image crop helpers
# ─────────────────────────────────────────────────────────────────────────────

def _crop(img: np.ndarray, cx: float, cy: float,
          crop_w: int = 300, crop_h: int = 220) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Return a region around (cx, cy) biased slightly downward (bumpers sit
    below the robot's geometric center). Also returns the crop's top-left
    corner so callers can map OCR bboxes back to image coordinates.
    """
    H, W = img.shape[:2]
    # shift down ~15 % of crop height to bias toward bumpers
    cy_adj = cy + crop_h * 0.15
    x0 = int(max(0, cx - crop_w / 2))
    y0 = int(max(0, cy_adj - crop_h / 2))
    x1 = min(W, x0 + crop_w)
    y1 = min(H, y0 + crop_h)
    return img[y0:y1, x0:x1].copy(), (x0, y0)


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """
    Light preprocessing to improve OCR accuracy on bumper text:
      - Upscale 2× (EasyOCR works better on larger text)
      - CLAHE to even out uneven arena lighting
    """
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Core OCR + assignment
# ─────────────────────────────────────────────────────────────────────────────

def _run_ocr_on_crop(reader, crop: np.ndarray) -> list[str]:
    """Return all digit sequences found by EasyOCR in this crop, ordered by
    detection confidence (highest first)."""
    results = reader.readtext(crop, allowlist="0123456789", detail=True)
    # results: list of (bbox, text, conf)
    results.sort(key=lambda r: -r[2])
    return [r[1] for r in results if r[1].strip()]


def _score_matrix(ocr_texts_per_robot: list[list[str]],
                  candidates: list[int]) -> np.ndarray:
    """
    Build a (n_robots × n_candidates) score matrix.
    Entry [i, j] = best similarity score between robot i's OCR texts and candidate j.
    """
    n_robots = len(ocr_texts_per_robot)
    n_cands = len(candidates)
    mat = np.zeros((n_robots, n_cands), dtype=np.float32)
    for i, texts in enumerate(ocr_texts_per_robot):
        for j, cand in enumerate(candidates):
            best = max((_similarity(t, cand) for t in texts), default=0.0)
            mat[i, j] = best
    return mat


def _hungarian_assign(score_matrix: np.ndarray) -> list[int | None]:
    """
    Optimal 1-to-1 assignment of robots to candidates (maximise total score).
    Returns a list of length n_robots: assigned candidate index, or None if
    n_candidates < n_robots (shouldn't happen in normal use).
    """
    from scipy.optimize import linear_sum_assignment
    cost = 1.0 - score_matrix                   # scipy minimises cost
    row_ind, col_ind = linear_sum_assignment(cost)
    result: list[int | None] = [None] * len(score_matrix)
    for r, c in zip(row_ind, col_ind):
        result[r] = c
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class RobotOCR:
    """
    Lazy-loads EasyOCR once; call identify() for each frame.

    Parameters
    ----------
    gpu : bool
        Pass False to force CPU (slower but no CUDA needed).
    crop_w, crop_h : int
        Pixel size of the crop around each detected robot center.
        Defaults work well for 1280×720 broadcast footage.
    min_score : float
        Minimum assignment score (0–1) to consider a match confident.
        Below this threshold the assignment is still returned but flagged
        as low-confidence in the output.
    """

    def __init__(self, gpu: bool = True, crop_w: int = 300, crop_h: int = 220,
                 min_score: float = 0.4):
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.min_score = min_score

    def identify(
        self,
        img_bgr: np.ndarray,
        detections: list,          # [[x, y, cls_str, score], ...]  from HRNet
        red_teams: list[int],      # 3 team numbers
        blue_teams: list[int],     # 3 team numbers
    ) -> list[dict]:
        """
        Match each detected robot to a team number.

        Returns a list of dicts (one per detection), each with:
          x, y          – pixel coordinates (from HRNet)
          alliance      – "red" or "blue"
          detect_score  – HRNet confidence
          team          – assigned team number (int)
          ocr_score     – similarity score for the assignment (0–1)
          confident     – True if ocr_score >= min_score
          ocr_texts     – raw OCR strings found in this robot's crop
        """
        by_alliance: dict[str, list] = {"red": [], "blue": []}
        for i, det in enumerate(detections):
            x, y, cls, score = det[0], det[1], det[2], det[3]
            by_alliance.setdefault(cls, []).append((i, x, y, cls, score))

        candidates_map = {"red": [int(t) for t in red_teams],
                          "blue": [int(t) for t in blue_teams]}

        # Preserve original order of detections for output
        result: list[dict | None] = [None] * len(detections)

        for alliance in ("red", "blue"):
            robots = by_alliance.get(alliance, [])
            candidates = candidates_map[alliance]
            if not robots:
                continue

            # Gather OCR texts for each robot in this alliance
            ocr_per_robot: list[list[str]] = []
            for _, x, y, _, _ in robots:
                crop, _ = _crop(img_bgr, x, y, self.crop_w, self.crop_h)
                crop_pp = _preprocess_crop(crop)
                texts = _run_ocr_on_crop(self.reader, crop_pp)
                ocr_per_robot.append(texts)

            scores = _score_matrix(ocr_per_robot, candidates)
            assignment = _hungarian_assign(scores)

            for robot_idx, (orig_i, x, y, cls, det_score) in enumerate(robots):
                cand_idx = assignment[robot_idx]
                if cand_idx is None:
                    team = candidates[0] if candidates else -1
                    ocr_score = 0.0
                else:
                    team = candidates[cand_idx]
                    ocr_score = float(scores[robot_idx, cand_idx])

                result[orig_i] = {
                    "x": x,
                    "y": y,
                    "alliance": alliance,
                    "detect_score": det_score,
                    "team": team,
                    "ocr_score": ocr_score,
                    "confident": ocr_score >= self.min_score,
                    "ocr_texts": ocr_per_robot[robot_idx],
                }

        return [r for r in result if r is not None]


# ─────────────────────────────────────────────────────────────────────────────
# Inset camera OCR
# FRC Championship / District broadcasts have a consistent layout:
#   main field view (full frame, top ~60%)
#   blue inset     bottom-left  (~35% width, ~35% height)
#   red inset      bottom-right (~35% width, ~35% height)
#   scoreboard     bottom-center
#
# We detect the inset borders by looking for large saturated blue/red rectangles
# in the lower half of the frame, then OCR inside them.
# ─────────────────────────────────────────────────────────────────────────────

def _find_inset_roi(img_bgr: np.ndarray, alliance: str,
                    search_frac: float = 0.35) -> np.ndarray | None:
    """
    Return the interior of the alliance-colour bordered inset camera box,
    or None if the inset is not detected.

    alliance: "blue" or "red"
    search_frac: look only in the bottom `search_frac` fraction of the image
    """
    H, W = img_bgr.shape[:2]
    search_top = int(H * (1 - search_frac))
    roi = img_bgr[search_top:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if alliance == "blue":
        # hue ~100-130, high saturation
        mask = cv2.inRange(hsv, (95, 120, 80), (135, 255, 255))
    else:
        # hue ~0-10 and ~165-180
        mask1 = cv2.inRange(hsv, (0, 140, 80), (12, 255, 255))
        mask2 = cv2.inRange(hsv, (165, 140, 80), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)

    # Find the largest contour — that's the border rectangle
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    largest = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < (W * H * search_frac) * 0.02:   # must be > 2% of search area
        return None

    x, y, w, h = cv2.boundingRect(largest)
    # inset interior: shrink border by a few px
    pad = 8
    x0 = max(0, x + pad)
    y0 = max(0, y + pad)
    x1 = min(roi.shape[1], x + w - pad)
    y1 = min(roi.shape[0], y + h - pad)
    return roi[y0:y1, x0:x1]


def ocr_insets(
    img_bgr: np.ndarray,
    reader,
    red_teams: list[int],
    blue_teams: list[int],
) -> dict[str, dict]:
    """
    OCR the alliance-specific inset cameras and return best team match per alliance.

    Returns:
      {
        "blue": {"teams_seen": [3939, 6326], "raw_texts": ["3939", ...]},
        "red":  {"teams_seen": [686],        "raw_texts": ["686", ...]},
      }
    """
    results = {}
    for alliance, candidates in (("blue", blue_teams), ("red", red_teams)):
        inset = _find_inset_roi(img_bgr, alliance)
        if inset is None or inset.size == 0:
            results[alliance] = {"teams_seen": [], "raw_texts": [], "inset_found": False}
            continue

        pp = _preprocess_crop(inset)
        raw = reader.readtext(pp, allowlist="0123456789", detail=True)
        raw_texts = [r[1] for r in raw if r[1].strip()]

        teams_seen = []
        for cand in candidates:
            best = max((_similarity(t, cand) for t in raw_texts), default=0.0)
            if best >= 0.7:
                teams_seen.append(cand)

        results[alliance] = {
            "teams_seen": teams_seen,
            "raw_texts": raw_texts,
            "inset_found": True,
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: scan whole image when HRNet detections aren't available
# ─────────────────────────────────────────────────────────────────────────────

def identify_without_detections(
    img_bgr: np.ndarray,
    reader,
    red_teams: list[int],
    blue_teams: list[int],
) -> dict[int, list[dict]]:
    """
    Run OCR across the whole image (no detection model needed) and return
    all text detections that match any of the 6 team numbers.

    Returns {team_number: [{"x", "y", "w", "h", "text", "score"}, ...]}
    """
    all_candidates = {t: "red" for t in red_teams}
    all_candidates.update({t: "blue" for t in blue_teams})

    raw = reader.readtext(img_bgr, allowlist="0123456789", detail=True)
    findings: dict[int, list[dict]] = {t: [] for t in all_candidates}

    for bbox, text, conf in raw:
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            continue
        for team, alliance in all_candidates.items():
            score = _similarity(digits, team)
            if score >= 0.7:
                pts = np.array(bbox, dtype=np.int32)
                x, y = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                w = int(pts[:, 0].max() - pts[:, 0].min())
                h = int(pts[:, 1].max() - pts[:, 1].min())
                findings[team].append({
                    "x": x, "y": y, "w": w, "h": h,
                    "text": text, "score": score, "alliance": alliance,
                })

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = {
    "blue": (220, 100, 0),    # BGR orange-tinted blue
    "red":  (0, 60, 220),     # BGR red
}


def draw_assignments(img_bgr: np.ndarray, assignments: list[dict]) -> np.ndarray:
    out = img_bgr.copy()
    for a in assignments:
        x, y = int(a["x"]), int(a["y"])
        color = _COLORS.get(a["alliance"], (0, 255, 0))
        label = f"#{a['team']}"
        if not a["confident"]:
            label += "?"
        cv2.circle(out, (x, y), 8, color, -1)
        cv2.putText(out, label, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        score_txt = f"{a['ocr_score']:.2f}"
        cv2.putText(out, score_txt, (x + 10, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Identify robot team numbers in an FRC frame.")
    ap.add_argument("image", help="Path to the frame (jpg/png)")
    ap.add_argument("--red", nargs=3, type=int, required=True, metavar="TEAM",
                    help="Three red alliance team numbers")
    ap.add_argument("--blue", nargs=3, type=int, required=True, metavar="TEAM",
                    help="Three blue alliance team numbers")
    ap.add_argument("--ckpt", default=None,
                    help="HRNet checkpoint (best.pt). Omit to use --no-detect mode.")
    ap.add_argument("--no-detect", action="store_true",
                    help="Skip HRNet; run OCR on the full image instead.")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="HRNet peak threshold (default 0.3)")
    ap.add_argument("--gpu", action="store_true", default=False,
                    help="Use GPU for EasyOCR (default: CPU)")
    ap.add_argument("--no-gpu", dest="gpu", action="store_false")
    ap.add_argument("--out", default=None,
                    help="Save annotated image here (default: show in window)")
    ap.add_argument("--crop-w", type=int, default=300)
    ap.add_argument("--crop-h", type=int, default=220)
    ap.add_argument("--min-score", type=float, default=0.4,
                    help="Min OCR similarity to flag as confident (default 0.4)")
    args = ap.parse_args()

    img_path = pathlib.Path(args.image)
    if not img_path.exists():
        sys.exit(f"[error] image not found: {img_path}")
    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit(f"[error] could not decode image: {img_path}")

    import easyocr
    print("[ocr] loading EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=args.gpu, verbose=False)

    if args.no_detect or args.ckpt is None:
        print("[ocr] scanning full image (no-detect mode)...")
        findings = identify_without_detections(img, reader, args.red, args.blue)
        for team, hits in findings.items():
            if hits:
                best = max(hits, key=lambda h: h["score"])
                print(f"  team {team:>4} ({best['alliance']:4}): "
                      f"found '{best['text']}' at ({best['x']}, {best['y']}) "
                      f"score={best['score']:.2f}")
            else:
                print(f"  team {team:>4}: not found")
        annotated = img.copy()
        for team, hits in findings.items():
            alliance = args.red and "red" if team in args.red else "blue"
            color = _COLORS.get(alliance, (0, 255, 0))
            for h in hits:
                cv2.circle(annotated, (h["x"], h["y"]), 6, color, -1)
                cv2.putText(annotated, f"#{team}", (h["x"] + 8, h["y"] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    else:
        # Load HRNet and run detection
        import torch, yaml
        ROOT = pathlib.Path(__file__).parent
        sys.path.insert(0, str(ROOT))
        from model import HeatmapNet, decode_peaks, CLASSES
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[detect] loading checkpoint: {args.ckpt}")
        ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
        cfg = ck.get("cfg") or yaml.safe_load(open(ROOT / "config.yaml"))
        hm = cfg["heatmap"]
        # Skip backbone pretrain download — checkpoint already has the weights.
        import timm as _timm
        _orig_create = _timm.create_model
        _timm.create_model = lambda *a, **kw: _orig_create(*a, **{**kw, "pretrained": False})
        try:
            model = HeatmapNet(cfg["train"]["backbone"], len(CLASSES),
                               hm["output_stride"]).to(DEVICE)
        finally:
            _timm.create_model = _orig_create
        model.load_state_dict(ck["model"])
        model.eval()

        in_h, in_w = hm["input_size"]
        H0, W0 = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_r = cv2.resize(img_rgb, (in_w, in_h))
        _MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)
        _STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)
        t = (torch.from_numpy(img_r).permute(2, 0, 1).float().to(DEVICE) / 255.0 - _MEAN) / _STD
        with torch.no_grad():
            logits = model(t.unsqueeze(0))
        dets_hm = decode_peaks(logits, args.threshold, hm["nms_kernel"], hm["max_instances"])[0]

        # Map heatmap coords → original image pixels
        CH_TO_CLS = {i: c for i, c in enumerate(CLASSES)}
        detections = []
        for x_hm, y_hm, c, score in dets_hm:
            x_in = x_hm * hm["output_stride"]
            y_in = y_hm * hm["output_stride"]
            detections.append([x_in * (W0 / in_w), y_in * (H0 / in_h),
                               CH_TO_CLS[c], score])

        print(f"[detect] {len(detections)} robots detected "
              f"({sum(1 for d in detections if d[2]=='blue')} blue, "
              f"{sum(1 for d in detections if d[2]=='red')} red)")

        ocr_sys = RobotOCR(gpu=args.gpu, crop_w=args.crop_w,
                           crop_h=args.crop_h, min_score=args.min_score)

        # --- Inset camera OCR (always run, independent of field crops) ---
        print("[ocr] scanning alliance inset cameras...")
        inset_results = ocr_insets(img, ocr_sys.reader, args.red, args.blue)
        for alliance, info in inset_results.items():
            found = info["inset_found"]
            teams = info["teams_seen"]
            texts = info["raw_texts"]
            print(f"  {alliance}: inset={'found' if found else 'NOT found'}  "
                  f"teams_seen={teams}  raw={texts}")

        print("[ocr] running OCR on field robot crops...")
        assignments = ocr_sys.identify(img, detections, args.red, args.blue)

        print("\n[field results]")
        for a in assignments:
            conf_tag = "Y" if a["confident"] else "?"
            print(f"  {conf_tag} team {a['team']:>4} ({a['alliance']:4}) "
                  f"at ({a['x']:.0f}, {a['y']:.0f})  "
                  f"ocr_score={a['ocr_score']:.2f}  "
                  f"texts={a['ocr_texts']}")

        annotated = draw_assignments(img, assignments)

    if args.out:
        cv2.imwrite(args.out, annotated)
        print(f"[saved] {args.out}")
    else:
        cv2.imshow("Robot OCR", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
