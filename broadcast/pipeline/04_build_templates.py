#!/usr/bin/env python3
"""
Step 3 -- Calibration, part 2: fuse the raw digit-glyph crops harvested by
03_calibrate.py into one higher-quality template per (kind, digit). Pure
image processing over already-saved crops, no video/OCR -- deliberately
separate from collection, which is video/OCR-bound. See 03_calibrate.py's
module docstring for the full cadence-split rationale and why "every
instance" (not just one clean read per digit) was worth collecting in the
first place: a non-monospaced font re-flows the whole field on ANY digit
change, so different instances of the same glyph land at different
sub-pixel phases -- exactly the diversity multi-frame super-resolution
needs.

Algorithm
---------
For each (kind, digit) bucket under data/digit_instances/<kind>/<digit>/:

1. Normalize every crop to grayscale, and for `kind == "clock"` INVERT it
   (clock digits are dark-on-white -- see 03_calibrate.py's polarity
   discussion) so every template ends up in the same convention regardless
   of source polarity: bright glyph on a dark field. This is purely a
   presentation/matching-convenience choice, not a claim that clock and
   score share a font -- badge/score/clock stay in separate buckets
   throughout (never averaged into each other), only the OUTPUT convention
   is unified. Legacy path only: post-refactor, `kind` is a single pooled
   "all" bucket mixing both polarities (see build_trackers in
   03_calibrate.py), so a directory-name check can no longer tell them
   apart -- 03 now normalizes polarity itself at harvest time
   (normalize_polarity), before a crop is ever written to disk, so "all"
   instances arrive here already bright-on-dark and this check correctly
   leaves them alone.
2. Resize every crop in the bucket to the bucket's own MEDIAN (w, h) via
   cubic interpolation. Crops already come out of connected-component
   bounding boxes tightly wrapping the glyph, so instance-to-instance size
   variation is small (1-3px, from antialiasing threshold noise, not a
   real size difference) -- normalizing to the median is a mild resample,
   not a distortion.
3. Upsample every resized crop by UPSCALE_FACTOR (cubic). This is what
   actually gives sub-pixel alignment somewhere to land -- there is no
   sub-pixel information to recover at native resolution.
4. Pick ONE fixed reference (the first instance in the bucket, upsampled)
   and phase-correlate (cv2.phaseCorrelate, Hanning-windowed) every other
   upsampled instance against it, then warp each by its own (dx, dy) to
   align. A FIXED reference (not an incrementally-updated running mean) is
   deliberate -- simpler to reason about and avoids the reference itself
   drifting as more instances are folded in, at the cost of being sensitive
   to the first instance being atypical; not swept against the alternative.
5. Take the per-pixel MEDIAN (not mean) across all aligned instances --
   robust to the occasional bad alignment or mis-set OCR/segmentation pair
   that survived 03's count-agreement check by coincidence. This is the
   final template.

Output
------
data/digit_templates/<kind>/<digit>.png -- the merged template.
data/digit_templates/<kind>/<digit>_compare.png -- SINGLE raw instance
(plain cubic upsample, no alignment/averaging) vs MERGED template,
side by side, at the same scale -- so the improvement (or lack of one) is
something to actually look at, not just infer from code. Same "a
verification image the user looks at catches things a numeric check
doesn't" lesson as detection's own validation (see broadcast's project
memory).

Known limitations / unvalidated
--------------------------------
- UPSCALE_FACTOR, the fixed-reference choice, and median-vs-mean are all
  starting points -- not compared against alternatives quantitatively, only
  eyeballed via the compare images.
- No outlier rejection before the median besides the median's own
  robustness -- a bucket contaminated by a systematically wrong
  segmentation (not just occasional noise) would bias the result and this
  wouldn't catch it.
- Badge templates from match6's blue_badges[0] are absent from that bucket
  entirely (0 valid captures out of 47 attempts during collection -- see
  03_calibrate.py's run log) -- the "badge" bucket's other 6 matches' worth
  of instances cover for it, but that specific box was never confirmed to
  be a real badge rather than a decoy.
"""

import argparse, json, pathlib

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
INSTANCES_DIR = DATA_DIR / "digit_instances"
TEMPLATES_DIR = DATA_DIR / "digit_templates"

UPSCALE_FACTOR = 10
MIN_INSTANCES = 5  # buckets thinner than this aren't worth merging


def load_bucket(kind_dir: pathlib.Path) -> list[np.ndarray]:
    invert = kind_dir.parent.name == "clock"
    crops = []
    for f in sorted(kind_dir.glob("*.png")):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            continue
        if invert:
            img = 255 - img
        crops.append(img)
    return crops


def normalize_size(crops: list[np.ndarray]) -> list[np.ndarray]:
    """Resize every crop to the bucket's median (w, h) -- see module
    docstring step 2 for why this is a mild resample, not a distortion."""
    med_h = int(np.median([c.shape[0] for c in crops]))
    med_w = int(np.median([c.shape[1] for c in crops]))
    med_h, med_w = max(med_h, 1), max(med_w, 1)
    return [cv2.resize(c, (med_w, med_h), interpolation=cv2.INTER_CUBIC) for c in crops]


def upsample(crops: list[np.ndarray], factor: int) -> list[np.ndarray]:
    return [cv2.resize(c, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC) for c in crops]


def align_to_reference(imgs: list[np.ndarray]) -> list[np.ndarray]:
    """Sub-pixel-align every image in `imgs` to imgs[0] via Hanning-windowed
    phase correlation, returning the aligned stack (imgs[0] included,
    unshifted)."""
    h, w = imgs[0].shape
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    ref_f = imgs[0].astype(np.float32)
    aligned = [imgs[0]]
    for img in imgs[1:]:
        img_f = img.astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(ref_f * hann, img_f * hann)
        m = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        aligned.append(shifted)
    return aligned


def build_template(crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (merged_template, single_instance_upsampled) both at the same
    scale, for the merged template itself and the before/after compare
    image respectively."""
    normalized = normalize_size(crops)
    up = upsample(normalized, UPSCALE_FACTOR)
    aligned = align_to_reference(up)
    stack = np.stack(aligned).astype(np.float32)
    merged = np.median(stack, axis=0)
    merged = np.clip(merged, 0, 255).astype(np.uint8)
    return merged, up[0]


def make_compare_image(single: np.ndarray, merged: np.ndarray, label: str) -> np.ndarray:
    h = max(single.shape[0], merged.shape[0])
    pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    single_c, merged_c = cv2.cvtColor(pad(single), cv2.COLOR_GRAY2BGR), cv2.cvtColor(pad(merged), cv2.COLOR_GRAY2BGR)
    gap = np.full((h, 20, 3), 30, dtype=np.uint8)
    combined = np.hstack([single_c, gap, merged_c])
    banner = np.full((30, combined.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(banner, f"{label}: SINGLE (cubic upsample)  |  MERGED ({UPSCALE_FACTOR}x aligned median)",
                (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, combined])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=None, help="restrict to these kinds (default: all found)")
    args = ap.parse_args()

    if not INSTANCES_DIR.exists():
        raise SystemExit(f"[error] {INSTANCES_DIR} not found -- run 03_calibrate.py --save first")

    kinds = args.kinds or sorted(d.name for d in INSTANCES_DIR.iterdir() if d.is_dir())
    summary = {}
    for kind in kinds:
        kind_dir = INSTANCES_DIR / kind
        if not kind_dir.is_dir():
            print(f"[skip] kind '{kind}' not found under {INSTANCES_DIR}")
            continue
        out_dir = TEMPLATES_DIR / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        summary[kind] = {}
        for digit_dir in sorted(kind_dir.iterdir()):
            if not digit_dir.is_dir():
                continue
            digit = digit_dir.name
            crops = load_bucket(digit_dir)
            if len(crops) < MIN_INSTANCES:
                print(f"[skip] {kind}/{digit}: only {len(crops)} instance(s), need >= {MIN_INSTANCES}")
                continue
            merged, single = build_template(crops)
            cv2.imwrite(str(out_dir / f"{digit}.png"), merged)
            compare = make_compare_image(single, merged, f"{kind}/{digit}")
            cv2.imwrite(str(out_dir / f"{digit}_compare.png"), compare)
            summary[kind][digit] = len(crops)
            print(f"[ok] {kind}/{digit}: {len(crops)} instances -> {out_dir / f'{digit}.png'}")

    (TEMPLATES_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] summary -> {TEMPLATES_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
