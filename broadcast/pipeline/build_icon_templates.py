#!/usr/bin/env python3
"""
Build icon-matching templates from sourced reference art (NOT harvested from
broadcast video). Unlike digit_templates (03_calibrate.py + 04_build_templates.py,
which fuse hundreds of low-res broadcast-rendered instances because the font
and rendering scale vary by broadcast), the FRC game-piece pictogram next to
each counter badge is a small, fixed, officially-published set -- one clean
upscaled source image per icon is enough. No harvesting step exists or is
needed for these.

This script only builds templates from whatever sits in icon_templates_src/;
it does not decide where those source images come from. Sourcing them
(finding an upscaled version of the official icon) is a manual, one-off step
done outside this pipeline.

Algorithm
---------
1. Locate the icon's white square within the sourced image and crop tightly
   to it. The source images are screenshots taken with deliberate margin
   around the square (easier to crop generously by hand than pixel-perfect),
   so this step does the precise crop: largest connected component of
   near-white pixels (high value, low saturation -- same test as
   06_identify.py's hue_class "white" bucket), bounding box of that
   component.
2. Within the crop, threshold the icon ink out of the white background via
   per-image Otsu on grayscale. This is deliberately NOT a fixed brightness
   cutoff: the icon's ink color differs per source (blue dots, red coral,
   red algae seen so far) and Otsu adapts to each image's own bg/fg
   separation instead of assuming one. Algae's fine internal swirl linework
   is kept, not filled solid -- it's part of the source art's actual
   silhouette, same as any other icon's shape detail.
3. Save the binary mask -- this is the template. Matching a broadcast icon
   crop against it (07, not written yet) means binarizing that crop the same
   way and comparing masks, not raw color, since the pictogram renders in
   whichever alliance's color it sits on.

Output
------
data/icon_templates/<name>.png            -- binary mask (0/255), the template
data/icon_templates/<name>_compare.png    -- source screenshot / tight crop /
                                              mask, side by side, so a bad
                                              crop or threshold is something
                                              to look at, not just trust
Known limitations / unvalidated
--------------------------------
- Only 3 source images exist so far (2025 coral, 2025 algae, 2026's single
  icon -- name unconfirmed, see icon_templates_src/). Whether Otsu holds up
  on a wider set of sourced images, and whether the sourced art's silhouette
  actually matches the broadcast production's redrawn icon closely enough to
  match on, is NOT yet checked against a real broadcast crop -- do that
  before wiring up 07.
- White-square localization assumes the square is the single largest
  near-white connected region in the source image. Fine for a cropped
  screenshot with only one icon in frame; would need reworking if source
  images ever contain more than one icon.

Usage
-----
  python pipeline/build_icon_templates.py
"""

import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
SRC_DIR = DATA_DIR / "icon_templates_src"
OUT_DIR = DATA_DIR / "icon_templates"

# Near-white test for locating the square: bright and low-saturation. Same
# thresholds as 06_identify.py's hue_class "white" bucket, since it's the
# same visual thing (a white chip) seen at higher resolution here.
WHITE_VAL_MIN = 160
WHITE_SAT_MAX = 40

# Shrink find_white_square's bbox by this many px on each side before
# thresholding, to drop the antialiased blend ring at its edge (measured
# ~4px wide on algae.png). See build_one's comment for why this matters.
CROP_INSET = 4


def find_white_square(img):
    """-> (x, y, w, h) of the largest near-white connected region, or None."""
    b, g, r = cv2.split(img.astype(np.int32))
    mx = np.maximum(np.maximum(b, g), r)
    mn = np.minimum(np.minimum(b, g), r)
    sat = mx - mn
    white = ((mx > WHITE_VAL_MIN) & (sat < WHITE_SAT_MAX)).astype(np.uint8) * 255
    # Close small gaps (anti-aliased ink strokes crossing near the border)
    # so the square doesn't get fragmented into several smaller components.
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[idx]
    return int(x), int(y), int(w), int(h)


def ink_mask(crop_bgr):
    """-> binary mask (0/255), icon ink = 255. Per-image Otsu, not a fixed
    cutoff, because ink color varies by icon (blue/red/whatever) but is
    always darker/more saturated than the white square it sits on."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


# A stray component this much smaller than the largest one is noise, not a
# disjoint part of the icon -- fuel's 6 dots come out within ~1% of each
# other's area, so a real generously-sized threshold here won't confuse a
# legitimate small part for a stray.
STRAY_AREA_RATIO = 0.15


def check_mask(mask):
    """Programmatic replacement for eyeballing a zoomed PNG -- returns a list
    of warning strings, empty if clean. Catches exactly the class of defect
    that a visual check missed here: a thin ink ring at the crop's edge
    (border-touching ink) and small stray flecks (components tiny relative
    to the icon's main body)."""
    warnings = []
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    n_border_ink = int((border > 0).sum())
    if n_border_ink:
        warnings.append(f"{n_border_ink}/{len(border)} border px are ink "
                         f"(icon should sit inset from the crop edge)")
    n_comp, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
    if areas:
        strays = [a for a in areas[1:] if a < STRAY_AREA_RATIO * areas[0]]
        if strays:
            warnings.append(f"{len(strays)} stray component(s) below "
                             f"{STRAY_AREA_RATIO:.0%} of main area {areas[0]}: {strays}")
    return warnings


def build_one(src_path: pathlib.Path):
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"[skip] {src_path.name}: not readable as an image", file=sys.stderr)
        return
    box = find_white_square(img)
    if box is None:
        print(f"[skip] {src_path.name}: no white square found", file=sys.stderr)
        return
    x, y, w, h = box
    # find_white_square's bbox edge sits ON the antialiased blend ring
    # between the white square and its surrounding margin (measured on
    # algae.png: ~172-179 gray right at the edge vs ~243 a few px in) --
    # left in, that ring reads as ink to Otsu and can surround the crop
    # completely, which made fill_holes see no reachable background and
    # flood-fill the entire mask solid. Insetting past the ring fixes it.
    x, y = x + CROP_INSET, y + CROP_INSET
    w, h = max(1, w - 2 * CROP_INSET), max(1, h - 2 * CROP_INSET)
    crop = img[y:y + h, x:x + w]
    mask = ink_mask(crop)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = src_path.stem
    cv2.imwrite(str(OUT_DIR / f"{name}.png"), mask)

    # Compare strip: source (resized to crop height) | crop | mask, so a bad
    # square-localization or a threshold that ate half the icon is visible
    # at a glance instead of trusted from a shape check.
    src_scaled = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    pad = np.full((h, 4, 3), 128, np.uint8)
    strip = np.hstack([src_scaled, pad, crop, pad, mask_bgr])
    cv2.imwrite(str(OUT_DIR / f"{name}_compare.png"), strip)

    warnings = check_mask(mask)
    status = "ok" if not warnings else "WARN"
    print(f"[{status}] {src_path.name} -> {name}.png  square={w}x{h}", file=sys.stderr)
    for w_ in warnings:
        print(f"    ! {w_}", file=sys.stderr)


def main():
    if not SRC_DIR.exists():
        print(f"[error] {SRC_DIR} does not exist", file=sys.stderr)
        return
    sources = sorted(p for p in SRC_DIR.iterdir()
                     if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not sources:
        print(f"[error] no source images in {SRC_DIR}", file=sys.stderr)
        return
    for src in sources:
        build_one(src)


if __name__ == "__main__":
    main()
