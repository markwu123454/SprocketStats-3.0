#!/usr/bin/env python3
"""
Splits 1920x1080 JPEG images and their YOLO label files into three regions.

Expected input layout (pass the dataset root as the argument):
    <root>/
        images/   *.jpg / *.jpeg
        labels/   *.txt   (YOLO format, one box per line)

Output layout (created automatically):
    <root>/T-split/
        images/   <stem>_top.jpg, <stem>_bottom_left.jpg, <stem>_bottom_right.jpg
        labels/   <stem>_top.txt, <stem>_bottom_left.txt, <stem>_bottom_right.txt

Regions (source-image pixels):
    top:          x=[0,   1920], y=[0,   700]   -> 1920 x 700
    bottom_left:  x=[0,    940], y=[740, 1080]  ->  940 x 340
    bottom_right: x=[980, 1920], y=[740, 1080]  ->  940 x 340

YOLO label format (one box per line, normalized to source image):
    <class> <x_center> <y_center> <width> <height>

For each region:
  1. Convert each box to absolute pixel coords.
  2. Clip to the region rectangle.
  3. Drop boxes with essentially zero area after clipping.
  4. Re-normalize surviving boxes to the region's own width/height.

python tsplit.py raw_data
"""

import argparse
from pathlib import Path
from PIL import Image

# (suffix, x1, y1, x2, y2) in source-image pixel coordinates
REGIONS = [
    ("top",          0,   0,   1920, 700),
    ("bottom_left",  0,   740, 940,  1080),
    ("bottom_right", 980, 740, 1920, 1080),
]

SRC_W, SRC_H = 1920, 1080
MIN_AREA_FRAC = 1e-6  # drop boxes whose clipped area is essentially zero
IMG_EXTS = (".jpg", ".jpeg", ".JPG", ".JPEG")


def split_image(image_path: Path, out_img_dir: Path, stem: str) -> None:
    with Image.open(image_path) as img:
        if img.size != (SRC_W, SRC_H):
            print(f"  warning: {image_path.name} is {img.size}, expected "
                  f"({SRC_W}, {SRC_H}). Proceeding anyway.")
        for suffix, x1, y1, x2, y2 in REGIONS:
            crop = img.crop((x1, y1, x2, y2))
            out_path = out_img_dir / f"{stem}_{suffix}.jpg"
            crop.save(out_path, "JPEG", quality=95)
            print(f"  wrote images/{out_path.name}  "
                  f"({crop.size[0]}x{crop.size[1]})")


def split_labels(label_path: Path, out_lbl_dir: Path, stem: str) -> None:
    """Split a YOLO .txt label file to match the three image regions."""
    with open(label_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    boxes = []
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            print(f"  skipping malformed line in {label_path.name}: {ln!r}")
            continue
        cls = parts[0]
        xc, yc, w, h = map(float, parts[1:])
        abs_w = w * SRC_W
        abs_h = h * SRC_H
        cx = xc * SRC_W
        cy = yc * SRC_H
        xmin = cx - abs_w / 2
        ymin = cy - abs_h / 2
        xmax = cx + abs_w / 2
        ymax = cy + abs_h / 2
        boxes.append((cls, xmin, ymin, xmax, ymax, abs_w * abs_h))

    for suffix, rx1, ry1, rx2, ry2 in REGIONS:
        region_w = rx2 - rx1
        region_h = ry2 - ry1
        out_lines = []
        for cls, xmin, ymin, xmax, ymax, orig_area in boxes:
            cxmin = max(xmin, rx1)
            cymin = max(ymin, ry1)
            cxmax = min(xmax, rx2)
            cymax = min(ymax, ry2)
            if cxmax <= cxmin or cymax <= cymin:
                continue
            clipped_area = (cxmax - cxmin) * (cymax - cymin)
            if clipped_area / max(orig_area, 1e-9) < MIN_AREA_FRAC:
                continue

            lx_center = ((cxmin + cxmax) / 2 - rx1) / region_w
            ly_center = ((cymin + cymax) / 2 - ry1) / region_h
            lw = (cxmax - cxmin) / region_w
            lh = (cymax - cymin) / region_h
            out_lines.append(
                f"{cls} {lx_center:.6f} {ly_center:.6f} {lw:.6f} {lh:.6f}"
            )

        out_path = out_lbl_dir / f"{stem}_{suffix}.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(out_lines))
            if out_lines:
                f.write("\n")
        print(f"  wrote labels/{out_path.name}  ({len(out_lines)} box(es))")


def process_one(image_path: Path, labels_in: Path,
                out_img_dir: Path, out_lbl_dir: Path) -> None:
    stem = image_path.stem
    print(f"processing {image_path.name}")
    split_image(image_path, out_img_dir, stem)

    label_path = labels_in / f"{stem}.txt"
    if label_path.exists():
        split_labels(label_path, out_lbl_dir, stem)
    else:
        print(f"  no label file found at labels/{stem}.txt, skipping labels")


def main():
    ap = argparse.ArgumentParser(
        description="Split 1920x1080 JPEGs and matching YOLO .txt labels "
                    "into three regions. Reads from <root>/images and "
                    "<root>/labels, writes to <root>/T-split/{images,labels}."
    )
    ap.add_argument(
        "root",
        help="Dataset root directory containing images/ and labels/ subfolders."
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    images_in = root / "images"
    labels_in = root / "labels"

    if not images_in.is_dir():
        raise SystemExit(f"error: {images_in} does not exist or is not a directory")
    if not labels_in.is_dir():
        print(f"warning: {labels_in} does not exist; will still split images "
              f"but no labels will be written.")

    out_root = root / "T-split"
    out_img_dir = out_root / "images"
    out_lbl_dir = out_root / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for ext in IMG_EXTS:
        image_paths.extend(images_in.glob(f"*{ext}"))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"no images found in {images_in}")
        return

    print(f"found {len(image_paths)} image(s) in {images_in}")
    print(f"writing output to {out_root}\n")

    for ip in image_paths:
        process_one(ip, labels_in, out_img_dir, out_lbl_dir)


if __name__ == "__main__":
    main()