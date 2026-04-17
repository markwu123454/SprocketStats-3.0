"""
ls_to_yolo.py
-------------
Convert a Label Studio export (JSON or JSON-MIN) of FRC robot annotations
into a YOLO-format dataset ready for Ultralytics training (YOLOv8/v11/v26 etc.).

Output layout:
    <out>/
        data.yaml
        images/
            train/...
            val/...
        labels/
            train/...   (.txt files, one line per box: class cx cy w h — all normalized)
            val/...

Usage:
    # If you have the JSON-MIN export AND the source images folder:
    python ls_to_yolo.py export.json --images sampled_frames --out yolo_dataset

    # Split ratio (default 0.2 = 20% val):
    python ls_to_yolo.py export.json --images sampled_frames --val-split 0.15

Notes:
    - Label Studio's "JSON-MIN" export is the easiest to work with; this script
      auto-detects whether you've given it JSON or JSON-MIN.
    - Images are COPIED into the dataset so the original sampled_frames stays untouched.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


# Keep this order stable — it defines the class indices YOLO will train against.
CLASS_NAMES = ["Red Alliance Robot", "Blue Alliance Robot"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def resolve_image_filename(raw: str) -> str:
    """
    Label Studio stores image references in a few different ways depending on how
    the task was imported. Pull out just the basename so we can match it against
    files in --images.
    """
    # Strip any URL prefix like /data/upload/1/ or http://localhost:8080/...
    path = urlparse(raw).path if "://" in raw else raw
    name = Path(unquote(path)).name
    # Label Studio sometimes prepends a hash like "a1b2c3d4-frame_0012.jpg"
    # We'll try the raw name first when matching, then a stripped version.
    return name


def candidate_names(raw_name: str):
    """Yield basename variants to try when locating the source image."""
    yield raw_name
    # LS hash prefix pattern: "<8hex>-<original>.ext"
    if "-" in raw_name:
        stripped = raw_name.split("-", 1)[1]
        if stripped != raw_name:
            yield stripped


def find_source_image(raw_ref: str, images_dir: Path) -> Path | None:
    base = resolve_image_filename(raw_ref)
    for name in candidate_names(base):
        candidate = images_dir / name
        if candidate.exists():
            return candidate
    # Last resort: glob for any file ending with the stripped name
    for name in candidate_names(base):
        matches = list(images_dir.rglob(f"*{name}"))
        if matches:
            return matches[0]
    return None


def is_jsonmin_format(data) -> bool:
    """JSON-MIN looks like: [{"image": "...", "label": [ {x,y,...}, ... ]}, ...]"""
    if not isinstance(data, list) or not data:
        return False
    sample = data[0]
    return isinstance(sample, dict) and "label" in sample and "annotations" not in sample


def iter_tasks_jsonmin(data):
    """Yield (image_ref, [box_dict, ...]) from JSON-MIN format."""
    for task in data:
        # The image field name depends on the labeling config's $variable — we used $image
        image_ref = task.get("image") or task.get("img") or ""
        boxes = task.get("label") or []
        yield image_ref, boxes


def iter_tasks_full(data):
    """Yield (image_ref, [box_dict, ...]) from the full JSON export."""
    for task in data:
        image_ref = ""
        if "data" in task and isinstance(task["data"], dict):
            # Grab the first string value from data — typically "image"
            for v in task["data"].values():
                if isinstance(v, str):
                    image_ref = v
                    break

        boxes = []
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            for result in ann.get("result", []):
                if result.get("type") != "rectanglelabels":
                    continue
                val = result.get("value", {})
                labels = val.get("rectanglelabels", [])
                if not labels:
                    continue
                boxes.append({
                    "x": val.get("x", 0),
                    "y": val.get("y", 0),
                    "width": val.get("width", 0),
                    "height": val.get("height", 0),
                    "rectanglelabels": labels,
                })
        yield image_ref, boxes


def box_to_yolo_line(box: dict) -> str | None:
    """
    Label Studio stores rectangle coords as percentages (0-100) of image dims:
      x, y = top-left corner; width, height = box size.
    YOLO wants: class_id cx cy w h — all fractions (0-1) of image dims.
    """
    labels = box.get("rectanglelabels") or []
    if not labels:
        return None
    cls = labels[0]
    if cls not in CLASS_TO_ID:
        print(f"  ! Skipping unknown label '{cls}'", file=sys.stderr)
        return None

    x = float(box["x"]) / 100.0
    y = float(box["y"]) / 100.0
    w = float(box["width"]) / 100.0
    h = float(box["height"]) / 100.0
    cx = x + w / 2.0
    cy = y + h / 2.0

    # Clamp to [0,1] just in case of floating-point overruns on edge boxes
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)

    return f"{CLASS_TO_ID[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Label Studio export → YOLO dataset.")
    ap.add_argument("export", type=Path, help="Label Studio export .json file")
    ap.add_argument("--images", type=Path, required=True,
                    help="Folder containing the source images referenced by the export")
    ap.add_argument("--out", type=Path, default=Path("yolo_dataset"),
                    help="Output dataset directory (default: yolo_dataset)")
    ap.add_argument("--val-split", type=float, default=0.2,
                    help="Fraction of images to put in val set (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = ap.parse_args()

    if not args.export.is_file():
        sys.exit(f"Export file not found: {args.export}")
    if not args.images.is_dir():
        sys.exit(f"Images folder not found: {args.images}")
    if not 0.0 <= args.val_split < 1.0:
        sys.exit("--val-split must be in [0, 1)")

    with args.export.open() as f:
        data = json.load(f)

    task_iter = iter_tasks_jsonmin(data) if is_jsonmin_format(data) else iter_tasks_full(data)

    # Build the list of (source_image_path, [yolo_lines]) first so we can split cleanly
    prepared = []
    skipped_missing = 0
    skipped_empty = 0
    for image_ref, boxes in task_iter:
        src = find_source_image(image_ref, args.images)
        if src is None:
            print(f"  ! Source image not found for '{image_ref}' — skipping",
                  file=sys.stderr)
            skipped_missing += 1
            continue
        lines = [ln for ln in (box_to_yolo_line(b) for b in boxes) if ln]
        if not lines:
            # Keeping images with zero labels would teach the model they contain no
            # robots — which is fine if intentional, but usually it means the task
            # wasn't annotated yet. Skip by default.
            skipped_empty += 1
            continue
        prepared.append((src, lines))

    if not prepared:
        sys.exit("No usable annotations found. Check the export and --images path.")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(prepared)
    n_val = int(len(prepared) * args.val_split)
    val_items = prepared[:n_val]
    train_items = prepared[n_val:]

    # Wipe & recreate output layout
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    def write_split(items, split_name: str):
        for src, lines in items:
            dst_img = args.out / "images" / split_name / src.name
            dst_lbl = args.out / "labels" / split_name / (src.stem + ".txt")
            shutil.copy2(src, dst_img)
            dst_lbl.write_text("\n".join(lines) + "\n")

    write_split(train_items, "train")
    write_split(val_items, "val")

    # data.yaml — path is absolute so training works from any CWD
    yaml_text = (
        f"# FRC robot detection dataset\n"
        f"path: {args.out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    (args.out / "data.yaml").write_text(yaml_text)

    print("\nDone.")
    print(f"  train images: {len(train_items)}")
    print(f"  val images:   {len(val_items)}")
    if skipped_missing:
        print(f"  skipped (image not found): {skipped_missing}")
    if skipped_empty:
        print(f"  skipped (no boxes):        {skipped_empty}")
    print(f"  dataset:      {args.out.resolve()}")
    print(f"  config:       {(args.out / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()