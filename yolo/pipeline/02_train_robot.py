#!/usr/bin/env python3
"""
Train single-class robot detector (blue + red merged → 'robot').

Reuses images already synced by 01_pull.py, remaps label files on the fly
into data/labels_robot/{train,val}/ and writes data/dataset_robot.yaml.

Usage:
  python 02_train_robot.py
  python 02_train_robot.py --resume
  python 02_train_robot.py --model yolo26s.pt
"""
import argparse, pathlib, shutil, sys
import yaml

ROOT     = pathlib.Path(__file__).parent.parent
CFG_PATH = ROOT / "config.yaml"
CFG      = yaml.safe_load(open(CFG_PATH))

DATASET_YAML  = ROOT / "data" / "dataset_robot.yaml"
RUNS_DIR      = ROOT / CFG["paths"]["runs_dir"]
LABEL_SRC     = ROOT / "data" / "labels"
LABEL_DST     = ROOT / "data" / "labels_robot"


def remap_labels():
    """Copy label files with all class ids → 0 (robot)."""
    remapped = 0
    for split in ("train", "val"):
        src_dir = LABEL_SRC / split
        dst_dir = LABEL_DST / split
        if not src_dir.exists():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.glob("*.txt"):
            lines = src.read_text().splitlines()
            new_lines = []
            for line in lines:
                parts = line.split()
                if parts:
                    parts[0] = "0"   # any alliance → robot
                    new_lines.append(" ".join(parts))
            (dst_dir / src.name).write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
            remapped += 1
    print(f"[remap] {remapped} label files → labels_robot/")


def write_dataset_yaml():
    """
    Ultralytics always looks for labels in a sibling 'labels/' next to 'images/'.
    Stage images + remapped labels into data/staging_robot/ so the layout is correct.
    Dataset is tiny (44 frames) so copying is fine.
    """
    staging = ROOT / "data" / "staging_robot"
    for split in ("train", "val"):
        dst_img = staging / "images" / split
        dst_lbl = staging / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for img in (ROOT / "data" / "images" / split).glob("*.jpg"):
            shutil.copy2(img, dst_img / img.name)
        for lbl in (ROOT / "data" / "labels_robot" / split).glob("*.txt"):
            shutil.copy2(lbl, dst_lbl / lbl.name)

    ds = {
        "path":  str(staging),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["robot"],
    }
    with open(DATASET_YAML, "w") as f:
        yaml.dump(ds, f, default_flow_style=False)
    print(f"[dataset] {DATASET_YAML}")


def main():
    ap = argparse.ArgumentParser()
    _default_model = CFG["train"]["model"].replace("m.pt", "n.pt")
    _default_model = str(ROOT / _default_model) if (ROOT / _default_model).exists() else _default_model
    ap.add_argument("--model",  default=_default_model)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[error] pip install ultralytics")

    src_labels = ROOT / "data" / "labels"
    if not src_labels.exists():
        sys.exit("[error] data/labels/ not found — run 01_pull.py first")

    remap_labels()
    write_dataset_yaml()

    t = CFG["train"]

    if args.resume:
        last = sorted(RUNS_DIR.rglob("last.pt"), key=lambda p: p.stat().st_mtime)
        if not last:
            sys.exit("[error] no last.pt found")
        model = YOLO(str(last[-1]))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=str(DATASET_YAML),
            epochs=t["epochs"],
            imgsz=t["imgsz"],
            batch=t["batch"],
            device=t["device"],
            seed=t["seed"],
            project=str(RUNS_DIR),
            name="robot_detection_1cls",
            exist_ok=True,
        )

    print("[done]")


if __name__ == "__main__":
    main()
