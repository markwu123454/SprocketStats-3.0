#!/usr/bin/env python3
"""
Step 2 — Train YOLO26 robot detector.

Requires: ultralytics (pip install ultralytics)
Requires: data/dataset.yaml created by 01_pull.py

Usage:
  python 02_train.py
  python 02_train.py --resume       # resume from last checkpoint
  python 02_train.py --model yolo26n.pt  # override model size
"""
import argparse, pathlib, sys
import yaml

CFG_PATH = pathlib.Path(__file__).parent / "config.yaml"
CFG = yaml.safe_load(open(CFG_PATH))

ROOT        = pathlib.Path(__file__).parent
DATASET_YAML = ROOT / CFG["paths"]["dataset_yaml"]
RUNS_DIR     = ROOT / CFG["paths"]["runs_dir"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default=CFG["train"]["model"],
                    help="pretrained weights (default: %(default)s)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from last.pt in the run directory")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[error] ultralytics not installed — run: pip install ultralytics")

    if not DATASET_YAML.exists():
        sys.exit(f"[error] {DATASET_YAML} not found — run 01_pull.py first")

    t = CFG["train"]

    if args.resume:
        last = sorted(RUNS_DIR.rglob("last.pt"), key=lambda p: p.stat().st_mtime)
        if not last:
            sys.exit("[error] no last.pt found to resume from")
        ckpt = str(last[-1])
        print(f"[resume] {ckpt}")
        model = YOLO(ckpt)
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
            name="robot_detection",
            exist_ok=True,
        )

    print("[done] training complete")


if __name__ == "__main__":
    main()
