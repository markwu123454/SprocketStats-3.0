#!/usr/bin/env python3
"""
Step 2 — Train YOLO26 robot detector.

Requires: ultralytics (pip install ultralytics)
Requires: data/dataset.yaml created by 01_pull.py

Usage:
  python 02_train.py
  python 02_train.py --resume       # resume from last checkpoint
  python 02_train.py --model yolo26n.pt  # override model size
  python 02_train.py --cutmix 0.15 --translate 0.2  # occlusion-robustness augmentation
"""
import argparse, pathlib, sys
import yaml

ROOT = pathlib.Path(__file__).parent.parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml",
                    help="config file under the yolo/ root (default: %(default)s)")
    # parse --config first so its value is known before the rest of the
    # config-dependent defaults (--model) are built
    cfg_name = ap.parse_known_args()[0].config
    CFG_PATH = ROOT / cfg_name
    CFG = yaml.safe_load(open(CFG_PATH))
    DATASET_YAML = ROOT / CFG["paths"]["dataset_yaml"]
    RUNS_DIR     = ROOT / CFG["paths"]["runs_dir"]

    _default_model = CFG["train"]["model"]
    _default_model = str(ROOT / _default_model) if (ROOT / _default_model).exists() else _default_model
    ap.add_argument("--model",  default=_default_model,
                    help="pretrained weights (default: %(default)s)")
    ap.add_argument("--name",   default="robot_detection",
                    help="run directory name under runs_dir (default: robot_detection)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from last.pt in the run directory")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override config.yaml train.epochs")
    ap.add_argument("--cutmix", type=float, default=0.0,
                    help="CutMix probability — pastes a rectangular patch from another "
                         "training image, the built-in stand-in for occlusion since "
                         "copy_paste requires segmentation masks we don't have")
    ap.add_argument("--translate", type=float, default=0.1,
                    help="translation augmentation fraction — raising this pushes more "
                         "boxes toward/past frame edges, simulating frame-boundary crops")
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
            epochs=args.epochs if args.epochs is not None else t["epochs"],
            imgsz=t["imgsz"],
            batch=t["batch"],
            device=t["device"],
            seed=t["seed"],
            project=str(RUNS_DIR),
            name=args.name,
            exist_ok=True,
            cutmix=args.cutmix,
            translate=args.translate,
        )

    print("[done] training complete")


if __name__ == "__main__":
    main()
