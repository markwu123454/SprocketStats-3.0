#!/usr/bin/env python3
"""
Step 2 viz (interactive) — eyeball best.pt on sample frames.

Loads checkpoints/best.pt and runs it over frames listed in
exports/manifest.jsonl, drawing predicted keypoints (filled dots) next to
ground-truth keypoints (hollow circles, where labeled) in a matplotlib window.

Uses the cfg SAVED INSIDE the checkpoint (backbone/heatmap params) rather
than the live config.yaml, since the two can drift out of sync after retrains.

matplotlib (not cv2) drives the interactive window because this venv's
opencv is opencv-python-headless (no GUI backend, required by
label-studio-sdk) — cv2 is only used here for image I/O.

Controls:
  d / Right / Space   next frame
  a / Left            previous frame
  r                   jump to a random frame
  [ / ]               lower / raise peak_threshold by 0.05
  g                   toggle ground-truth overlay
  q / Esc             quit

Usage:
  python 02_view.py                      # every frame in the manifest
  python 02_view.py --match 2022alhu_qm20
  python 02_view.py --unlabeled-only      # skip frames with no GT points
  python 02_view.py --shuffle
"""
import argparse, json, pathlib, random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

import sys as _sys
ROOT = pathlib.Path(__file__).parent.parent
_sys.path.insert(0, str(ROOT))

from model import HeatmapNet, decode_peaks, CLASSES
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLOR = {"blue": "tab:blue", "red": "tab:red"}
CH_TO_CLS = {i: c for i, c in enumerate(CLASSES)}


def load_model_and_cfg(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ck["cfg"]
    hm = cfg["heatmap"]
    model = HeatmapNet(cfg["train"]["backbone"], len(CLASSES), hm["output_stride"]).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[load] {ckpt_path.name} epoch={ck.get('epoch')} backbone={cfg['train']['backbone']}")
    return model, hm


def load_records(manifest, match=None, unlabeled_only=False):
    recs = []
    for line in open(manifest, encoding="utf-8"):
        r = json.loads(line)
        if match and r["match"] != match:
            continue
        if unlabeled_only and r["points"]:
            continue
        recs.append(r)
    return recs


@torch.no_grad()
def predict(model, hm, img_bgr, threshold):
    H0, W0 = img_bgr.shape[:2]
    in_h, in_w = hm["input_size"]
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_r = cv2.resize(img, (in_w, in_h))
    t = torch.from_numpy(img_r).permute(2, 0, 1).float().to(DEVICE) / 255.0
    t = (t - torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)) \
        / torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)
    logits = model(t.unsqueeze(0))
    dets = decode_peaks(logits, threshold, hm["nms_kernel"], hm["max_instances"])[0]
    out = []
    sx, sy = W0 / in_w, H0 / in_h
    for x_hm, y_hm, c, score in dets:
        x_in = x_hm * hm["output_stride"]
        y_in = y_hm * hm["output_stride"]
        out.append((x_in * sx, y_in * sy, CH_TO_CLS[c], score))
    return out


class Viewer:
    def __init__(self, model, hm, records, threshold, shuffle):
        self.model = model
        self.hm = hm
        self.records = records
        if shuffle:
            random.shuffle(self.records)
        self.threshold = threshold
        self.show_gt = True
        self.idx = 0

        self.fig, self.ax = plt.subplots(figsize=(11, 7))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.subplots_adjust(top=0.90, bottom=0.02, left=0.02, right=0.98)
        self.render()

    def load_and_predict(self):
        rec = self.records[self.idx]
        img = cv2.imread(rec["image"])
        if img is None:
            return None, None, rec
        preds = predict(self.model, self.hm, img, self.threshold)
        return img, preds, rec

    def render(self):
        img, preds, rec = self.load_and_predict()
        self.ax.clear()
        self.ax.axis("off")
        if img is None:
            self.ax.text(0.5, 0.5, f"could not read:\n{rec['image']}",
                        ha="center", va="center", transform=self.ax.transAxes)
            self.fig.canvas.draw_idle()
            return

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.ax.imshow(rgb)

        if self.show_gt:
            for p in rec["points"]:
                self.ax.add_patch(plt.Circle((p["x_px"], p["y_px"]), 12,
                                             fill=False, edgecolor=COLOR.get(p["cls"], "lime"),
                                             linewidth=2))
        for x, y, cls, score in preds:
            self.ax.plot(x, y, "o", color=COLOR.get(cls, "lime"), markersize=8)
            self.ax.annotate(f"{score:.2f}", (x, y), xytext=(6, 6),
                            textcoords="offset points", color=COLOR.get(cls, "lime"), fontsize=8)

        name = pathlib.Path(rec["image"]).name
        self.fig.suptitle(
            f"[{self.idx + 1}/{len(self.records)}] {rec['match']} / {name}   "
            f"thr={self.threshold:.2f}  gt={'on' if self.show_gt else 'off'}\n"
            f"d/Right next   a/Left prev   r random   [ ] threshold   g gt   q quit",
            fontsize=9)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(self.fig)
        elif event.key in ("d", " ", "right"):
            self.idx = (self.idx + 1) % len(self.records)
            self.render()
        elif event.key in ("a", "left"):
            self.idx = (self.idx - 1) % len(self.records)
            self.render()
        elif event.key == "r":
            self.idx = random.randrange(len(self.records))
            self.render()
        elif event.key == "g":
            self.show_gt = not self.show_gt
            self.render()
        elif event.key == "[":
            self.threshold = max(0.05, self.threshold - 0.05)
            self.render()
        elif event.key == "]":
            self.threshold = min(0.95, self.threshold + 0.05)
            self.render()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/frc/exports/manifest.jsonl"))
    ap.add_argument("--ckpt", default=str(ROOT / "data/frc/checkpoints/best.pt"))
    ap.add_argument("--match", default=None, help="only show frames from this match")
    ap.add_argument("--unlabeled-only", action="store_true")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the checkpoint's peak_threshold")
    args = ap.parse_args()

    model, hm = load_model_and_cfg(pathlib.Path(args.ckpt))
    threshold = args.threshold if args.threshold is not None else hm["peak_threshold"]

    records = load_records(args.manifest, match=args.match, unlabeled_only=args.unlabeled_only)
    if not records:
        raise SystemExit("[abort] no frames matched the given filters")
    print(f"[data] {len(records)} frames loaded")

    Viewer(model, hm, records, threshold, args.shuffle)
    plt.show()


if __name__ == "__main__":
    main()
