#!/usr/bin/env python3
"""
Step 3 — Predict on tasks and write predictions back to Label Studio.

This is the model-assisted / pre-labeling loop. Because the built-in
Ultralytics ML backend speaks box+associated-keypoint pose (which we do NOT
use), we push predictions directly via the SDK's predictions.create in the
EXACT result format the project's KeyPointLabels config expects.

Prediction result format (from the project's own import example):
  {
    "from_name": "kp", "to_name": "image", "type": "keypointlabels",
    "value": {"x": <percent>, "y": <percent>,
              "keypointlabels": ["Blue robot center"]}
  }
Note x/y are PERCENT of image dimensions, not pixels.

Usage:
  python 03_predict.py --only-unannotated   # pre-label tasks with no human label
"""
import os, argparse, pathlib, yaml, math
import numpy as np
import torch
import cv2
from label_studio_sdk import LabelStudio
from dotenv import load_dotenv

# Load credentials from a .env file next to config.yaml (LS_API_KEY, R2_*)
load_dotenv(pathlib.Path(__file__).parent / ".env")

from model import HeatmapNet, decode_peaks, CLASSES

CFG = yaml.safe_load(open(pathlib.Path(__file__).parent / "config.yaml"))
LS = CFG["label_studio"]; HM = CFG["heatmap"]; P = CFG["paths"]; R2 = CFG["r2"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CH_TO_LABEL = {0: LS["label_blue"], 1: LS["label_red"]}
IMG_ROOT = pathlib.Path(R2["local_image_root"])


def load_model():
    ck = torch.load(pathlib.Path(P["checkpoints"]) / "best.pt", map_location=DEVICE)
    model = HeatmapNet(CFG["train"]["backbone"], len(CLASSES), HM["output_stride"]).to(DEVICE)
    model.load_state_dict(ck["model"]); model.eval()
    return model


@torch.no_grad()
def predict_image(model, img_bgr):
    H0, W0 = img_bgr.shape[:2]
    in_h, in_w = HM["input_size"]
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_r = cv2.resize(img, (in_w, in_h))
    t = torch.from_numpy(img_r).permute(2, 0, 1).float().to(DEVICE) / 255.0
    t = (t - torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)) \
        / torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)
    logits = model(t.unsqueeze(0))
    dets = decode_peaks(logits, HM["peak_threshold"], HM["nms_kernel"],
                        HM["max_instances"])[0]
    # heatmap px -> original-image PERCENT
    out = []
    for x_hm, y_hm, c, score in dets:
        x_in = x_hm * HM["output_stride"]      # back to network-input px
        y_in = y_hm * HM["output_stride"]
        x_pct = x_in / in_w * 100.0
        y_pct = y_in / in_h * 100.0
        out.append((x_pct, y_pct, c, score))
    return out


def to_ls_results(dets):
    results = []
    scores = []
    for x_pct, y_pct, c, score in dets:
        results.append({
            "from_name": LS["from_name"],
            "to_name": LS["to_name"],
            "type": "keypointlabels",
            "value": {"x": x_pct, "y": y_pct,
                      "keypointlabels": [CH_TO_LABEL[c]]},
        })
        scores.append(score)
    avg = float(np.mean(scores)) if scores else 0.0
    return results, avg


def local_path_for_task(task):
    """Resolve task image to local mirror. Handles LS resolver URLs whose
    fileuri query param is base64 of s3://bucket/key."""
    import urllib.parse, base64
    raw = task.data["image"] if hasattr(task, "data") else task["data"]["image"]
    if "fileuri=" in raw:
        q = urllib.parse.urlparse(raw).query
        vals = urllib.parse.parse_qs(q).get("fileuri")
        if vals:
            b = vals[0]; b += "=" * (-len(b) % 4)
            try:
                decoded = base64.b64decode(b).decode()
            except Exception:
                decoded = base64.urlsafe_b64decode(b).decode()
            if decoded.startswith("s3://"):
                _, key = decoded[5:].split("/", 1)
                return IMG_ROOT / key
    if raw.startswith("s3://"):
        _, _, key = raw[5:].partition("/")
        return IMG_ROOT / key
    key = urllib.parse.urlparse(raw).path.lstrip("/")
    if key.startswith(R2["bucket"] + "/"):
        key = key[len(R2["bucket"]) + 1:]
    return IMG_ROOT / urllib.parse.unquote(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-unannotated", action="store_true",
                    help="skip tasks that already have a human annotation")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    client = LabelStudio(base_url=LS["base_url"], api_key=os.environ["LS_API_KEY"])
    model = load_model()

    n = 0
    # tasks.list paginates; iterate the whole project
    for task in client.tasks.list(project=LS["project_id"]):
        if args.only_unannotated and getattr(task, "total_annotations", 0):
            continue
        local = local_path_for_task(task)
        if not local.exists():
            # fetch on demand from R2 if not mirrored
            continue
        img = cv2.imread(str(local))
        if img is None:
            continue
        dets = predict_image(model, img)
        results, avg = to_ls_results(dets)
        client.predictions.create(
            task=task.id,
            project=LS["project_id"],
            model_version=LS["model_version"],
            score=avg,
            result=results,
        )
        n += 1
        if n % 500 == 0:
            print(f"[predict] pushed {n} predictions")
        if args.limit and n >= args.limit:
            break
    print(f"[done] pushed {n} predictions as model_version={LS['model_version']}")


if __name__ == "__main__":
    main()
