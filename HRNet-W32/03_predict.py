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
import os, argparse, pathlib, yaml, math, threading, queue
import numpy as np
import torch
import cv2
import boto3
from botocore.config import Config as BotoConfig
from label_studio_sdk import LabelStudio
from dotenv import load_dotenv

# Load credentials from a .env file next to config.yaml (LS_API_KEY, R2_*)
load_dotenv(pathlib.Path(__file__).parent / ".env")

from model import HeatmapNet, decode_peaks, CLASSES

CFG = yaml.safe_load(open(pathlib.Path(__file__).parent / "config.yaml"))
LS = CFG["label_studio"]; HM = CFG["heatmap"]; P = CFG["paths"]; R2 = CFG["r2"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CH_TO_LABEL = {0: LS["label_blue"], 1: LS["label_red"]}
_SCRIPT_DIR = pathlib.Path(__file__).parent
IMG_ROOT = (_SCRIPT_DIR / R2["local_image_root"]).resolve()


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


def to_ls_results(dets, orig_w, orig_h):
    results = []
    scores = []
    for x_pct, y_pct, c, score in dets:
        results.append({
            "from_name": LS["from_name"],
            "to_name": LS["to_name"],
            "type": "keypointlabels",
            "value": {
                "x": x_pct,
                "y": y_pct,
                "width": 0.15,          # marker size (~2px on 1280-wide image)
                "keypointlabels": [CH_TO_LABEL[c]],
            },
            "original_width": orig_w,
            "original_height": orig_h,
        })
        scores.append(score)
    avg = float(np.min(scores)) if scores else 0.0
    return results, avg


def make_r2():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=BotoConfig(signature_version="s3v4", retries={"max_attempts": 3}),
    )


def fetch_from_r2(s3, local: pathlib.Path, key: str):
    local.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(R2["bucket"], key, str(local))


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


_STOP = object()  # sentinel to signal end-of-queue


def _fetcher(task_iter, fetch_q, args, counters):
    """Thread: iterate LS tasks, apply filters, download images, enqueue."""
    s3 = make_r2()
    first = True
    for task in task_iter:
        if first:
            print(f"[fetch] first task received (id={task.id})")
            first = False

        if args.only_unannotated and getattr(task, "total_annotations", 0):
            counters["skip_annotated"] += 1
            continue
        if args.only_unpredicted and getattr(task, "total_predictions", 0):
            counters["skip_predicted"] += 1
            continue

        local = local_path_for_task(task)
        if not local.exists():
            try:
                key = local.relative_to(IMG_ROOT).as_posix()
            except ValueError:
                key = "/".join(local.parts[-4:])
            try:
                fetch_from_r2(s3, local, key)
            except Exception as e:
                counters["skip_fetch_fail"] += 1
                print(f"[fetch] R2 error {key}: {e}")
                continue

        img = cv2.imread(str(local))
        if img is None:
            counters["skip_unreadable"] += 1
            print(f"[fetch] unreadable: {local}")
            continue

        fetch_q.put((task.id, img, local.name))

        if args.limit and counters["fetched"] + fetch_q.qsize() >= args.limit:
            break

    fetch_q.put(_STOP)


def _uploader(upload_q, client):
    """Thread: pull (task_id, results, avg) and push to Label Studio."""
    while True:
        item = upload_q.get()
        if item is _STOP:
            break
        task_id, results, avg = item
        try:
            client.predictions.create(
                task=task_id,
                model_version=LS["model_version"],
                score=avg,
                result=results,
            )
            print(f"  [upload] task {task_id} pushed (score={avg:.3f})")
        except Exception as e:
            print(f"  [upload] task {task_id} FAILED: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-unannotated", action="store_true",
                    help="skip tasks that already have a human annotation")
    ap.add_argument("--only-unpredicted", action="store_true",
                    help="skip tasks that already have a model prediction")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after pushing this many predictions (0 = no limit)")
    args = ap.parse_args()

    print("[init] connecting to Label Studio...")
    client = LabelStudio(base_url=LS["base_url"], api_key=os.environ["LS_API_KEY"])
    print("[init] loading model checkpoint...")
    model = load_model()
    print(f"[init] model loaded on {DEVICE}, starting pipeline...")

    counters = {"fetched": 0, "skip_annotated": 0, "skip_predicted": 0,
                "skip_fetch_fail": 0, "skip_unreadable": 0, "pushed": 0}

    # fetch_q: (task_id, img_array, filename)  — bounded to cap memory use
    # upload_q: (task_id, ls_results, avg_score)
    fetch_q = queue.Queue(maxsize=8)
    upload_q = queue.Queue(maxsize=32)

    task_iter = client.tasks.list(project=LS["project_id"])

    fetcher_t = threading.Thread(target=_fetcher,
                                 args=(task_iter, fetch_q, args, counters),
                                 daemon=True)
    uploader_t = threading.Thread(target=_uploader,
                                  args=(upload_q, client),
                                  daemon=True)
    fetcher_t.start()
    uploader_t.start()

    # Main thread: GPU inference
    n = 0
    while True:
        item = fetch_q.get()
        if item is _STOP:
            break
        task_id, img, fname = item
        orig_h, orig_w = img.shape[:2]
        dets = predict_image(model, img)
        results, avg = to_ls_results(dets, orig_w, orig_h)
        print(f"[gpu] task {task_id} | {fname} | {len(dets)} dets")
        upload_q.put((task_id, results, avg))
        n += 1
        if args.limit and n >= args.limit:
            break

    upload_q.put(_STOP)
    uploader_t.join()
    fetcher_t.join(timeout=2)

    print(f"[done] pushed={n} "
          f"skip_annotated={counters['skip_annotated']} "
          f"skip_predicted={counters['skip_predicted']} "
          f"skip_fetch_fail={counters['skip_fetch_fail']} "
          f"skip_unreadable={counters['skip_unreadable']} "
          f"model_version={LS['model_version']}")


if __name__ == "__main__":
    main()
