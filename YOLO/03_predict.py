#!/usr/bin/env python3
"""
Step 3 — Run YOLO inference on LS tasks and upload predictions.

Uses the 2-class (blue/red) checkpoint so alliance is preserved in predictions
to help the labeler correct rather than draw from scratch.

Images are fetched from the public CDN (assets.markwu.org) — no R2 credentials
needed for inference. A fetcher thread downloads images while the GPU runs
inference, and an uploader thread pushes results back to Label Studio.

Prediction result format (RectangleLabels):
  x, y = top-left corner as % of image dimensions
  width, height = box size as % of image dimensions

Usage:
  python 03_predict.py                        # predict all tasks
  python 03_predict.py --only-unpredicted     # skip tasks that already have a prediction
  python 03_predict.py --only-unannotated     # skip tasks with a human label
  python 03_predict.py --limit 200            # stop after 200 predictions
  python 03_predict.py --threshold 0.2        # lower confidence threshold
"""
import os, argparse, pathlib, threading, queue, sys
import numpy as np
from tqdm import tqdm
import requests
import cv2
import yaml
from dotenv import load_dotenv
from label_studio_sdk import LabelStudio

_env = pathlib.Path(__file__).parent / ".env"
if not _env.exists():
    _env = pathlib.Path(__file__).parent.parent / "HRNet-W32" / ".env"
load_dotenv(_env)

CFG     = yaml.safe_load(open(pathlib.Path(__file__).parent / "config.yaml"))
ROOT    = pathlib.Path(__file__).parent
LS_CFG  = CFG["label_studio"]
CLASSES = CFG["classes"]   # ['blue', 'red']

CLS_TO_LABEL = {
    0: LS_CFG["label_blue"],   # "Blue robot center"
    1: LS_CFG["label_red"],    # "Red robot center"
}

MODEL_VERSION = "yolo26n-2cls-v2"
DEFAULT_CKPT  = ROOT / "data" / "runs" / "robot_detection" / "weights" / "best.pt"
_STOP = object()


# ── image fetch ───────────────────────────────────────────────────────────────

def fetch_image(url: str):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        arr = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None


def image_url(task) -> str:
    data = task.data if hasattr(task, "data") else task["data"]
    return data["image"] if isinstance(data, dict) else data.image


# ── deduplication ────────────────────────────────────────────────────────────

def dedup_boxes(boxes, corner_tol=20):
    """
    Remove near-duplicate boxes where every corner (x1,y1,x2,y2) of one box
    is within `corner_tol` pixels of another box. Keeps the higher-confidence
    box from each duplicate pair. Works across alliances.

    boxes: list of (cls_id, x1, y1, x2, y2, conf)
    """
    # process highest confidence first so the better detection survives
    sorted_boxes = sorted(boxes, key=lambda b: b[5], reverse=True)
    kept = []
    for candidate in sorted_boxes:
        _, cx1, cy1, cx2, cy2, _ = candidate
        duplicate = False
        for _, kx1, ky1, kx2, ky2, _ in kept:
            if (abs(cx1 - kx1) <= corner_tol and
                abs(cy1 - ky1) <= corner_tol and
                abs(cx2 - kx2) <= corner_tol and
                abs(cy2 - ky2) <= corner_tol):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


# ── LS result conversion ──────────────────────────────────────────────────────

def to_ls_results(boxes, orig_w, orig_h):
    """
    boxes: list of (cls_id, x1, y1, x2, y2, conf) in pixels
    Returns (ls_result_list, avg_confidence)
    """
    results = []
    confs   = []
    for cls_id, x1, y1, x2, y2, conf in boxes:
        label = CLS_TO_LABEL.get(int(cls_id))
        if label is None:
            continue
        x_pct = x1 / orig_w * 100
        y_pct = y1 / orig_h * 100
        w_pct = (x2 - x1) / orig_w * 100
        h_pct = (y2 - y1) / orig_h * 100
        results.append({
            "from_name": LS_CFG["from_name"],
            "to_name":   LS_CFG["to_name"],
            "type":      "rectanglelabels",
            "score":     float(conf),
            "value": {
                "x":       x_pct,
                "y":       y_pct,
                "width":   w_pct,
                "height":  h_pct,
                "rotation": 0,
                "rectanglelabels": [label],
            },
            "original_width":  orig_w,
            "original_height": orig_h,
        })
        confs.append(float(conf))
    avg = float(np.min(confs)) if confs else 0.0
    return results, avg


# ── threads ───────────────────────────────────────────────────────────────────

def _fetcher(task_iter, fetch_q, args, counters):
    for task in task_iter:
        if args.skip_annotated and getattr(task, "total_annotations", 0):
            counters["skip_annotated"] += 1
            continue

        url = image_url(task)
        img = fetch_image(url)
        if img is None:
            counters["skip_fetch_fail"] += 1
            print(f"[fetch] failed: {url}")
            continue

        fetch_q.put((task.id, img, url.rsplit("/", 1)[-1]))

        if args.limit and (counters["fetched"] + fetch_q.qsize()) >= args.limit:
            break

    fetch_q.put(_STOP)


def _uploader(upload_q, client):
    while True:
        item = upload_q.get()
        if item is _STOP:
            break
        task_id, results, avg = item
        try:
            client.predictions.create(
                task=task_id,
                model_version=MODEL_VERSION,
                score=avg,
                result=results,
            )
        except Exception as e:
            print(f"  [upload] task {task_id} FAILED: {e}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                    help="path to best.pt (default: robot_detection/weights/best.pt)")
    ap.add_argument("--threshold",       type=float, default=0.25)
    ap.add_argument("--skip-annotated", action="store_true",
                    help="skip tasks that already have a human annotation")
    ap.add_argument("--limit",           type=int, default=0,
                    help="stop after N predictions (0 = all)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[error] pip install ultralytics")

    ckpt = pathlib.Path(args.ckpt)
    if not ckpt.exists():
        sys.exit(f"[error] checkpoint not found: {ckpt}\n  Run 02_train.py first.")

    print(f"[init] loading {ckpt.name} ...")
    model = YOLO(str(ckpt))

    print("[init] connecting to Label Studio ...")
    client = LabelStudio(base_url=LS_CFG["base_url"], api_key=os.environ["LS_API_KEY"])
    total_tasks = client.projects.get(id=LS_CFG["project_id"]).task_number
    total = args.limit if args.limit else total_tasks
    print(f"[init] {total_tasks} tasks in project, running on {total}")

    counters = {
        "fetched": 0, "pushed": 0,
        "skip_annotated": 0,
        "skip_fetch_fail": 0,
    }

    fetch_q  = queue.Queue(maxsize=16)
    upload_q = queue.Queue(maxsize=32)

    task_iter = client.tasks.list(project=LS_CFG["project_id"])

    fetcher_t  = threading.Thread(target=_fetcher,
                                  args=(task_iter, fetch_q, args, counters),
                                  daemon=True)
    uploader_t = threading.Thread(target=_uploader,
                                  args=(upload_q, client),
                                  daemon=True)
    fetcher_t.start()
    uploader_t.start()

    n = 0
    with tqdm(total=total, unit="frame", dynamic_ncols=True) as bar:
        while True:
            item = fetch_q.get()
            if item is _STOP:
                break
            task_id, img, fname = item
            orig_h, orig_w = img.shape[:2]

            results_raw = model(img, conf=args.threshold, verbose=False)[0]
            boxes = []
            for box in results_raw.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append((cls_id, x1, y1, x2, y2, conf))

            before = len(boxes)
            boxes = dedup_boxes(boxes)
            deduped = before - len(boxes)

            ls_results, avg = to_ls_results(boxes, orig_w, orig_h)
            blue = sum(1 for b in boxes if b[0] == 0)
            red  = sum(1 for b in boxes if b[0] == 1)
            bar.set_postfix(blue=blue, red=red, score=f"{avg:.2f}",
                            dedup=deduped if deduped else None, refresh=False)
            bar.update(1)

            upload_q.put((task_id, ls_results, avg))
            n += 1
            if args.limit and n >= args.limit:
                break

    upload_q.put(_STOP)
    uploader_t.join()
    fetcher_t.join(timeout=2)

    print(f"\n[done] pushed={n} "
          f"skip_annotated={counters['skip_annotated']} "
          f"skip_fetch_fail={counters['skip_fetch_fail']}")


if __name__ == "__main__":
    main()
