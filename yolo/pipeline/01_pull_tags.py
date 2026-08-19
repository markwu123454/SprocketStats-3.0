#!/usr/bin/env python3
"""
Pull AprilTag labels from Label Studio project 279391 + images from R2.
All label names are merged into a single 'apriltag' class (class 0).
Writes data/tags_dataset.yaml ready for 02_train.py --config config_tags.yaml.

Usage:
  python 01_pull_tags.py
  python 01_pull_tags.py --reuse          # reuse last LS snapshot
  python 01_pull_tags.py --local-cache    # skip re-downloading snapshot
"""
import os, sys, time, json, re, pathlib, shutil, random, urllib.parse, base64
import yaml
import boto3
from botocore.config import Config as BotoConfig
from label_studio_sdk import LabelStudio
from dotenv import load_dotenv

_env = pathlib.Path(__file__).parent.parent / ".env"
if not _env.exists():
    _env = pathlib.Path(__file__).parent.parent.parent / "hrnet" / ".env"
load_dotenv(_env)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
CFG_PATH     = PROJECT_ROOT / "config_tags.yaml"
CFG          = yaml.safe_load(open(CFG_PATH))

def _p(s):
    p = pathlib.Path(s)
    return p if p.is_absolute() else PROJECT_ROOT / p

LS_CFG   = CFG["label_studio"]
R2_CFG   = CFG["r2"]
CLASSES  = CFG["classes"]
VAL_FRAC = CFG["train"]["val_frac"]
SEED     = CFG["train"]["seed"]

EXPORT_DIR   = _p(CFG["paths"]["export_dir"]); EXPORT_DIR.mkdir(parents=True, exist_ok=True)
IMG_RAW      = _p(R2_CFG["local_image_root"]); IMG_RAW.mkdir(parents=True, exist_ok=True)
DATASET_YAML = _p(CFG["paths"]["dataset_yaml"])


def export_snapshot(client, project_id, reuse=False, use_local_cache=False):
    out = EXPORT_DIR / f"snapshot_{project_id}.json"
    if use_local_cache and out.exists() and out.stat().st_size > 0:
        print(f"[export] using local cache {out}")
        return out

    DONE = {"completed", "complete", "finished"}
    FAIL = {"failed", "canceled", "cancelled", "error"}
    export_pk = None

    if reuse:
        try:
            completed = [e for e in client.projects.exports.list(id=project_id)
                         if (getattr(e, "status", "") or "").strip().lower() in DONE]
            if completed:
                export_pk = max(completed, key=lambda e: e.id).id
                print(f"[export] reusing snapshot {export_pk}")
        except Exception as ex:
            print(f"[export] could not list snapshots ({ex}); creating fresh")

    if export_pk is None:
        print("[export] creating fresh snapshot ...")
        export_pk = client.projects.exports.create(id=project_id).id

    t0 = time.time()
    while True:
        job    = client.projects.exports.get(id=project_id, export_pk=export_pk)
        status = (job.status or "").strip().lower()
        elapsed = int(time.time() - t0)
        if status in DONE:
            print(f"[export] done after {elapsed}s"); break
        if status in FAIL:
            sys.exit(f"[export] failed: {job.status}")
        if elapsed > 3600:
            sys.exit(f"[export] timed out at status={job.status!r}")
        print(f"[export] {job.status!r} {elapsed}s ...")
        time.sleep(5)

    with open(out, "wb") as f:
        for chunk in client.projects.exports.download(
            id=project_id, export_pk=export_pk,
            export_type="JSON", request_options={"chunk_size": 65536},
        ):
            f.write(chunk)
    return out


def r2_key_from_task(task):
    raw = task["data"]["image"]
    if "fileuri=" in raw:
        q    = urllib.parse.urlparse(raw).query
        vals = urllib.parse.parse_qs(q).get("fileuri")
        if vals:
            b = vals[0]; b += "=" * (-len(b) % 4)
            try:    decoded = base64.b64decode(b).decode()
            except: decoded = base64.urlsafe_b64decode(b).decode()
            if decoded.startswith("s3://"):
                _, _, key = decoded[5:].partition("/")
                return key.lstrip("/")
    if raw.startswith("s3://"):
        _, _, key = raw[5:].partition("/")
        return key.lstrip("/")
    if raw.startswith("https://assets.markwu.org/"):
        return raw[len("https://assets.markwu.org/"):]
    return urllib.parse.urlparse(raw).path.lstrip("/")


def stem_from_key(key):
    return pathlib.Path(key).stem


def parse_boxes(task, seen_labels):
    """
    Return list of (0, cx, cy, w, h) — all labels merged into class 0.
    Handles both rectanglelabels and polygonlabels (converts polygon to AABB).
    Returns a skip-reason string if the task should be skipped.
    """
    anns = task.get("annotations", [])
    if not anns:
        return "no_annotation"
    ann = next((a for a in anns if not a.get("was_cancelled")), None)
    if ann is None:
        return "all_cancelled"
    if not ann.get("result"):
        return []   # confirmed hard negative

    boxes = []
    for r in ann["result"]:
        v = r["value"]
        rtype = r.get("type", "")

        if rtype == "rectanglelabels":
            labels = v.get("rectanglelabels", [])
            if not labels: continue
            seen_labels.add(labels[0])
            cx = (v["x"] + v["width"]  / 2) / 100
            cy = (v["y"] + v["height"] / 2) / 100
            w  = v["width"]  / 100
            h  = v["height"] / 100

        elif rtype == "polygonlabels":
            labels = v.get("polygonlabels", [])
            if not labels: continue
            seen_labels.add(labels[0])
            pts = v.get("points", [])   # [[x%, y%], ...]
            if len(pts) < 3: continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            cx = (x0 + x1) / 2 / 100
            cy = (y0 + y1) / 2 / 100
            w  = (x1 - x0) / 100
            h  = (y1 - y0) / 100

        else:
            continue

        boxes.append((0, cx, cy, w, h))   # class 0 = apriltag
    return boxes


def ensure_image(s3, key):
    local = IMG_RAW / pathlib.Path(key).name
    if local.exists() and local.stat().st_size > 0:
        return local
    print(f"  [r2] downloading {key}")
    s3.download_file(R2_CFG["bucket"], key, str(local))
    return local


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse",       action="store_true")
    ap.add_argument("--local-cache", action="store_true")
    args = ap.parse_args()

    client = LabelStudio(base_url=LS_CFG["base_url"], api_key=os.environ["LS_API_KEY"])
    snap   = export_snapshot(client, LS_CFG["project_id"],
                             reuse=args.reuse, use_local_cache=args.local_cache)
    tasks  = json.load(open(snap))
    print(f"[parse] {len(tasks)} tasks in project {LS_CFG['project_id']}")

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=BotoConfig(signature_version="s3v4", retries={"max_attempts": 5}),
    )

    seen_labels = set()
    samples     = []   # (local_path, boxes, stem)
    from collections import Counter
    skips = Counter()

    for task in tasks:
        parsed = parse_boxes(task, seen_labels)
        if isinstance(parsed, str):
            skips[parsed] += 1
            continue
        key = r2_key_from_task(task)
        try:
            local = ensure_image(s3, key)
        except Exception as e:
            print(f"[warn] {key}: {e}")
            skips["fetch_failed"] += 1
            continue
        samples.append((local, parsed, stem_from_key(key)))

    print(f"[parse] kept={len(samples)} | skips={dict(skips)}")
    print(f"[labels] found in project: {sorted(seen_labels)}")
    print(f"[boxes] total boxes: {sum(len(s[1]) for s in samples)}")

    # simple random train/val split (only 3 images — split 2/1)
    rng = random.Random(SEED)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * VAL_FRAC))
    val_samples   = shuffled[:n_val]
    train_samples = shuffled[n_val:]
    print(f"[split] train={len(train_samples)}  val={len(val_samples)}")

    for split in ("train", "val"):
        (PROJECT_ROOT / "data" / "tags" / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / "data" / "tags" / "labels"  / split).mkdir(parents=True, exist_ok=True)

    written = 0
    for split, split_samples in [("train", train_samples), ("val", val_samples)]:
        img_dir = PROJECT_ROOT / "data" / "tags" / "images" / split
        lbl_dir = PROJECT_ROOT / "data" / "tags" / "labels"  / split
        for local_path, boxes, stem in split_samples:
            shutil.copy2(local_path, img_dir / local_path.name)
            lbl_path = lbl_dir / (stem + ".txt")
            with open(lbl_path, "w") as f:
                for cls, cx, cy, w, h in boxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            written += 1

    ds = {
        "path":  str(PROJECT_ROOT / "data" / "tags"),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    with open(DATASET_YAML, "w") as f:
        yaml.dump(ds, f, default_flow_style=False)
    print(f"[done] {written} frames written -> {DATASET_YAML}")


if __name__ == "__main__":
    main()
