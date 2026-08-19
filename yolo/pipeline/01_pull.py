#!/usr/bin/env python3
"""
Step 1 — Pull labels from Label Studio + sync images from R2.

Exports project 279385 (RectangleLabels), converts to YOLO format, syncs
images from R2, and writes a dataset.yaml ready for ultralytics training.

Output layout:
  data/
    exports/snapshot_<pid>.json   raw LS export
    images/train/<match>_frame_*.jpg
    images/val/<match>_frame_*.jpg
    labels/train/<match>_frame_*.txt
    labels/val/<match>_frame_*.txt
    dataset.yaml

YOLO label format per line:
  <class_id> <cx> <cy> <w> <h>   (all 0-1 normalised)

Env vars: LS_API_KEY, R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
"""
import os, sys, time, json, re, pathlib, shutil, random, urllib.parse, base64
import yaml
import boto3
from botocore.config import Config as BotoConfig
from label_studio_sdk import LabelStudio
from dotenv import load_dotenv

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

_env = PROJECT_ROOT / ".env"
if not _env.exists():
    _env = PROJECT_ROOT.parent / "hrnet" / ".env"
load_dotenv(_env)

CFG = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))

def _p(s): p = pathlib.Path(s); return p if p.is_absolute() else PROJECT_ROOT / p

LS_CFG   = CFG["label_studio"]
R2_CFG   = CFG["r2"]
CLASSES  = CFG["classes"]           # ['blue', 'red']
VAL_FRAC = CFG["train"]["val_frac"]
SEED     = CFG["train"]["seed"]

EXPORT_DIR   = _p(CFG["paths"]["export_dir"]); EXPORT_DIR.mkdir(parents=True, exist_ok=True)
IMG_RAW      = _p(R2_CFG["local_image_root"]); IMG_RAW.mkdir(parents=True, exist_ok=True)
DATASET_YAML = _p(CFG["paths"]["dataset_yaml"])

LABEL_TO_CLS = {LS_CFG["label_blue"]: 0, LS_CFG["label_red"]: 1}

# round_2 key pattern: sprocketstats/training_round_2/{year}/{match}_frame_{num}.jpg
ROUND2_RE = re.compile(r"training_round_2/\d{4}/([^/]+)_frame_\d+\.jpg$", re.I)


# ── helpers ────────────────────────────────────────────────────────────────────

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
        job = client.projects.exports.get(id=project_id, export_pk=export_pk)
        status = (job.status or "").strip().lower()
        elapsed = int(time.time() - t0)
        if status in DONE:
            print(f"[export] done after {elapsed}s")
            break
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
        q = urllib.parse.urlparse(raw).query
        vals = urllib.parse.parse_qs(q).get("fileuri")
        if vals:
            b = vals[0]; b += "=" * (-len(b) % 4)
            try: decoded = base64.b64decode(b).decode()
            except Exception: decoded = base64.urlsafe_b64decode(b).decode()
            if decoded.startswith("s3://"):
                _, _, key = decoded[5:].partition("/")
                return key.lstrip("/")
    if raw.startswith("s3://"):
        _, _, key = raw[5:].partition("/")
        return key.lstrip("/")
    if raw.startswith("https://assets.markwu.org/"):
        return raw[len("https://assets.markwu.org/"):]
    return urllib.parse.urlparse(raw).path.lstrip("/")


def match_from_key(key):
    """Extract match code from round_2 key, e.g. '2022alhu_qm20'."""
    m = ROUND2_RE.search(key)
    if m:
        return m.group(1)
    # fallback: last two path segments
    parts = key.split("/")
    if len(parts) >= 2:
        fname = parts[-1]
        if "_frame_" in fname:
            return fname.split("_frame_")[0]
    return "unknown"


def parse_boxes(task):
    """Return list of (cls_id, cx, cy, w, h) normalised, or a skip reason str."""
    anns = task.get("annotations", [])
    if not anns:
        return "no_annotation"
    ann = next((a for a in anns if not a.get("was_cancelled")), None)
    if ann is None:
        return "all_cancelled"
    if not ann.get("result"):
        return []   # annotator confirmed no boxes — valid hard negative

    boxes = []
    for r in ann["result"]:
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        labels = v.get("rectanglelabels", [])
        if not labels:
            continue
        cls = LABEL_TO_CLS.get(labels[0])
        if cls is None:
            continue
        # LS percentages -> YOLO normalised
        cx = (v["x"] + v["width"]  / 2) / 100
        cy = (v["y"] + v["height"] / 2) / 100
        w  = v["width"]  / 100
        h  = v["height"] / 100
        boxes.append((cls, cx, cy, w, h))
    return boxes   # empty list = confirmed hard negative (no valid boxes placed)


def ensure_image(s3, key):
    local = IMG_RAW / pathlib.Path(key).name
    if local.exists() and local.stat().st_size > 0:
        return local
    s3.download_file(R2_CFG["bucket"], key, str(local))
    return local


# ── main ───────────────────────────────────────────────────────────────────────

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
    print(f"[parse] {len(tasks)} tasks")

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=BotoConfig(signature_version="s3v4", retries={"max_attempts": 5}),
    )

    # collect labeled samples grouped by match (for split)
    by_match: dict[str, list[tuple]] = {}   # match -> [(local_path, boxes), ...]
    from collections import Counter
    skips = Counter()

    for task in tasks:
        parsed = parse_boxes(task)
        if isinstance(parsed, str):
            skips[parsed] += 1
            continue
        # parsed is a list (possibly empty = confirmed hard negative)
        key = r2_key_from_task(task)
        try:
            local = ensure_image(s3, key)
        except Exception as e:
            print(f"[warn] {key}: {e}")
            skips["fetch_failed"] += 1
            continue
        match = match_from_key(key)
        by_match.setdefault(match, []).append((local, parsed))

    total = sum(len(v) for v in by_match.values())
    print(f"[parse] kept={total} across {len(by_match)} matches | skips={dict(skips)}")

    # match-level train/val split
    rng = random.Random(SEED)
    matches = sorted(by_match)
    rng.shuffle(matches)
    n_val = max(1, int(len(matches) * VAL_FRAC))
    val_set  = set(matches[:n_val])
    train_set = set(matches[n_val:])
    print(f"[split] train={len(train_set)} matches  val={len(val_set)} matches")

    # write images + labels
    for split in ("train", "val"):
        (PROJECT_ROOT / "data" / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / "data" / "labels"  / split).mkdir(parents=True, exist_ok=True)

    written = 0
    for match, samples in by_match.items():
        split = "val" if match in val_set else "train"
        img_dir = PROJECT_ROOT / "data" / "images" / split
        lbl_dir = PROJECT_ROOT / "data" / "labels"  / split
        for local_path, boxes in samples:
            dst_img = img_dir / local_path.name
            shutil.copy2(local_path, dst_img)
            lbl_path = lbl_dir / (local_path.stem + ".txt")
            with open(lbl_path, "w") as f:
                for cls, cx, cy, w, h in boxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            written += 1

    print(f"[write] {written} image+label pairs")

    # dataset.yaml for ultralytics
    ds = {
        "path": str(PROJECT_ROOT / "data"),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    with open(DATASET_YAML, "w") as f:
        yaml.dump(ds, f, default_flow_style=False)
    print(f"[done] dataset.yaml -> {DATASET_YAML}")


if __name__ == "__main__":
    main()
