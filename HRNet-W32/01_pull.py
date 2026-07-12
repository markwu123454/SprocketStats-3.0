#!/usr/bin/env python3
"""
Step 1 — Pull labels (Label Studio snapshot) + images (Cloudflare R2).

Two responsibilities:
  A) Export all annotations from the LS project as a JSON snapshot.
     Uses the async snapshot flow (create -> poll -> download) because the
     project has ~112k tasks and the synchronous /export endpoint times out
     at that scale.
  B) Parse the snapshot into a flat, model-ready manifest and sync the
     referenced images down from R2 to a local mirror.

Output:
  exports/snapshot_<pid>.json      raw LS export
  exports/manifest.jsonl           one line per labeled frame:
        {"image": "<local_path>", "season": 2023,
         "points": [{"x_px":.., "y_px":.., "cls":"blue"}, ...],
         "width": W, "height": H}

Env vars required:
  LS_API_KEY, R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
"""
import os, sys, time, json, re, pathlib, urllib.parse
import yaml
import boto3
from botocore.config import Config as BotoConfig
from label_studio_sdk import LabelStudio
from dotenv import load_dotenv

# Load credentials from a .env file next to config.yaml (LS_API_KEY, R2_*)
load_dotenv(pathlib.Path(__file__).parent / ".env")

CFG = yaml.safe_load(open(pathlib.Path(__file__).parent / "config.yaml"))

# Anchor relative config paths to the project folder so they resolve the same
# regardless of the current working directory (and don't land at the drive
# root on Windows when a leading-slash path is used).
PROJECT_ROOT = pathlib.Path(__file__).parent

def _p(path_str):
    p = pathlib.Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)

LS = CFG["label_studio"]
R2 = CFG["r2"]
EXPORT_DIR = _p(CFG["paths"]["export_dir"]); EXPORT_DIR.mkdir(parents=True, exist_ok=True)
IMG_ROOT = _p(R2["local_image_root"]); IMG_ROOT.mkdir(parents=True, exist_ok=True)

LABEL_TO_CLS = {LS["label_blue"]: "blue", LS["label_red"]: "red"}


# ----------------------------------------------------------------------
# A) Export snapshot
# ----------------------------------------------------------------------
def export_snapshot(client, project_id: int, reuse: bool = False,
                    use_local_cache: bool = False) -> pathlib.Path:
    """Create a FRESH export snapshot by default so newly-labeled tasks are
    always included. Set reuse=True to reuse the latest completed server-side
    snapshot (faster, but may miss recent labels). Set use_local_cache=True to
    reuse a previously-downloaded local JSON without touching the server."""
    out = EXPORT_DIR / f"snapshot_{project_id}.json"
    if use_local_cache and out.exists() and out.stat().st_size > 0:
        print(f"[export] using local cached file {out} (may be stale)")
        return out

    DONE = {"completed", "complete", "finished"}
    FAIL = {"failed", "canceled", "cancelled", "error"}

    export_pk = None
    if reuse:
        # Opt-in: reuse the most recent completed snapshot on the server.
        try:
            completed = [e for e in client.projects.exports.list(id=project_id)
                         if (getattr(e, "status", "") or "").strip().lower() in DONE]
            if completed:
                # pick the newest by id (ids increase over time)
                latest = max(completed, key=lambda e: e.id)
                export_pk = latest.id
                print(f"[export] reusing latest completed snapshot {export_pk} "
                      f"(--reuse set; may miss very recent labels)")
        except Exception as ex:
            print(f"[export] could not list snapshots ({ex}); creating fresh")

    if export_pk is None:
        print("[export] creating FRESH snapshot ...")
        snap = client.projects.exports.create(id=project_id)
        export_pk = snap.id

    # Poll until the background worker finishes (Enterprise async export).
    # Status values vary by LS version; normalize and match generously so a
    # different completion string can't spin the loop forever.
    TIMEOUT_S = 60 * 60            # hard cap: 1 hour
    t0 = time.time()
    while True:
        job = client.projects.exports.get(id=project_id, export_pk=export_pk)
        status = (job.status or "").strip().lower()
        elapsed = int(time.time() - t0)
        if status in DONE:
            print(f"[export] completed after {elapsed}s")
            break
        if status in FAIL:
            sys.exit(f"[export] snapshot {export_pk} ended: {job.status}")
        if elapsed > TIMEOUT_S:
            sys.exit(f"[export] timed out after {elapsed}s at status={job.status!r}; "
                     f"check ls.projects.exports.list(id={project_id})")
        print(f"[export] status={job.status!r} elapsed={elapsed}s ...")
        time.sleep(5)

    print(f"[export] downloading snapshot {export_pk} -> {out}")
    with open(out, "wb") as f:
        for chunk in client.projects.exports.download(
            id=project_id, export_pk=export_pk,
            export_type="JSON", request_options={"chunk_size": 1024 * 64},
        ):
            f.write(chunk)
    return out


# ----------------------------------------------------------------------
# helpers to parse LS tasks
# ----------------------------------------------------------------------
import base64


def r2_bucket_key_from_task(task: dict):
    """
    Resolve (bucket, key) from a snapshot task.

    Primary form: data['image'] is an LS resolver URL
        /tasks/<id>/resolve/?fileuri=<base64 of s3://bucket/key>
    Fallbacks: plain s3://, direct URL.
    """
    raw = task["data"]["image"]

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
                bucket, _, key = decoded[5:].partition("/")
                return bucket, key.lstrip("/")

    if raw.startswith("s3://"):
        bucket, _, key = raw[5:].partition("/")
        return bucket, key.lstrip("/")

    path = urllib.parse.urlparse(raw).path.lstrip("/")
    if path.startswith(R2["bucket"] + "/"):
        path = path[len(R2["bucket"]) + 1:]
    return R2["bucket"], urllib.parse.unquote(path)


# FRC event-code path segment, e.g. '2022alhu_qm20' -> season 2022, event 2022alhu
EVENT_RE = re.compile(r"(20\d\d)([a-z0-9]+?)_([a-z]+\d+)", re.I)


def parse_event(key: str):
    """Return (season, event, match) from the R2 key path.
    e.g. '.../2022alhu_qm20/frame_x.jpg' -> (2022, '2022alhu', '2022alhu_qm20')."""
    for part in key.split("/"):
        m = EVENT_RE.match(part)
        if m:
            return int(m.group(1)), (m.group(1) + m.group(2)).lower(), part.lower()
    return None, None, None


def parse_points(task: dict):
    """Extract keypoint annotations. KeyPointLabels x/y are PERCENTAGES.
    Returns (pts, W, H) on success, or a string reason on skip."""
    anns = task.get("annotations", [])
    if not anns:
        return "no_annotation"          # task not labeled yet
    ann = next((a for a in anns if not a.get("was_cancelled")), None)
    if ann is None:
        return "all_cancelled"
    if not ann.get("result"):
        return "empty_result"           # annotation opened but no point placed

    pts, W, H = [], None, None
    saw_unmapped = False
    for r in ann["result"]:
        if r.get("type") != "keypointlabels":
            continue
        v = r["value"]
        W = W or r.get("original_width")
        H = H or r.get("original_height")
        labels = v.get("keypointlabels", [])
        if not labels:
            continue
        cls = LABEL_TO_CLS.get(labels[0])
        if cls is None:
            saw_unmapped = True       # a dot WAS placed, but its label is unknown
            continue
        pts.append({
            "x_px": v["x"] / 100.0 * W,
            "y_px": v["y"] / 100.0 * H,
            "cls": cls,
        })
    if not pts:
        # No usable dots. Separate "labeler placed a dot we couldn't map" (a
        # LABEL_TO_CLS problem — NOT a real negative) from "genuinely no dots".
        return "unmapped_labels" if saw_unmapped else "no_keypoints"
    if not W or not H:
        return "no_dims"
    return pts, W, H


# ----------------------------------------------------------------------
# B) R2 sync
# ----------------------------------------------------------------------
def make_r2():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=BotoConfig(signature_version="s3v4", retries={"max_attempts": 5}),
    )


def ensure_image(s3, bucket: str, key: str) -> pathlib.Path:
    local = IMG_ROOT / key
    if local.exists() and local.stat().st_size > 0:
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(local))
    return local


# ----------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true",
                    help="reuse latest completed server snapshot (faster, may "
                         "miss very recent labels). Default: create fresh.")
    ap.add_argument("--local-cache", action="store_true",
                    help="reuse the previously-downloaded local JSON without "
                         "contacting the server (fastest, definitely stale).")
    args = ap.parse_args()

    client = LabelStudio(base_url=LS["base_url"], api_key=os.environ["LS_API_KEY"])
    snap_path = export_snapshot(client, LS["project_id"],
                                reuse=args.reuse, use_local_cache=args.local_cache)
    tasks = json.load(open(snap_path))
    print(f"[parse] {len(tasks)} tasks in snapshot")

    s3 = make_r2()
    manifest_path = EXPORT_DIR / "manifest.jsonl"
    kept = 0
    from collections import Counter
    skips = Counter()
    neg_reasons = Counter()      # kept HARD NEGATIVES, by reason
    # Frames a human labeled with no usable dots are intentional negatives: keep
    # them with points:[] so the model learns to output a blank heatmap (fewer
    # false positives). `unmapped_labels` is included per config, but it may
    # signal a LABEL_TO_CLS bug rather than a true empty, so we warn on it below.
    EMPTY_NEGATIVE = {"empty_result", "no_keypoints", "unmapped_labels"}
    with open(manifest_path, "w") as mf:
        for i, task in enumerate(tasks):
            parsed = parse_points(task)
            if isinstance(parsed, str):
                if parsed not in EMPTY_NEGATIVE:
                    skips[parsed] += 1
                    continue
                pts, W, H = [], None, None       # labeled negative, no dots
                neg_reasons[parsed] += 1
            else:
                pts, W, H = parsed
            bucket, key = r2_bucket_key_from_task(task)
            try:
                local = ensure_image(s3, bucket, key)
            except Exception as e:
                print(f"[warn] could not fetch {bucket}/{key}: {e}")
                skips["fetch_failed"] += 1
                continue
            season, event, match = parse_event(key)
            mf.write(json.dumps({
                "image": str(local),
                "season": season, "event": event, "match": match,
                "points": pts,
                "width": W, "height": H,
            }) + "\n")
            kept += 1
            if (i + 1) % 2000 == 0:
                print(f"[parse] {i+1}/{len(tasks)} kept={kept}")

    neg_kept = sum(neg_reasons.values())
    print(f"[done] kept={kept} ({neg_kept} labeled negatives) -> {manifest_path}")
    print(f"[negatives] kept by reason: {dict(neg_reasons)}")
    if neg_reasons["unmapped_labels"]:
        print(f"[warn] {neg_reasons['unmapped_labels']} frames had a keypoint whose "
              f"label didn't map to any class and were kept as NEGATIVES. If those "
              f"frames actually contain robots, fix LABEL_TO_CLS — otherwise you are "
              f"training the model that real robots are background.")
    print(f"[skips] unlabeled={skips['no_annotation']} "
          f"cancelled={skips['all_cancelled']} "
          f"other={skips['no_dims']+skips['fetch_failed']}")
    print(f"[skips] full breakdown: {dict(skips)}")


if __name__ == "__main__":
    main()
