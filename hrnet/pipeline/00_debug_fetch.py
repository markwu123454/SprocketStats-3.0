#!/usr/bin/env python3
"""
Debug helper — grab a SMALL sample of labeled tasks fast, no snapshot.

Use this to build manifest_debug.jsonl for the overfit-100 sanity check while
the full export snapshot is still running (or before you ever run it).

    python scripts/00_debug_fetch.py --n 200

Writes exports/manifest_debug.jsonl in the SAME format as 01_pull.py, so
02_train.py can point straight at it.

Env: LS_API_KEY, R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
"""
import os, json, argparse, pathlib, urllib.parse, re
import yaml, boto3
from botocore.config import Config as BotoConfig
from label_studio_sdk import LabelStudio
from dotenv import load_dotenv

# Load credentials from a .env file next to config.yaml (LS_API_KEY, R2_*)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CFG = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
LS, R2, P = CFG["label_studio"], CFG["r2"], CFG["paths"]
LABEL_TO_CLS = {LS["label_blue"]: "blue", LS["label_red"]: "red"}
def _p(path_str):
    p = pathlib.Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)

IMG_ROOT = _p(R2["local_image_root"]); IMG_ROOT.mkdir(parents=True, exist_ok=True)
EXPORT_DIR = _p(P["export_dir"]); EXPORT_DIR.mkdir(parents=True, exist_ok=True)
# FRC event-code path segment, e.g. '2022alhu_qm20' -> season 2022, event 2022alhu
EVENT_RE = re.compile(r"(20\d\d)([a-z0-9]+?)_([a-z]+\d+)", re.I)


def parse_event(key):
    """Return (season:int|None, event:str|None, match:str|None) from the key."""
    for part in key.split("/"):
        m = EVENT_RE.match(part)
        if m:
            return int(m.group(1)), (m.group(1) + m.group(2)).lower(), part.lower()
    return None, None, None


import base64


def key_from_task(td, debug=False):
    """
    Resolve the true R2 (bucket, key) for a task.

    Primary form on this instance: data['image'] is an LS resolver URL
        /tasks/<id>/resolve/?fileuri=<base64 of s3://bucket/key>
    We decode the fileuri to recover the real s3 URI.

    Fallbacks: storage_filename, plain s3://, direct URL.

    Returns (bucket, key) or None.
    """
    raw = td.get("data", {}).get("image", "")
    if debug:
        print(f"[debug] raw image ref = {raw!r}")
        print(f"[debug] storage_filename = {td.get('storage_filename')!r}")

    # 1) LS resolver URL with base64 fileuri (primary case here)
    if "fileuri=" in raw:
        q = urllib.parse.urlparse(raw).query
        vals = urllib.parse.parse_qs(q).get("fileuri")
        if vals:
            # base64 may be urlsafe and/or missing padding
            b = vals[0]
            b += "=" * (-len(b) % 4)
            try:
                decoded = base64.b64decode(b).decode()
            except Exception:
                decoded = base64.urlsafe_b64decode(b).decode()
            if decoded.startswith("s3://"):
                bucket, _, key = decoded[5:].partition("/")
                return bucket, key.lstrip("/")

    # 2) storage_filename (bucket-relative key), if present
    sf = td.get("storage_filename")
    if sf:
        return R2["bucket"], sf.lstrip("/")

    # 3) plain s3://bucket/key
    if raw.startswith("s3://"):
        bucket, _, key = raw[5:].partition("/")
        return bucket, key.lstrip("/")

    # 4) bare resolve URL with no fileuri -> not resolvable
    if "/resolve/" in raw or "/tasks/" in raw:
        return None

    # 5) direct URL
    k = urllib.parse.urlparse(raw).path.lstrip("/")
    if k.startswith(R2["bucket"] + "/"):
        k = k[len(R2["bucket"]) + 1:]
    k = urllib.parse.unquote(k)
    return (R2["bucket"], k) if k else None


def parse_points(anns):
    ann = next((a for a in anns if not a.get("was_cancelled")), None)
    if ann is None: return None
    pts, W, H = [], None, None
    for r in ann["result"]:
        if r.get("type") != "keypointlabels": continue
        v = r["value"]; W = W or r.get("original_width"); H = H or r.get("original_height")
        labels = v.get("keypointlabels", [])
        if not labels: continue
        cls = LABEL_TO_CLS.get(labels[0])
        if cls is None: continue
        pts.append({"x_px": v["x"]/100.0*W, "y_px": v["y"]/100.0*H, "cls": cls})
    if not pts or not W or not H: return None
    return pts, W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--page-size", type=int, default=100)
    args = ap.parse_args()

    ls = LabelStudio(base_url=LS["base_url"], api_key=os.environ["LS_API_KEY"])
    s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"],
                      aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                      aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                      config=BotoConfig(signature_version="s3v4",
                                        retries={"max_attempts": 3}))

    out = EXPORT_DIR / "manifest_debug.jsonl"
    kept = 0
    seen = 0
    first = True
    with open(out, "w") as mf:
        # Fetch bounded pages explicitly rather than iterating the lazy pager
        # across the whole 112k-task project (that hangs / times out).
        page = 1
        while kept < args.n:
            resp = ls.tasks.list(
                project=LS["project_id"], fields="all",
                page=page, page_size=args.page_size,
            )
            batch = list(resp)          # one page only
            if not batch:
                print("[info] no more tasks"); break
            page += 1
            for t in batch:
                seen += 1
                td = t.dict() if hasattr(t, "dict") else t
                anns = td.get("annotations") or []
                if not anns:
                    continue
                parsed = parse_points(anns)
                if parsed is None:
                    continue
                pts, W, H = parsed
                resolved = key_from_task(td, debug=first)
                first = False
                if not resolved:
                    print(f"[skip] task {td.get('id')}: could not resolve R2 key")
                    continue
                bucket, key = resolved
                local = IMG_ROOT / key
                if not local.exists():
                    local.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        s3.download_file(bucket, key, str(local))
                    except Exception as e:
                        print(f"[skip] {bucket}/{key}: {e}"); continue
                season, event, match = parse_event(key)
                mf.write(json.dumps({
                    "image": str(local),
                    "season": season, "event": event, "match": match,
                    "points": pts, "width": W, "height": H,
                }) + "\n")
                kept += 1
                if kept >= args.n:
                    break
    print(f"[done] scanned {seen} tasks, wrote {kept} debug samples -> {out}")


if __name__ == "__main__":
    main()
