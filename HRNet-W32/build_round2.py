#!/usr/bin/env python3
"""
Build training_round_2 on R2.

Reads all objects already in sprocketstats/training_round_1/, groups by
match, randomly samples up to N frames per match, and copies them into
sprocketstats/training_round_2/{year}/{match}_frame_XXXXXX.jpg

Folder structure change:
  round_1: .../{match}/frame_XXXXXX.jpg         (event-level folder)
  round_2: .../{year}/{match}_frame_XXXXXX.jpg  (year-level folder)

Usage:
  python build_round2.py              # sample 5 frames per match (default)
  python build_round2.py -n 3         # sample 3 frames per match
  python build_round2.py --dry-run    # print plan without touching R2
"""
import os, re, random, argparse, pathlib
import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent / ".env")

BUCKET     = "assets"
SRC_PREFIX = "sprocketstats/training_round_1/"
DST_PREFIX = "sprocketstats/training_round_2/"

# match folder name, e.g. '2022alhu_qm20'
MATCH_RE = re.compile(r"^(20\d\d)[a-z0-9_]+$", re.I)


def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=BotoConfig(signature_version="s3v4", retries={"max_attempts": 5}),
    )


def list_round1(s3) -> dict[str, list[str]]:
    """Return {match_code: [full_r2_key, ...]} for every frame in round_1."""
    paginator = s3.get_paginator("list_objects_v2")
    by_match: dict[str, list[str]] = {}
    for page in paginator.paginate(Bucket=BUCKET, Prefix=SRC_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]                          # e.g. sprocketstats/training_round_1/2022alhu_qm20/frame_000003.jpg
            rel = key[len(SRC_PREFIX):]               # e.g. 2022alhu_qm20/frame_000003.jpg
            parts = rel.split("/")
            if len(parts) != 2:
                continue
            match_dir, fname = parts
            if not fname.lower().endswith(".jpg"):
                continue
            if not MATCH_RE.match(match_dir):
                continue
            by_match.setdefault(match_dir, []).append(key)
    return by_match


def dst_key(match: str, src_key: str) -> str:
    """Build the round_2 destination key."""
    year = match[:4]
    fname = src_key.rsplit("/", 1)[-1]               # e.g. frame_000003.jpg
    return f"{DST_PREFIX}{year}/{match}_{fname}"     # e.g. ...2022/2022alhu_qm20_frame_000003.jpg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5, metavar="N",
                    help="frames to sample per match (default 5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be copied without touching R2")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    s3 = make_s3()

    print(f"[list] scanning {SRC_PREFIX} ...")
    by_match = list_round1(s3)
    print(f"[list] found {len(by_match)} matches, "
          f"{sum(len(v) for v in by_match.values())} total frames")

    plan: list[tuple[str, str]] = []   # (src_key, dst_key)
    for match in sorted(by_match):
        frames = by_match[match]
        sample = rng.sample(frames, min(args.n, len(frames)))
        for src in sample:
            plan.append((src, dst_key(match, src)))

    print(f"[plan] {len(plan)} copies across {len(by_match)} matches "
          f"({args.n} per match)")

    if args.dry_run:
        for src, dst in plan[:20]:
            print(f"  COPY {src}\n    -> {dst}")
        if len(plan) > 20:
            print(f"  ... and {len(plan) - 20} more")
        return

    copied = errors = 0
    for i, (src, dst) in enumerate(plan, 1):
        try:
            s3.copy_object(
                Bucket=BUCKET,
                CopySource={"Bucket": BUCKET, "Key": src},
                Key=dst,
            )
            copied += 1
        except Exception as e:
            print(f"  [error] {src}: {e}")
            errors += 1
        if i % 50 == 0 or i == len(plan):
            print(f"  [{i}/{len(plan)}] copied={copied} errors={errors}")

    print(f"[done] copied={copied} errors={errors}")
    print(f"[dest] {DST_PREFIX}{{year}}/{{match}}_frame_XXXXXX.jpg")


if __name__ == "__main__":
    main()
