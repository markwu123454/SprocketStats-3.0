"""Import public R2 images into a Label Studio project.

Images live at https://assets.markwu.org/<key> so no presigning needed.
Uses the R2 S3-compatible API only to list objects.

Usage:
  python upload_r2_to_ls.py                      # round_1 -> project 272005 (default)
  python upload_r2_to_ls.py --round 2 --project 279385
"""
import os, re, argparse
import boto3
from dotenv import load_dotenv
from ls_ext import LabelStudioClient

load_dotenv()

PUBLIC_BASE = "https://assets.markwu.org"
CHUNK       = 250                   # LS import limit
LS_URL      = "https://app.humansignal.com"
BUCKET      = "assets"

ROUNDS = {
    1: {"prefix": "sprocketstats/training_round_1/", "project": 272005},
    2: {"prefix": "sprocketstats/training_round_2/", "project": 279385},
}

ap = argparse.ArgumentParser()
ap.add_argument("--round",   type=int, default=1, choices=[1, 2])
ap.add_argument("--project", type=int, default=None,
                help="override project ID")
args = ap.parse_args()

cfg        = ROUNDS[args.round]
PREFIX     = cfg["prefix"]
PROJECT_ID = args.project if args.project is not None else cfg["project"]

print(f"[config] round={args.round}  prefix={PREFIX}  project={PROJECT_ID}")

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT_URL"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
)

keys = []
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
    for obj in page.get("Contents", []):
        keys.append(obj["Key"])

print(f"Found {len(keys)} objects")
if not keys:
    raise SystemExit("No objects matched — check BUCKET name or PREFIX")

tasks = [{"data": {"image": f"{PUBLIC_BASE}/{k}"}} for k in keys]

ls = LabelStudioClient(base_url=LS_URL, api_key=os.environ["LS_API_KEY"])

for i in range(0, len(tasks), CHUNK):
    chunk = tasks[i : i + CHUNK]
    ls.projects.import_tasks(id=PROJECT_ID, request=chunk)
    print(f"  imported {min(i + CHUNK, len(tasks))}/{len(tasks)}")

print("Done.")
