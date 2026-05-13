#!/usr/bin/env python3
"""
FRC Match Video Frame Sampler
- Downloads to a temp file (fixes Windows pipe issues with yt-dlp muxing)
- Samples at 1 FPS with random frame offset
- Skips matches with no valid videos
- Resumes from previous state automatically
- Saves frames as JPEG q85 at 720p
"""

import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
MATCHES_JSON   = "matches.json"
OUTPUT_DIR     = Path("frames")
PROGRESS_FILE  = Path("progress.txt")
COOKIES_FILE   = Path("www.youtube.com_cookies.txt")
FPS            = 1
JPEG_QUALITY   = 85
TARGET_HEIGHT  = 720
MAX_OFFSET_SEC = 5
# ──────────────────────────────────────────────────────────────────────────────


def check_dependencies():
    missing = [t for t in ("yt-dlp", "ffmpeg") if shutil.which(t) is None]
    if missing:
        for t in missing:
            print(f"Error: '{t}' not found in PATH. Install it and try again.")
        sys.exit(1)


def load_completed() -> set[str]:
    if not PROGRESS_FILE.exists():
        return set()
    return set(PROGRESS_FILE.read_text(encoding="utf-8").splitlines())


def mark_completed(match_key: str):
    with PROGRESS_FILE.open("a", encoding="utf-8") as f:
        f.write(match_key + "\n")


def pick_video(match: dict) -> str | None:
    valid = match.get("videos", [])
    return random.choice(valid) if valid else None


def cookie_args() -> list[str]:
    if COOKIES_FILE.exists():
        return ["--cookies", str(COOKIES_FILE)]
    return []


def sample_video(match_key: str, url: str, pbar: tqdm) -> bool:
    out_dir = OUTPUT_DIR / match_key
    out_dir.mkdir(parents=True, exist_ok=True)

    offset    = random.uniform(0, MAX_OFFSET_SEC)
    short_url = url.split("v=")[-1] if "v=" in url else url[-11:]
    pbar.set_postfix_str(f"{match_key}  [{short_url}]  downloading...", refresh=True)

    # Use a temp directory so cleanup is automatic even on failure
    with tempfile.TemporaryDirectory() as tmp:
        tmp_video = str(Path(tmp) / "video.mp4")

        # Step 1: download to temp file
        yt_cmd = [
            "yt-dlp",
            "--no-warnings",
            "--remote-components", "ejs:github",
            *cookie_args(),
            "-f", "bestvideo[height<=720]+bestaudio/bestvideo[height<=720]/best[height<=720]/best",
            "--merge-output-format", "mp4",
            "-o", tmp_video,
            url,
        ]

        try:
            result = subprocess.run(yt_cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                tqdm.write(f"  [error] yt-dlp ({match_key}): {result.stderr.decode()[:300]}")
                return False
        except subprocess.TimeoutExpired:
            tqdm.write(f"  [error] yt-dlp timed out: {match_key}")
            return False
        except Exception as e:
            tqdm.write(f"  [error] yt-dlp ({match_key}): {e}")
            return False

        # Check the file actually exists and has content
        video_path = Path(tmp_video)
        if not video_path.exists() or video_path.stat().st_size < 1024:
            # yt-dlp sometimes changes the extension; find whatever it wrote
            candidates = list(Path(tmp).glob("video.*"))
            if not candidates:
                tqdm.write(f"  [error] No video file found after download: {match_key}")
                return False
            video_path = candidates[0]

        pbar.set_postfix_str(f"{match_key}  [{short_url}]  offset={offset:.1f}s  extracting...", refresh=True)

        # Step 2: extract frames with ffmpeg
        vf = f"scale=-2:{TARGET_HEIGHT},fps={FPS}"
        ff_cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-ss", str(offset),
            "-i", str(video_path),
            "-vf", vf,
            "-q:v", str(int((100 - JPEG_QUALITY) / 100 * 31)),
            "-f", "image2",
            str(out_dir / "frame_%06d.jpg"),
        ]

        try:
            result = subprocess.run(ff_cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                tqdm.write(f"  [error] ffmpeg ({match_key}): {result.stderr.decode()[:300]}")
                return False
        except subprocess.TimeoutExpired:
            tqdm.write(f"  [error] ffmpeg timed out: {match_key}")
            return False
        except Exception as e:
            tqdm.write(f"  [error] ffmpeg ({match_key}): {e}")
            return False

    n = len(list(out_dir.glob("*.jpg")))
    #tqdm.write(f"  + {match_key}  ->  {n} frames")
    return True


def main():
    check_dependencies()

    if not COOKIES_FILE.exists():
        print(f"[warn] Cookie file '{COOKIES_FILE}' not found — proceeding without it")

    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(MATCHES_JSON, encoding="utf-8") as f:
        matches = json.load(f)

    completed = load_completed()
    pending   = [m for m in matches if m["match_key"] not in completed]
    total     = len(matches)

    print(f"Total matches : {total}")
    print(f"Already done  : {total - len(pending)}")
    print(f"To process    : {len(pending)}\n")

    skipped_no_video = 0
    processed        = 0
    failed           = 0

    with tqdm(
        total=len(pending),
        desc="Sampling matches",
        unit="match",
        dynamic_ncols=True,
        colour="cyan",
    ) as pbar:
        for match in pending:
            key   = match["match_key"]
            video = pick_video(match)

            if video is None:
                tqdm.write(f"  - {key}  (no valid video, skipping)")
                skipped_no_video += 1
                mark_completed(key)
                pbar.update(1)
                continue

            ok = sample_video(key, video, pbar)

            if ok:
                mark_completed(key)
                processed += 1
            else:
                failed += 1
                tqdm.write(f"  [!] {key} failed -- will retry on next run")

            pbar.update(1)
            time.sleep(random.uniform(1.5, 3.0))

    print(f"\n-- Done ----------------------------------------------")
    print(f"  Processed              : {processed}")
    print(f"  Skipped (no video)     : {skipped_no_video}")
    print(f"  Failed (retry on rerun): {failed}")


if __name__ == "__main__":
    main()