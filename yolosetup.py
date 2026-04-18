"""
yolo_setup.py
----------------
Unified pipeline for FRC YOLO dataset preparation.

Available Commands:
  1. sample   - Sample frames from local video or YouTube.
                Usage: python yolo_setup.py sample match.mp4 --step 12
  2. download - Download images from GCS and convert Label Studio JSON to YOLO txt.
                Usage: python yolo_setup.py download export.json --out-dir raw_data
  3. split    - Split images/labels into train and val folders.
                Usage: python yolo_setup.py split --source-images raw_data/images --source-labels raw_data/labels
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
from google.cloud import storage

# --- CONFIGURATION DEFAULTS ---
DEFAULT_CLASSES = ["Red Alliance Robot", "Blue Alliance Robot"]
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
YT_MARKERS = ("youtube.com/", "youtu.be/", "www.youtube.com/")


# ==============================================================================
# 1. FRAME SAMPLING UTILS
# ==============================================================================

def is_youtube_url(s: str) -> bool:
    lower = s.lower()
    return lower.startswith(("http://", "https://")) and any(m in lower for m in YT_MARKERS)


def parse_time(value: str | None) -> float | None:
    if value is None: return None
    value = value.strip()
    if not value: return None
    try:
        parts = [float(p) for p in value.split(":")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid time format: {value!r}")
    if len(parts) == 1:
        seconds = parts[0]
    elif len(parts) == 2:
        seconds = parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
    else:
        raise argparse.ArgumentTypeError(f"Invalid time format: {value!r}")
    if seconds < 0: raise argparse.ArgumentTypeError(f"Time cannot be negative: {value!r}")
    return seconds


def fmt_seconds(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:06.3f}"


def download_youtube(url: str, dest_dir: Path, start_sec: float | None, end_sec: float | None) -> Path:
    if shutil.which("yt-dlp") is None:
        sys.exit("yt-dlp not found on PATH. Install it with: pip install yt-dlp")
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp", "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "--merge-output-format", "mp4", "-o", str(dest_dir / "%(id)s.%(ext)s"),
        "--no-playlist", "--no-warnings", "--print", "after_move:filepath",
    ]
    if start_sec is not None and end_sec is not None:
        section = f"*{fmt_seconds(start_sec)}-{fmt_seconds(end_sec)}"
        cmd += ["--download-sections", section, "--force-keyframes-at-cuts"]
        print(f"  Downloading YouTube slice {section} from {url}")
    else:
        print(f"  Downloading full YouTube video from {url}")
    cmd.append(url)
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr, file=sys.stderr)
        sys.exit(1)
    printed = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    if not printed:
        candidates = [p for p in dest_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS]
        if not candidates: sys.exit("yt-dlp finished but no video file was found.")
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return Path(printed[-1])


def sample_video(video_path: Path, out_dir: Path, step: int, quality: int, start_sec: float | None, end_sec: float | None, already_trimmed: bool = False) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! Could not open {video_path.name}", file=sys.stderr)
        return 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps if fps > 0 else 0.0

    if already_trimmed:
        start_frame, end_frame = 0, total_frames
        window_desc = f"full trimmed clip ({duration:.1f}s)"
    else:
        s_sec = start_sec if start_sec is not None else 0.0
        e_sec = end_sec if end_sec is not None else duration
        if e_sec <= s_sec:
            print(f"  ! End time must be > start. Skipping.", file=sys.stderr)
            return 0
        start_frame = int(round(s_sec * fps))
        end_frame = min(total_frames, int(round(e_sec * fps)))
        window_desc = f"{fmt_seconds(s_sec)}–{fmt_seconds(e_sec)}"

    expected = max(0, (end_frame - start_frame + step - 1) // step)
    print(f"  {video_path.name}: {total_frames} frames @ {fps:.1f} fps, range {window_desc} → ~{expected} frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    stem, frame_idx, saved = video_path.stem, start_frame, 0

    if start_frame > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    while frame_idx < end_frame:
        if not cap.grab(): break
        if (frame_idx - start_frame) % step == 0:
            ret, frame = cap.retrieve()
            if ret:
                cv2.imwrite(str(out_dir / f"{stem}_f{frame_idx:07d}.jpg"), frame, encode_params)
                saved += 1
        frame_idx += 1
    cap.release()
    print(f"  → saved {saved} frames to {out_dir}")
    return saved


# ==============================================================================
# 2. COMMAND IMPLEMENTATIONS
# ==============================================================================

def run_sample(args):
    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)
    videos = []
    tmp_dir = None

    if is_youtube_url(args.input):
        tmp_dir = Path(tempfile.mkdtemp(prefix="frc_yt_"))
        video_path = download_youtube(args.input, tmp_dir, start_sec, end_sec)
        videos.append((video_path, start_sec is not None and end_sec is not None))
    else:
        input_path = Path(args.input)
        if input_path.is_dir():
            videos = [(p, False) for p in input_path.iterdir() if p.suffix.lower() in VIDEO_EXTS]
        elif input_path.is_file():
            videos.append((input_path, False))

    total_saved = 0
    try:
        for video_path, already_trimmed in videos:
            total_saved += sample_video(video_path, args.out, args.step, args.quality, start_sec, end_sec, already_trimmed)
    finally:
        if tmp_dir and not args.keep_download:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\nDone. {total_saved} frames total in {args.out.resolve()}")


def run_download_and_extract(args):
    client = storage.Client()
    os.makedirs(f"{args.out_dir}/images", exist_ok=True)
    os.makedirs(f"{args.out_dir}/labels", exist_ok=True)

    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Processing {len(data)} tasks...")

    for task in data:
        gcs_path = task['data']['image']
        path_parts = gcs_path.replace("gs://", "").split("/", 1)
        bucket_name, blob_name = path_parts[0], path_parts[1]
        filename = os.path.basename(blob_name)
        basename = os.path.splitext(filename)[0]

        print(f"Downloading {filename}...")
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(f"{args.out_dir}/images/{filename}")

        label_path = f"{args.out_dir}/labels/{basename}.txt"
        with open(label_path, 'w') as f_out:
            if not task.get('annotations'): continue
            results = task['annotations'][0].get('result', [])
            for res in results:
                if res['type'] != 'rectanglelabels': continue
                val = res['value']
                label_name = val['rectanglelabels'][0]
                if label_name not in DEFAULT_CLASSES:
                    print(f"Warning: Unknown label {label_name}")
                    continue
                class_id = DEFAULT_CLASSES.index(label_name)
                w = val['width'] / 100
                h = val['height'] / 100
                x_c = (val['x'] / 100) + (w / 2)
                y_c = (val['y'] / 100) + (h / 2)
                f_out.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
    print(f"\n✅ Success! Dataset downloaded to {args.out_dir}")


def run_split(args):
    os.makedirs(os.path.join(args.target_dir, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'train/labels'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'val/images'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'val/labels'), exist_ok=True)

    all_images = [f for f in os.listdir(args.source_images) if f.endswith('.jpg')]
    random.shuffle(all_images)
    split_index = int(len(all_images) * args.ratio)

    def move_files(files, folder_type):
        for filename in files:
            name_only = os.path.splitext(filename)[0]
            shutil.copy(os.path.join(args.source_images, filename),
                        os.path.join(args.target_dir, folder_type, 'images', filename))
            label_file = name_only + ".txt"
            label_path = os.path.join(args.source_labels, label_file)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(args.target_dir, folder_type, 'labels', label_file))

    print(f"Moving {len(all_images[:split_index])} files to TRAIN")
    move_files(all_images[:split_index], 'train')
    print(f"Moving {len(all_images[split_index:])} files to VAL")
    move_files(all_images[split_index:], 'val')
    print("✅ Split Complete!")


# ==============================================================================
# 3. CLI ROUTING
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FRC YOLO Dataset Setup Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Sample Subcommand ---
    parser_sample = subparsers.add_parser("sample", help="Sample frames from video")
    parser_sample.add_argument("input", help="Video file path, folder, OR YouTube URL.")
    parser_sample.add_argument("--out", type=Path, default=Path("sampled_frames"), help="Output directory")
    parser_sample.add_argument("--step", type=int, default=12, help="Save one frame every N frames")
    parser_sample.add_argument("--quality", type=int, default=92, help="JPEG quality 1-100")
    parser_sample.add_argument("--start", type=str, default=None, help="Start time (SS, MM:SS, or HH:MM:SS)")
    parser_sample.add_argument("--end", type=str, default=None, help="End time (SS, MM:SS, or HH:MM:SS)")
    parser_sample.add_argument("--keep-download", action="store_true", help="Keep downloaded YT video")

    # --- Download/Extract Subcommand ---
    parser_dl = subparsers.add_parser("download", help="Download images from GCS & extract YOLO labels")
    parser_dl.add_argument("json_file", type=str, help="Label Studio JSON export file")
    parser_dl.add_argument("--out-dir", type=str, default=".", help="Output directory for /images and /labels")

    # --- Split Subcommand ---
    parser_split = subparsers.add_parser("split", help="Split dataset into train and val")
    parser_split.add_argument("--source-images", type=str, default="images", help="Folder containing images")
    parser_split.add_argument("--source-labels", type=str, default="labels", help="Folder containing labels")
    parser_split.add_argument("--target-dir", type=str, default="yolo_data", help="Output YOLO dataset folder")
    parser_split.add_argument("--ratio", type=float, default=0.8, help="Train ratio (e.g. 0.8 for 80%)")

    args = parser.parse_args()

    if args.command == "sample":
        run_sample(args)
    elif args.command == "download":
        run_download_and_extract(args)
    elif args.command == "split":
        run_split(args)


if __name__ == "__main__":
    main()