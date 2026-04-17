"""
sample_frames.py
----------------
Sample every Nth frame from an FRC match video for labeling.

Input can be:
  - a path to a local video file
  - a path to a folder containing videos
  - a YouTube URL (youtube.com/watch?v=..., youtu.be/..., youtube.com/shorts/...)

Time range:
  --start and --end accept flexible formats:
    "90"        → 90 seconds
    "1:30"      → 1 minute 30 seconds
    "1:30:45"   → 1 hour 30 minutes 45 seconds

Defaults to every 12 frames (at 30fps that's 2.5 frames/sec; at 60fps that's 5 fps).

Usage examples:
    python sample_frames.py match.mp4
    python sample_frames.py videos_folder/
    python sample_frames.py "https://youtu.be/dQw4w9WgXcQ"
    python sample_frames.py "https://youtu.be/..." --start 2:30 --end 5:00 --step 12
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
YT_MARKERS = ("youtube.com/", "youtu.be/", "www.youtube.com/")


# ---------------------------------------------------------------------------
# Input classification & helpers
# ---------------------------------------------------------------------------

def is_youtube_url(s: str) -> bool:
    lower = s.lower()
    return lower.startswith(("http://", "https://")) and any(m in lower for m in YT_MARKERS)


def parse_time(value: str | None) -> float | None:
    """
    Parse H:MM:SS / MM:SS / SS (ints or floats) into seconds.
    Returns None if value is None.
    """
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
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
        raise argparse.ArgumentTypeError(
            f"Invalid time format: {value!r} (use SS, MM:SS, or HH:MM:SS)"
        )
    if seconds < 0:
        raise argparse.ArgumentTypeError(f"Time cannot be negative: {value!r}")
    return seconds


def fmt_seconds(sec: float) -> str:
    """Format seconds as H:MM:SS.mmm — used for ffmpeg / logging."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# YouTube download (yt-dlp)
# ---------------------------------------------------------------------------

def download_youtube(
    url: str,
    dest_dir: Path,
    start_sec: float | None,
    end_sec: float | None,
) -> Path:
    """
    Download a YouTube video via yt-dlp into dest_dir and return the file path.

    If both start_sec and end_sec are provided, uses yt-dlp's --download-sections
    to only fetch the slice (requires ffmpeg on PATH; yt-dlp will re-mux the cut).
    """
    if shutil.which("yt-dlp") is None:
        sys.exit(
            "yt-dlp not found on PATH. Install it with:\n"
            "    pip install yt-dlp\n"
            "(ffmpeg is also required for format merging — most systems have it.)"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Prefer mp4 since OpenCV reads it reliably across platforms.
    # bv*+ba/best means: best video + best audio, falling back to best combined.
    cmd = [
        "yt-dlp",
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "--merge-output-format", "mp4",
        "-o", str(dest_dir / "%(id)s.%(ext)s"),
        "--no-playlist",
        "--no-warnings",
        "--print", "after_move:filepath",  # yt-dlp will print the final path
    ]

    if start_sec is not None and end_sec is not None:
        # Download only the requested slice. Forces a keyframe-aware cut.
        section = f"*{fmt_seconds(start_sec)}-{fmt_seconds(end_sec)}"
        cmd += ["--download-sections", section, "--force-keyframes-at-cuts"]
        print(f"  Downloading YouTube slice {section} from {url}")
    else:
        print(f"  Downloading full YouTube video from {url}")

    cmd.append(url)

    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("yt-dlp failed:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

    # The last non-empty line of stdout is the final filepath (due to --print after_move:filepath)
    printed = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    if not printed:
        # Fall back: pick the newest video file in dest_dir
        candidates = [p for p in dest_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS]
        if not candidates:
            sys.exit("yt-dlp finished but no video file was found.")
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return Path(printed[-1])


# ---------------------------------------------------------------------------
# Core sampling
# ---------------------------------------------------------------------------

def sample_video(
    video_path: Path,
    out_dir: Path,
    step: int,
    quality: int,
    start_sec: float | None,
    end_sec: float | None,
    already_trimmed: bool = False,
) -> int:
    """
    Sample every `step`-th frame of a video. Returns number of frames saved.

    If already_trimmed is True, the video file itself is the slice we want
    (e.g. yt-dlp already cut it) — in that case start/end are ignored for
    seeking. We still apply them for logging clarity.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! Could not open {video_path.name}", file=sys.stderr)
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps if fps > 0 else 0.0

    # Compute which frame indices bound the sampling window
    if already_trimmed:
        start_frame = 0
        end_frame = total_frames  # exclusive
        window_desc = f"full trimmed clip ({duration:.1f}s)"
    else:
        s_sec = start_sec if start_sec is not None else 0.0
        e_sec = end_sec if end_sec is not None else duration
        if e_sec <= s_sec:
            print(f"  ! End time ({e_sec:.1f}s) must be > start ({s_sec:.1f}s). "
                  f"Skipping {video_path.name}.", file=sys.stderr)
            cap.release()
            return 0
        start_frame = int(round(s_sec * fps))
        end_frame = min(total_frames, int(round(e_sec * fps)))
        window_desc = f"{fmt_seconds(s_sec)}–{fmt_seconds(e_sec)}"

    expected = max(0, (end_frame - start_frame + step - 1) // step)
    print(f"  {video_path.name}: {total_frames} frames @ {fps:.1f} fps, "
          f"range {window_desc} → ~{expected} frames")

    out_dir.mkdir(parents=True, exist_ok=True)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    stem = video_path.stem

    # Seek to start. On some containers CAP_PROP_POS_FRAMES snaps to the nearest
    # keyframe, so we record the actual position we ended up at.
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    frame_idx = start_frame
    saved = 0

    while frame_idx < end_frame:
        # grab() is cheap — we only decode frames we actually keep
        if not cap.grab():
            break

        # Sample aligned to absolute frame index so step is consistent across runs
        if (frame_idx - start_frame) % step == 0:
            ret, frame = cap.retrieve()
            if ret:
                out_path = out_dir / f"{stem}_f{frame_idx:07d}.jpg"
                cv2.imwrite(str(out_path), frame, encode_params)
                saved += 1

        frame_idx += 1

    cap.release()
    print(f"  → saved {saved} frames to {out_dir}")
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample every Nth frame from FRC video(s) or a YouTube URL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", type=str,
                    help="Video file path, folder of videos, OR YouTube URL.")
    ap.add_argument("--out", type=Path, default=Path("sampled_frames"),
                    help="Output directory for frames (default: sampled_frames)")
    ap.add_argument("--step", type=int, default=12,
                    help="Save one frame every N frames (default: 12)")
    ap.add_argument("--quality", type=int, default=92,
                    help="JPEG quality 1-100 (default: 92)")
    ap.add_argument("--start", type=str, default=None,
                    help="Start time (SS, MM:SS, or HH:MM:SS). Default: 0.")
    ap.add_argument("--end", type=str, default=None,
                    help="End time (SS, MM:SS, or HH:MM:SS). Default: end of video.")
    ap.add_argument("--keep-download", action="store_true",
                    help="Keep the downloaded YouTube video instead of deleting it.")
    args = ap.parse_args()

    if args.step < 1:
        ap.error("--step must be >= 1")

    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)
    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
        ap.error(f"--end ({end_sec}s) must be greater than --start ({start_sec}s)")

    videos: list[tuple[Path, bool]] = []  # (path, already_trimmed)
    tmp_dir: Path | None = None

    # ---- Classify input ----
    if is_youtube_url(args.input):
        tmp_dir = Path(tempfile.mkdtemp(prefix="frc_yt_"))
        video_path = download_youtube(args.input, tmp_dir, start_sec, end_sec)
        already_trimmed = start_sec is not None and end_sec is not None
        videos.append((video_path, already_trimmed))
    else:
        input_path = Path(args.input)
        if input_path.is_dir():
            found = sorted(p for p in input_path.iterdir()
                           if p.suffix.lower() in VIDEO_EXTS)
            if not found:
                sys.exit(f"No videos found in {input_path}")
            videos.extend((p, False) for p in found)
        elif input_path.is_file():
            videos.append((input_path, False))
        else:
            sys.exit(f"Input does not exist: {input_path}")

    # ---- Sample ----
    print(f"Sampling every {args.step} frames from {len(videos)} video(s)...")
    total_saved = 0
    try:
        for video_path, already_trimmed in videos:
            total_saved += sample_video(
                video_path, args.out, args.step, args.quality,
                start_sec, end_sec, already_trimmed=already_trimmed,
            )
    finally:
        # Clean up the downloaded YouTube file unless the user asked to keep it
        if tmp_dir is not None and not args.keep_download:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif tmp_dir is not None:
            print(f"\nDownloaded video kept at: {tmp_dir}")

    print(f"\nDone. {total_saved} frames total in {args.out.resolve()}")


if __name__ == "__main__":
    main()