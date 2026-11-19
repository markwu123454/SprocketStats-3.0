"""Match inputs: the six teams from TBA, the match video, and its frames.

Fetching TBA data and match video is what better_sampling/ is for. This began as
ocr's quick_test.py; tracking (mtmct/run/track_robots.py) and ocr's probes both use it.
"""

import json
import os
import re
import sys
from pathlib import Path

import cv2
import requests
import yt_dlp
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
MATCHES_JSON = Path(__file__).resolve().parent / "matches.json"
VIDEO_DIR = Path(__file__).resolve().parent / "data" / "videos"  # gitignored

YEAR = "2026"
NUM_MATCHES = 3  # ask before raising: each video is ~75 MB
TBA_BASE = "https://www.thebluealliance.com/api/v3"

DIGITS = re.compile(r"\d+")


# ── TBA ──────────────────────────────────────────────────────────────────────

def tba_session() -> requests.Session:
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("TBA_API_KEY", "").strip()
    if not key:
        sys.exit("TBA_API_KEY is missing or empty in the repo-root .env")
    s = requests.Session()
    s.headers["X-TBA-Auth-Key"] = key
    r = s.get(f"{TBA_BASE}/status", timeout=20)
    if r.status_code != 200:
        sys.exit(f"TBA key check failed: HTTP {r.status_code} {r.text[:200]}")
    return s


def match_teams(s: requests.Session, match_key: str) -> dict[str, list[str]]:
    r = s.get(f"{TBA_BASE}/match/{match_key}/simple", timeout=20)
    r.raise_for_status()
    alliances = r.json()["alliances"]
    # "frc1071" -> "1071"; drop letter suffixes like "frc254B" -> "254".
    return {
        color: [DIGITS.search(k).group() for k in alliances[color]["team_keys"]]
        for color in ("red", "blue")
    }


# ── Video ────────────────────────────────────────────────────────────────────

def pick_matches() -> list[dict]:
    matches = json.loads(MATCHES_JSON.read_text(encoding="utf-8"))
    picked = [
        m for m in matches
        if m["match_key"].startswith(YEAR) and "_qm" in m["match_key"] and m["videos"]
    ]
    return picked[:NUM_MATCHES]


def download(url: str) -> Path:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    opts = {
        # Video-only stream so yt-dlp never needs ffmpeg to merge audio.
        "format": "bv*[height<=1080][ext=mp4]/bv*[height<=1080]",
        "outtmpl": str(VIDEO_DIR / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        path = Path(ydl.prepare_filename(info))
        if not path.exists():
            ydl.download([url])
    return path


def sample_frames(path: Path, fps: float):
    """Yield (t_seconds, BGR frame) at roughly `fps` frames per second."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / fps))
    i = 0
    while True:
        if i % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            yield i / src_fps, frame
        elif not cap.grab():  # grab() skips decoding-to-BGR: much faster
            break
        i += 1
    cap.release()
