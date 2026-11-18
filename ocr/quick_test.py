"""Quick test: can PP-OCRv6 read FRC bumper numbers off 2026 match broadcasts?

For the first N 2026 qualification matches in better_sampling/matches.json that have
a video:
  1. Look up the 6 teams on TBA (/match/{key}/simple).
  2. Download the video (video-only, <=1080p, so no ffmpeg merge) into ocr/data/videos/.
  3. Sample 2 frames/s, OCR each whole frame, keep all-digit reads outside the
     broadcast score bar that are within one edit of one of the 6 team numbers.
  4. Save annotated frames to ocr/data/quick_test/<match_key>/ and print a summary.

Run from the repo root:
    .venv\\Scripts\\python ocr\\quick_test.py
"""

import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# check_model sets the ocr/.cache env vars before paddleocr is imported.
from check_model import build_ocr

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
import yt_dlp  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MATCHES_JSON = REPO_ROOT / "better_sampling" / "matches.json"
DATA_DIR = Path(__file__).resolve().parent / "data"  # gitignored via ocr/.gitignore
VIDEO_DIR = DATA_DIR / "videos"
OUT_DIR = DATA_DIR / "quick_test"

YEAR = "2026"
NUM_MATCHES = 3  # ask before raising: each video is ~75 MB
SAMPLE_FPS = 2.0
TBA_BASE = "https://www.thebluealliance.com/api/v3"

# 2026 FIRST broadcast overlay: score bar (team numbers, score, timer) is a fixed
# band at the top centre. Fractions of frame width/height, so resolution-independent.
# Measured on 2026mabil_qm46; text boxes whose centre falls inside are ignored.
SCORE_BAR = (0.22, 0.0, 0.78, 0.16)  # x0, y0, x1, y1

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


# ── Matching ─────────────────────────────────────────────────────────────────

def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insert/delete/substitute, each cost 1)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def match_team(read: str, teams: list[str]) -> str | None:
    """Return the team number an all-digit OCR read refers to, or None.

    `read` is already all digits; `teams` is the 6 team numbers in the match.
    Rules: an exact match always wins; otherwise accept a read one edit away from
    exactly one team (two candidates = ambiguous -> None). Reads under 3 digits are
    rejected: "75" is one edit from team 175, but 2-digit noise (scores, timers,
    field markings) is far more common than a real 2-digit bumper read.
    """
    if len(read) < 3:
        return None
    if read in teams:
        return read
    close = [t for t in teams if edit_distance(read, t) <= 1]
    return close[0] if len(close) == 1 else None


# ── Geometry / drawing ───────────────────────────────────────────────────────

def text_height(poly: np.ndarray) -> float:
    """Height of a 4-point text box (tl, tr, br, bl), robust to slight rotation."""
    tl, tr, br, bl = np.asarray(poly, dtype=float)
    return (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2


def in_score_bar(poly: np.ndarray, w: int, h: int) -> bool:
    cx, cy = np.asarray(poly, dtype=float).mean(axis=0)
    x0, y0, x1, y1 = SCORE_BAR
    return x0 * w <= cx <= x1 * w and y0 * h <= cy <= y1 * h


def annotate(frame, reads, alliance_of):
    img = frame.copy()
    h, w = img.shape[:2]
    x0, y0, x1, y1 = SCORE_BAR
    cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (128, 128, 128), 1)
    for poly, text, score, team in reads:
        pts = np.asarray(poly, dtype=np.int32)
        if team is None:
            color, label = (0, 165, 255), f"{text}?"  # orange: digits, no team
        else:
            color = (0, 0, 255) if alliance_of[team] == "red" else (255, 128, 0)
            label = f"{team} ({text} {score:.2f})"
        cv2.polylines(img, [pts], True, color, 2)
        org = (int(pts[:, 0].min()), max(15, int(pts[:, 1].min()) - 6))
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


# ── Main ─────────────────────────────────────────────────────────────────────

def run_match(ocr, match_key: str, video: Path, alliances: dict[str, list[str]]) -> None:
    teams = alliances["red"] + alliances["blue"]
    alliance_of = {t: c for c, ts in alliances.items() for t in ts}
    out = OUT_DIR / match_key
    out.mkdir(parents=True, exist_ok=True)

    frames_seen = defaultdict(set)   # team -> frame indices it was read in
    heights = defaultdict(list)      # team -> text heights (px)
    unmatched = 0                    # all-digit reads matching no team
    n_frames = 0

    for t, frame in sample_frames(video, SAMPLE_FPS):
        n_frames += 1
        h, w = frame.shape[:2]
        res = ocr.predict(frame)[0]
        reads = []
        for poly, text, score in zip(res["rec_polys"], res["rec_texts"], res["rec_scores"]):
            text = text.strip()
            if not text.isdigit() or in_score_bar(poly, w, h):
                continue
            team = match_team(text, teams)
            reads.append((poly, text, score, team))
            if team is None:
                unmatched += 1
            else:
                frames_seen[team].add(n_frames)
                heights[team].append(text_height(poly))

        if any(r[3] is not None for r in reads):
            cv2.imwrite(str(out / f"t{t:07.2f}.jpg"), annotate(frame, reads, alliance_of))
        if n_frames % 50 == 0:
            print(f"  {match_key}: {n_frames} frames ({t:.0f}s)", flush=True)

    print(f"\n{match_key}  ({n_frames} frames @ {SAMPLE_FPS:g} fps)  -> {out}")
    print(f"  {'team':>6} {'alliance':>8} {'frames':>7} {'median h px':>12}")
    for team in teams:
        hs = heights[team]
        med = f"{statistics.median(hs):.1f}" if hs else "-"
        print(f"  {team:>6} {alliance_of[team]:>8} {len(frames_seen[team]):>7} {med:>12}")
    print(f"  digit reads matching no team: {unmatched}")


def main() -> None:
    session = tba_session()
    print("TBA key OK")

    picked = pick_matches()
    if len(picked) < NUM_MATCHES:
        sys.exit(f"Only found {len(picked)} {YEAR} qual matches with videos.")

    jobs = []
    for m in picked:
        alliances = match_teams(session, m["match_key"])
        print(f"{m['match_key']}: red {alliances['red']}  blue {alliances['blue']}")
        video = download(m["videos"][0])
        print(f"  video: {video.name} ({video.stat().st_size / 2**20:.1f} MiB)")
        jobs.append((m["match_key"], video, alliances))

    ocr = build_ocr()
    for match_key, video, alliances in jobs:
        run_match(ocr, match_key, video, alliances)


if __name__ == "__main__":
    main()
