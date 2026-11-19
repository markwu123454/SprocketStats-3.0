"""Quick test: can PP-OCRv6 read FRC bumper numbers off 2026 match broadcasts?

For the first N 2026 qualification matches in better_sampling/matches.json that have
a video:
  1. Look up the 6 teams on TBA (/match/{key}/simple).
  2. Download the video (video-only, <=1080p, so no ffmpeg merge) into better_sampling/data/videos/.
  3. Sample 2 frames/s, OCR each whole frame, keep all-digit reads outside the
     broadcast score bar that are within one edit of one of the 6 team numbers.
  4. Save annotated frames to ocr/data/quick_test/<match_key>/ and print a summary.

Run from the repo root:
    .venv\\Scripts\\python -m ocr.debug.probe_bumper_reads
"""

import statistics
import sys
from collections import defaultdict
from pathlib import Path

# ocr_engine sets the ocr/.cache env vars before paddleocr is imported.
from ocr.run.ocr_engine import build_ocr
from better_sampling.match_data import (NUM_MATCHES, YEAR, download, match_teams,
                                        pick_matches, sample_frames, tba_session)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "quick_test"  # gitignored via ocr/.gitignore
SAMPLE_FPS = 2.0

# 2026 FIRST broadcast overlay: score bar (team numbers, score, timer) is a fixed
# band at the top centre. Fractions of frame width/height, so resolution-independent.
# Measured on 2026mabil_qm46; text boxes whose centre falls inside are ignored.
SCORE_BAR = (0.22, 0.0, 0.78, 0.16)  # x0, y0, x1, y1


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
