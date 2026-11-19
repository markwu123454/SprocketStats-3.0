"""Labelled top view of a tracked match, from the JSON mtmct/run/track_robots.py saved:
each robot's dot and team ("?" if unknown, "*" if inferred rather than read), with the
bumper text boxes drawn on the frames where they were read.

Run from the repo root:
    .venv\\Scripts\\python -m mtmct.debug.render_tracks <match_key> [...]
Writes <match_key>.mp4 and a few stills next to the saved JSON (mtmct/data/tracks/).
"""

import itertools
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from better_sampling.match_data import sample_frames
from broadcast.run.layouts import layout_for
from mtmct.run import track_robots
from mtmct.run.tracker import FPS as SPOT_FPS
from mtmct.run.tracker import DUP_DIST, frames_index

OUT_DIR = track_robots.OUT_DIR

ALLIANCE_BGR = {"red": (40, 40, 220), "blue": (220, 120, 30)}
LABEL_RISE = 70           # px between a robot's dot and its label, so the label clears the bumper


def draw_tag(img, x, y, text, color, highlight, no_go, placed=None):
    """Label above the robot (below it if that hits an overlay or another label), joined to
    its dot by a line. `placed` collects the rectangles of labels already drawn this frame."""
    placed = placed if placed is not None else []
    h, w = img.shape[:2]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    bw, bh = tw + 12, th + 12
    bx = int(min(max(x - bw // 2, 0), w - bw))

    def clear(top):
        return 0 <= top and top + bh <= h and not any(
            bx < x1 and x0 < bx + bw and top < y1 and y0 < top + bh
            for x0, y0, x1, y1 in [*no_go, *placed])

    # Above first, then below, stepping further out when a neighbour's label is in the way.
    spots = [s for k in (0, 40, 80)
             for s in ((y - LABEL_RISE - bh - k, "above"), (y + LABEL_RISE + k, "below"))]
    for spot, side in spots:
        if clear(spot):
            top, anchor_y = spot, (spot + bh if side == "above" else spot)
            break
    else:  # robot sits under an overlay: hang the label just below that overlay
        under = [y1 for x0, y0, x1, y1 in no_go if bx < x1 and x0 < bx + bw]
        top = min(h - bh, max(under, default=y) + 6)
        anchor_y = top
    anchor_x = int(min(max(x, bx + 4), bx + bw - 4))
    placed.append((bx, top, bx + bw, top + bh))

    cv2.line(img, (x, y), (anchor_x, anchor_y), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), 7, color, -1)
    cv2.circle(img, (x, y), 7, (255, 255, 255), 2)
    cv2.rectangle(img, (bx, top), (bx + bw, top + bh), color, -1)
    cv2.rectangle(img, (bx, top), (bx + bw, top + bh),
                  (0, 255, 255) if highlight else (255, 255, 255), 2)
    cv2.putText(img, text, (bx + 6, top + th + 5), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def render(video, match_key, per_frame, tracks, labels, reads, alliance_of, layout, offset=0, limit=None):
    """`offset`/`limit`: frame 0 of this run is video frame `offset`; stop after `limit` frames."""
    read_at = {(r["frame"], r["track"]): r for r in reads}
    stills = {int(s * SPOT_FPS) for s in (20, 50, 62, 90, 120)}
    out_mp4 = OUT_DIR / f"{match_key}.mp4"
    writer = None
    stop = offset + limit if limit else None
    for i, (t, frame) in enumerate(itertools.islice(sample_frames(video, SPOT_FPS), offset, stop)):
        h, w = frame.shape[:2]
        v0, v1 = layout.view_px(h)
        img = frame[v0:v1].copy()
        top_h = v1 - v0
        no_go = [(x0, y0 - v0, x1, y1 - v0) for x0, y0, x1, y1 in layout.overlay_px(w, h)]
        drawn = []  # (team, x, y): one tag per robot even if two tracks sit on it
        placed = []  # label rectangles drawn so far in this frame
        entries = sorted((e for e in per_frame.get(i, []) if e[0] in tracks),
                         key=lambda e: -len(tracks[e[0]].points))
        for tid, x, y in entries:
            y -= v0  # frame coordinates -> view coordinates
            team, how, _ = labels.get(tid, (None, None, None))
            if any(math.hypot(x - dx, y - dy) < DUP_DIST and (team is None or dt == team)
                   for dt, dx, dy in drawn):
                continue
            drawn.append((team, x, y))
            r = read_at.get((i, tid))
            if r is not None:
                quad = np.asarray(r["quad"]) - (0, v0)
                cv2.polylines(img, [quad.astype(np.int32)], True, (0, 255, 255), 2)
            color = ALLIANCE_BGR[alliance_of[team]] if team else (150, 150, 150)
            text = (team + ("*" if how != "read" else "")) if team else "?"
            draw_tag(img, int(x), int(y), text, color, r is not None, no_go, placed)
        cv2.putText(img, f"{match_key}   video {t:5.1f}s   yellow = bumper read now   * = inferred, not read",
                    (16, top_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if i in stills:
            cv2.imwrite(str(OUT_DIR / f"{match_key}_t{i / SPOT_FPS:05.1f}.jpg"), img)
        small = cv2.resize(img, (1280, round(1280 * top_h / w)), interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), SPOT_FPS,
                                     (small.shape[1], small.shape[0]))
        writer.write(small)
    writer.release()
    return out_mp4, i + 1


def render_saved(match_key):
    """Render the labelled video for a saved run (all of its frames, partial runs too)."""
    d = json.loads((OUT_DIR / f"{match_key}.json").read_text())
    tracks = track_robots.tracks_from_saved(d)
    noise = set(d["noise"])
    shown = {tid: t for tid, t in tracks.items() if tid not in noise}
    labels = {int(tid): tuple(lab) for tid, lab in d["labels"].items()}
    alliance_of = {t: side for side, ts in d["teams"].items() for t in ts}
    out_mp4, n = render(Path(d["video"]), match_key, frames_index(shown), shown, labels, d["reads"],
                        alliance_of, layout_for(match_key), d.get("offset", 0), d["stats"].get("n_frames"))
    print(f"{match_key}: {n} frames -> {out_mp4}")


def main() -> None:
    for key in sys.argv[1:]:
        render_saved(key)


if __name__ == "__main__":
    main()
