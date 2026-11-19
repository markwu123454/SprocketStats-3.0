"""Bumper reads: for each tracked robot dot, find the bumper text in a crop around it and
score the six team numbers against it (team_scorer.py does the scoring).

mtmct/run/track_robots.py calls read_bumpers every OCR_EVERY tracked frames. Each accepted
read is attached to the track whose dot it was found on, and is evidence for both cutting
tracks (mtmct/run/split_tracks.py, via is_confident) and naming them (assign_teams.py).
"""

import math

import cv2
import numpy as np

OCR_EVERY = 5             # read bumpers on every 5th spotted frame -> 2 fps
# Where the overlays are on screen differs per event: see broadcast/run/layouts.py.

CROP_HALF = 110           # px around a robot dot; overhead robots are ~100 px across
CROP_SCALE = 3            # enlarge so 8-12 px bumper digits become 24-36 px
# Crops with no accepted read are re-detected rotated by these. 90 degrees catches sideways
# side-bumper text; +-30 was dropped after it found 3 of 618 reads on qm46 for ~2/3 of run time.
RETRY_ANGLES = (90,)
ATTACH_DIST = 85          # a text box belongs to the nearest robot dot within this many px
MIN_LOGP_PER_CHAR = -1.0  # weaker reads match no team (measured: real >= -0.91, junk <= -1.53)
CONFIDENT_PER_CHAR = -0.2  # reads at least this sure may split a track


def is_confident(read) -> bool:
    """Is this read sure enough to cut a track where it disagrees with earlier reads?"""
    return read["per_char"] > CONFIDENT_PER_CHAR


def mask_overlays(frame, layout):
    """Grey out the layout's overlays (score bar etc.): they print team numbers and digits."""
    out = frame.copy()
    h, w = out.shape[:2]
    for x0, y0, x1, y1 in layout.overlay_px(w, h):
        out[y0:y1, x0:x1] = 127
    return out


def _rotate(img, angle):
    """Rotate about the centre without cutting off corners. Returns (image, 2x3 matrix)."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=(127, 127, 127)), m


def _find_texts(scorer, jobs, dots, angle):
    """Text boxes in these robot crops (rotated by `angle`), kept only on their own robot."""
    from ocr.run.team_scorer import warp_quad

    imgs, inverses = [], []
    for _, crop, _ in jobs:
        if angle:
            img, m = _rotate(crop, angle)
            imgs.append(img)
            inverses.append(cv2.invertAffineTransform(m))
        else:
            imgs.append(crop)
            inverses.append(None)
    found = []  # (track id, straightened text image, quad in frame px)
    for (tid, _, (x0, y0)), img, inv, quads in zip(jobs, imgs, inverses, scorer.detect(imgs)):
        for q in quads:
            qc = q if inv is None else cv2.transform(q[None].astype(np.float32), inv)[0]
            qf = qc / CROP_SCALE + (x0, y0)
            cx, cy = qf.mean(axis=0)
            dist, nearest = min((math.hypot(cx - dx, cy - dy), dtid) for dtid, dx, dy in dots)
            if nearest != tid or dist > ATTACH_DIST:
                continue  # text on a neighbouring robot (or the field), not this one
            text_img = warp_quad(img, q)
            if text_img is not None:
                found.append((tid, text_img, qf))
    return found


def _keep_best(scorer, found, teams, angle, best):
    for (tid, _, qf), (text, _, logp) in zip(found, scorer.score([f[1] for f in found], teams)):
        j = int(np.argmax(logp))
        per_char = float(logp[j]) / len(teams[j])
        if per_char < MIN_LOGP_PER_CHAR:
            continue
        if tid not in best or per_char > best[tid]["per_char"]:
            best[tid] = dict(track=tid, team=teams[j], text=text, per_char=per_char, angle=angle,
                             logp=[float(v) for v in logp], quad=qf.round(1).tolist())


def read_bumpers(scorer, frame, tracked, teams, layout):
    """Best accepted bumper read per tracked robot in this frame."""
    h, w = frame.shape[:2]
    _, top_h = layout.view_px(h)  # crops stay inside the view (e.g. not into close-up cameras)
    masked = mask_overlays(frame, layout)
    dots = [(tid, d[0], d[1]) for tid, d in tracked]

    jobs = []  # (track id, enlarged crop, crop origin)
    for tid, (x, y, _, _) in tracked:
        x0, y0 = int(max(0, x - CROP_HALF)), int(max(0, y - CROP_HALF))
        x1, y1 = int(min(w, x + CROP_HALF)), int(min(top_h, y + CROP_HALF))
        if x1 - x0 < 40 or y1 - y0 < 40:
            continue
        crop = cv2.resize(masked[y0:y1, x0:x1], None, fx=CROP_SCALE, fy=CROP_SCALE,
                          interpolation=cv2.INTER_CUBIC)
        jobs.append((tid, crop, (x0, y0)))

    best = {}
    _keep_best(scorer, _find_texts(scorer, jobs, dots, 0), teams, 0, best)
    for angle in RETRY_ANGLES:  # diagonal / sideways bumpers the detector missed upright
        retry = [j for j in jobs if j[0] not in best]
        if not retry:
            break
        _keep_best(scorer, _find_texts(scorer, retry, dots, angle), teams, angle, best)
    return list(best.values())
