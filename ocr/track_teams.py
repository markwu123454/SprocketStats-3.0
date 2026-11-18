"""Label every robot with its team number in every frame (top view of the broadcast).

Three pieces work together:
  1. Robot spotter (spotter.py, the repo's HRNet): one dot per robot + alliance colour, 10 fps.
  2. Tracker (below): links dots frame to frame, so each robot becomes one track. A track
     never jumps: after missed frames it may only pick its robot back up near where it was
     lost, and a run of confident bumper reads for another team splits it.
  3. Bumper reading (scoring.py), 2 fps: crop around each dot, find the bumper text (retrying
     on rotated copies when nothing is found), score the six team numbers. Evidence is summed
     along each track, and a track takes the team it best matches under two constraints: one
     robot per team at any moment, and the alliance colour the spotter saw. A long-lived track
     with no readable bumper can still be named by elimination: the only team on its alliance
     already seen elsewhere at the same time.

Run from the repo root:
    .venv\\Scripts\\python ocr\\track_teams.py [match_key ...]
    .venv\\Scripts\\python ocr\\track_teams.py --relabel <match_key>   # redo naming + video from saved JSON
Writes ocr/data/track_teams/<match_key>.mp4 (labelled top view), <match_key>.json
(evidence) and a few stills, then prints a summary.
"""

import itertools
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from layouts import layout_for
from quick_test import (DATA_DIR, MATCHES_JSON, download, match_teams,
                        pick_matches, sample_frames, tba_session)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# The OCR and spotter models are imported where they are used, so --relabel (naming and
# video only, from saved results) runs without loading torch or PaddleOCR.

OUT_DIR = DATA_DIR / "track_teams"

SPOT_FPS = 10.0           # spotter + tracker rate
OCR_EVERY = 5             # read bumpers on every 5th spotted frame -> 2 fps
# Where the field and the overlays are on screen differs per event: see layouts.py.

# Bumper reading
CROP_HALF = 110           # px around a robot dot; overhead robots are ~100 px across
CROP_SCALE = 3            # enlarge so 8-12 px bumper digits become 24-36 px
# Crops with no accepted read are re-detected rotated by these. 90 degrees catches sideways
# side-bumper text; +-30 was dropped after it found 3 of 618 reads on qm46 for ~2/3 of run time.
RETRY_ANGLES = (90,)
ATTACH_DIST = 85          # a text box belongs to the nearest robot dot within this many px
MIN_LOGP_PER_CHAR = -1.0  # weaker reads match no team (measured: real >= -0.91, junk <= -1.53)
CONFIDENT_PER_CHAR = -0.2  # reads at least this sure may split a track

# Naming tracks
READ_FLOOR = -8.0         # cap how far a single read can count against a team
MIN_MARGIN = 2.0          # log-evidence lead a track's team needs over the runner-up (~7x)
ALLIANCE_SHARE = 0.7      # trust the spotter's colour for a track if >= 70% of its dots agree
DUP_DIST = 90             # two dots closer than this (px) may be the same robot
MIN_APART = 5             # tracks are different robots only if seen this many frames far apart
ELIM_MIN_SECONDS = 3.0    # elimination only names tracks at least this long...
ELIM_MAX_DUP_SHARE = 0.5  # ...that don't sit on another track's robot for most of their life
BRIDGE_GAP_S = 4.0        # a piece is bridged if the same team ends/starts within this many s
MAX_SPEED_PX = 600        # px/s; faster implied motion between pieces is not the same robot
STITCH_GAP = 10           # frames: a piece starting this soon after a named piece ended...
STITCH_DIST = 100         # ...this close to where it ended, with no rival nearby, continues it
AMBIG_MIN_MISS = 3        # a track resuming after this many missed frames...
AMBIG_DIST = 150          # ...near another same-alliance robot (px)...
AMBIG_LOOKBACK = 10       # ...seen during the gap or this many frames before it, is cut there
MERGE_MAX_S = 5.0         # a robot vanishing into a track's dot and one emerging from it within
                          # this many s makes the track's identity after that a coin flip: cut

# Tracking
MIN_TRACK_LEN = 5         # shorter tracks are spotter noise
GATE_PX = 70              # max move between consecutive frames (px)
REACQUIRE_DIST = 100      # after missed frames, a dot farther than this from where the track
                          # was lost is a new track, never a jump
OLD_GATE_PER_MISS = 35    # the old gate's growth per missed frame; only used to spot would-be jumps
OTHER_ALLIANCE_PENALTY = 30  # px added to the matching cost when colours disagree
MAX_MISSES = 20           # frames a track can go unseen and still be picked back up (2 s)

# Power: a run keeps the GPU busy for minutes and drains a laptop battery fast (one run
# ended in a Windows critical-battery shutdown). Unplugged below this, don't start or go on.
MIN_BATTERY_PCT = 25
BATTERY_CHECK_EVERY = 50  # frames

# Drawing
ALLIANCE_BGR = {"red": (40, 40, 220), "blue": (220, 120, 30)}
LABEL_RISE = 70           # px between a robot's dot and its label, so the label clears the bumper


# ── Overlay handling ─────────────────────────────────────────────────────────

def mask_overlays(frame, layout):
    """Grey out the layout's overlays (score bar etc.): they print team numbers and digits."""
    out = frame.copy()
    h, w = out.shape[:2]
    for x0, y0, x1, y1 in layout.overlay_px(w, h):
        out[y0:y1, x0:x1] = 127
    return out


# ── Tracking ─────────────────────────────────────────────────────────────────

@dataclass
class Track:
    id: int
    first: int
    last: int = -1
    x: float = 0.0
    y: float = 0.0
    closed: bool = False
    votes: Counter = field(default_factory=Counter)
    points: dict = field(default_factory=dict)  # frame index -> (x, y, alliance the spotter saw)

    def add(self, i, x, y, side):
        self.x, self.y, self.last = x, y, i
        self.votes[side] += 1
        self.points[i] = (x, y, side)

    def alliance(self) -> tuple[str, float]:
        side, n = self.votes.most_common(1)[0]
        return side, n / sum(self.votes.values())

    def seconds(self) -> float:
        return (self.last - self.first + 1) / SPOT_FPS


class Tracker:
    """Greedy nearest-neighbour association of spotter dots across frames, without jumps."""

    def __init__(self):
        self.tracks: dict[int, Track] = {}
        self._next = 0
        self.jump_splits = 0

    def new_track(self, i) -> Track:
        t = Track(self._next, i)
        self._next += 1
        self.tracks[t.id] = t
        return t

    def update(self, i, dots):
        active = [t for t in self.tracks.values() if not t.closed and i - t.last <= MAX_MISSES]
        pairs = []
        for t in active:
            # Seen last frame: normal motion gate. After missed frames: the robot has to
            # reappear near where it was lost.
            gate = GATE_PX if i - t.last == 1 else REACQUIRE_DIST
            side = t.alliance()[0]
            for k, (x, y, cls, _) in enumerate(dots):
                d = math.hypot(x - t.x, y - t.y)
                if d <= gate:
                    pairs.append((d + (OTHER_ALLIANCE_PENALTY if cls != side else 0), t.id, k))
        pairs.sort()
        used_t, used_d, out = set(), set(), []
        for _, tid, k in pairs:
            if tid in used_t or k in used_d:
                continue
            used_t.add(tid)
            used_d.add(k)
            x, y, cls, _ = dots[k]
            self.tracks[tid].add(i, x, y, cls)
            out.append((tid, dots[k]))

        for k, (x, y, cls, _) in enumerate(dots):
            if k in used_d:
                continue
            # A coasting track that the old, ever-growing gate would have let jump onto this
            # dot is ended instead, so its identity can't carry over to another robot.
            jumpers = [(math.hypot(x - t.x, y - t.y), t.id) for t in active
                       if t.id not in used_t and not t.closed and i - t.last > 1
                       and REACQUIRE_DIST < math.hypot(x - t.x, y - t.y)
                       <= GATE_PX + OLD_GATE_PER_MISS * (i - t.last - 1)]
            if jumpers:
                self.tracks[min(jumpers)[1]].closed = True
                self.jump_splits += 1
            t = self.new_track(i)
            t.add(i, x, y, cls)
            out.append((t.id, dots[k]))
        return out

    def cut(self, track: Track, frame: int) -> Track:
        """Move `track`'s points from `frame` on into a new track and return it."""
        tail = self.new_track(frame)
        for i in sorted(k for k in track.points if k >= frame):
            x, y, side = track.points.pop(i)
            tail.add(i, x, y, side)
        track.last = max(track.points)
        track.votes = Counter(p[2] for p in track.points.values())
        tail.closed = track.closed
        track.closed = True
        return tail


def split_by_reads(tracker: Tracker, reads) -> int:
    """Split tracks where confident reads switch team.

    Confident reads of each track are grouped into runs of the same team, in time order.
    Only runs of 2+ reads count, so one stray read never splits a track. Wherever a counting
    run names a different team than the previous counting run, the track is cut at the
    run's first read, and each piece keeps only its own reads.
    """
    by_track = defaultdict(list)
    for r in reads:
        by_track[r["track"]].append(r)
    n_cuts = 0
    for tid, rs in by_track.items():
        confident = sorted((r for r in rs if r["per_char"] > CONFIDENT_PER_CHAR),
                           key=lambda r: r["frame"])
        runs = []
        for r in confident:
            if runs and runs[-1][0] == r["team"]:
                runs[-1][1].append(r)
            else:
                runs.append((r["team"], [r]))
        solid = [run for run in runs if len(run[1]) >= 2]
        cuts = [run[1][0]["frame"] for prev, run in zip(solid, solid[1:]) if run[0] != prev[0]]

        piece = tracker.tracks[tid]
        for frame in cuts:
            tail = tracker.cut(piece, frame)
            for r in rs:
                if r["frame"] >= frame:
                    r["track"] = tail.id
            piece = tail
            n_cuts += 1
    return n_cuts


def cut_track(track: Track, frame: int, new_id: int) -> Track:
    """Move `track`'s points from `frame` on into a new track with id `new_id`."""
    tail = Track(new_id, frame)
    for i in sorted(k for k in track.points if k >= frame):
        x, y, side = track.points.pop(i)
        tail.add(i, x, y, side)
    track.last = max(track.points)
    track.votes = Counter(p[2] for p in track.points.values())
    return tail


def split_ambiguous(tracks, reads, next_id=None) -> int:
    """Cut tracks where they could have swapped onto another robot of their alliance.

    1) Resume: while two same-alliance robots are both hidden (typically under the score
       bar), a track that lost its robot can resume on the other one without any jump.
    2) Merge-split: one robot drives behind another, the spotter sees one dot for both,
       then two again. The track keeps whichever dot is nearer -- a coin flip, never a
       miss (side views, 2026mimar_qm50 at 42 s: 9183's track slid onto 7597).
    Either way its later reads would name the whole track after the wrong robot. Cutting
    makes each piece answer for itself; pieces nobody can name stay unknown.
    """
    others = {tid: t for tid, t in tracks.items()}
    per_frame = frames_index(others)
    side_of = {tid: t.alliance()[0] for tid, t in tracks.items()}
    # Where each track starts and ends, taken before any cut moves points to a new piece.
    starts, ends = defaultdict(list), defaultdict(list)
    for o, ot in others.items():
        starts[ot.first].append((o, ot.points[ot.first]))
        ends[ot.last].append((o, ot.points[ot.last]))
    merge_frames = int(MERGE_MAX_S * SPOT_FPS)
    next_id = next_id if next_id is not None else max(tracks) + 1
    n_cuts = 0

    def near(t, frame, dot):
        """Is `dot` within DUP_DIST of t's dot at `frame`?"""
        x, y, _ = t.points[frame]
        return math.hypot(x - dot[0], y - dot[1]) <= DUP_DIST

    for tid in list(tracks):
        t = tracks[tid]
        frames = sorted(t.points)
        cuts = []
        for a, b in zip(frames, frames[1:]):
            if b - a - 1 < AMBIG_MIN_MISS:
                continue
            xb, yb, _ = t.points[b]
            if any(o != tid and side_of[o] == side_of[tid] and math.hypot(xb - ox, yb - oy) <= AMBIG_DIST
                   for i in range(a - AMBIG_LOOKBACK, b + 1) for o, ox, oy in per_frame.get(i, [])):
                cuts.append(b)
        # Merges: a same-alliance robot's track ends right on this track's dot. The dot is
        # then both robots at once, so it belongs to neither until they separate -- cut at
        # the merge, and again where one emerges. If this track itself ends and a robot
        # restarts at its last dot, the separation is confirmed the same way.
        for e in frames[:-1]:
            if not any(o != tid and side_of[o] == side_of[tid] and near(t, e, dot)
                       for o, dot in ends.get(e, [])):
                continue
            window = [f for f in frames if e < f <= e + merge_frames]
            split = next((s for s in window
                          if any(o != tid and side_of[o] == side_of[tid] and near(t, s, dot)
                                 for o, dot in starts.get(s, []))), None)
            if split is not None:
                cuts += [window[0], split]
            elif t.last <= e + merge_frames and any(
                    o != tid and side_of[o] == side_of[tid] and near(t, t.last, dot)
                    for s in range(t.last - STITCH_GAP, t.last + STITCH_GAP + 1)
                    for o, dot in starts.get(s, [])):
                cuts.append(window[0])
        cuts = sorted(set(cuts))
        piece = t
        for frame in cuts:
            tail = cut_track(piece, frame, next_id)
            tracks[next_id] = tail
            for r in reads:
                if r["track"] == piece.id and r["frame"] >= frame:
                    r["track"] = next_id
            next_id += 1
            n_cuts += 1
            piece = tail
    return n_cuts


def frames_index(tracks):
    """frame index -> [(track id, x, y)]"""
    per_frame = defaultdict(list)
    for t in tracks.values():
        for i, (x, y, _) in t.points.items():
            per_frame[i].append((t.id, x, y))
    return per_frame


# ── Bumper reading ───────────────────────────────────────────────────────────

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
    from scoring import warp_quad

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


# ── Naming tracks ────────────────────────────────────────────────────────────

def solid_enough_to_eliminate(t: Track) -> bool:
    """Has this track been seen enough to be named by elimination, with no bumper read?

    Elimination trusts that the track is a robot at all. Measured on 8 runs:
      every elimination checked correct ............ >= 30 dots, density >= 0.63
      2026milak_qm73 T121, a camera operator ........ 14 dots in a 30-frame span (0.47)
    where density = dots / (t.last - t.first + 1). The old rule, t.seconds() >= 3.0,
    measured the span only, so T121's flickering dot passed at exactly 3.0 s.
    """
    # TODO(human): replace this placeholder -- the old span-only rule -- with yours.
    return t.seconds() >= ELIM_MIN_SECONDS


def dup_share(tracks) -> dict[int, float]:
    """Fraction of each track's frames spent within DUP_DIST of another track's dot."""
    per_frame = frames_index(tracks)
    share = {}
    for t in tracks.values():
        close = sum(1 for i, (x, y, _) in t.points.items()
                    if any(o != t.id and math.hypot(x - ox, y - oy) < DUP_DIST
                           for o, ox, oy in per_frame[i]))
        share[t.id] = close / len(t.points)
    return share


def assign_teams(tracks, reads, teams, alliance_of):
    """track id -> (team, "read" | "elim" | "bridge" | "stitch", margin) for every nameable track."""
    evidence = defaultdict(lambda: np.zeros(len(teams)))
    for r in reads:
        lp = np.asarray(r["logp"])
        posterior = lp - np.logaddexp.reduce(lp)  # log P(team | this read), uniform prior
        evidence[r["track"]] += np.maximum(posterior, READ_FLOOR)

    allowed = {}
    for t in tracks.values():
        side, share = t.alliance()
        allowed[t.id] = [j for j, team in enumerate(teams)
                         if share < ALLIANCE_SHARE or alliance_of[team] == side]

    labels = {}
    apart_cache = {}

    def apart(a, b):
        """Frames where both tracks have a dot and the dots are far apart."""
        key = (min(a, b), max(a, b))
        if key not in apart_cache:
            pa, pb = tracks[a].points, tracks[b].points
            apart_cache[key] = sum(1 for i in pa.keys() & pb.keys()
                                   if math.hypot(pa[i][0] - pb[i][0], pa[i][1] - pb[i][1]) > DUP_DIST)
        return apart_cache[key]

    def taken(tid, team):
        """Is `team` provably a different robot while this track exists?

        Only frames where both tracks are seen far apart count. A broken track often
        overlaps its own continuation for a few frames, and that must not block the label.
        """
        return any(lab[0] == team and o != tid and apart(tid, o) >= MIN_APART
                   for o, lab in labels.items())

    # 1) Named by bumper reads, most decisive track first.
    pending = {r["track"] for r in reads}
    while pending:
        best = None
        for tid in pending:
            free = sorted((j for j in allowed[tid] if not taken(tid, teams[j])),
                          key=lambda j: -evidence[tid][j])
            if not free:
                continue
            if len(free) == 1:
                # The only team left. Count it as read if the reads favour it anyway.
                if free[0] != max(allowed[tid], key=lambda j: evidence[tid][j]):
                    continue  # reads point at a team that is taken: leave to elimination
                margin = math.inf
            else:
                margin = evidence[tid][free[0]] - evidence[tid][free[1]]
            if best is None or margin > best[2]:
                best = (tid, free[0], margin)
        if best is None or best[2] < MIN_MARGIN:
            break
        tid, j, margin = best
        labels[tid] = (teams[j], "read", min(float(margin), 999.0))
        pending.discard(tid)

    def plausible(t, team):
        """Could team's robot have driven from its previous piece to t, and from t to its next?"""
        gap = int(BRIDGE_GAP_S * SPOT_FPS)
        named = [tracks[o] for o, lab in labels.items() if lab[0] == team]
        before = [o for o in named if 0 <= t.first - o.last <= gap]
        after = [o for o in named if 0 <= o.first - t.last <= gap]
        if before:
            b = max(before, key=lambda o: o.last)
            if _speed(b.points[b.last], b.last, t.points[t.first], t.first) > MAX_SPEED_PX:
                return False
        if after:
            a = min(after, key=lambda o: o.first)
            if _speed(t.points[t.last], t.last, a.points[a.first], a.first) > MAX_SPEED_PX:
                return False
        return True

    # 2) Named by elimination: the only team on the track's alliance not already elsewhere.
    #    Only for tracks that last a while, aren't a second dot on another robot, and that
    #    team's robot could have driven to and from.
    dup = dup_share(tracks)
    changed = True
    while changed:
        changed = False
        for t in sorted(tracks.values(), key=lambda t: -len(t.points)):
            side, share = t.alliance()
            if (t.id in labels or share < ALLIANCE_SHARE or not solid_enough_to_eliminate(t)
                    or dup[t.id] > ELIM_MAX_DUP_SHARE):
                continue
            free = [team for team in teams if alliance_of[team] == side and not taken(t.id, team)]
            if len(free) == 1 and plausible(t, free[0]):
                labels[t.id] = (free[0], "elim", 0.0)
                changed = True

    # 3) Bridged: a short unnamed piece (typically a robot half hidden under the score bar)
    #    between a piece of team X that ended just before it and one that starts just after
    #    it. Like elimination, every other robot of the alliance must be seen elsewhere
    #    meanwhile: "X fits both sides" says nothing when another robot is hidden too and
    #    could be this one (2026mimar_qm50 38 s: all three blue robots were out of sight,
    #    5926 and 818 both "fit", and the piece was 6120).
    gap = int(BRIDGE_GAP_S * SPOT_FPS)
    changed = True
    while changed:
        changed = False
        for t in tracks.values():
            side, share = t.alliance()
            if t.id in labels or share < ALLIANCE_SHARE or dup[t.id] > ELIM_MAX_DUP_SHARE:
                continue
            free = [team for team in teams if alliance_of[team] == side and not taken(t.id, team)]
            if len(free) != 1:
                continue
            fits = []  # (implied speed px/s, team)
            for team in free:
                named = [tracks[o] for o, lab in labels.items() if lab[0] == team]
                before = [o for o in named if 0 <= t.first - o.last <= gap]
                after = [o for o in named if 0 <= o.first - t.last <= gap]
                if before and after:
                    b = max(before, key=lambda o: o.last)
                    a = min(after, key=lambda o: o.first)
                    speed = max(_speed(b.points[b.last], b.last, t.points[t.first], t.first),
                                _speed(t.points[t.last], t.last, a.points[a.first], a.first))
                    if speed <= MAX_SPEED_PX:
                        fits.append((speed, team))
            fits.sort()
            # One team fits, or one fits clearly better: its robot was much nearer, both sides.
            if len(fits) == 1 or (len(fits) > 1 and fits[0][0] <= 0.5 * fits[1][0]):
                labels[t.id] = (fits[0][1], "bridge", 0.0)
                changed = True

    # 4) Stitched: an unnamed piece that starts right where a named piece of team X just
    #    ended (or ends right where one starts), with no other robot of its alliance near
    #    that junction -- the same robot's track broke and restarted. One-sided, so it covers
    #    the start and end of a match that bridging cannot.
    per_frame = frames_index(tracks)
    side_of = {tid: tr.alliance()[0] for tid, tr in tracks.items()}

    def lonely(tid, other, i, x, y):
        """No rival of tid's alliance near (x, y) around frame i.

        A robot still seen on its own track at frame i, clearly apart from (x, y), is no
        rival: it is visibly a different robot. The risky case is a nearby robot that is
        hidden at that moment, which could be the one that just reappeared.
        """
        for f in range(i - STITCH_GAP, i + STITCH_GAP + 1):
            for o, ox, oy in per_frame.get(f, []):
                if o in (tid, other) or side_of[o] != side_of[tid]:
                    continue
                if math.hypot(x - ox, y - oy) > AMBIG_DIST:
                    continue
                now = tracks[o].points.get(i)
                if now is not None and math.hypot(x - now[0], y - now[1]) > DUP_DIST:
                    continue  # visibly a different robot at the hand-over
                return False
        return True

    changed = True
    while changed:
        changed = False
        for t in tracks.values():
            side, share = t.alliance()
            if t.id in labels or share < ALLIANCE_SHARE or dup[t.id] > ELIM_MAX_DUP_SHARE:
                continue
            fits = set()
            for o, lab in labels.items():
                p = tracks[o]
                if alliance_of[lab[0]] != side or taken(t.id, lab[0]):
                    continue
                if 0 <= t.first - p.last <= STITCH_GAP:   # p ends, t starts
                    (ax, ay, _), (bx, by, _) = p.points[p.last], t.points[t.first]
                    if math.hypot(ax - bx, ay - by) <= STITCH_DIST and lonely(t.id, o, t.first, bx, by):
                        fits.add(lab[0])
                if 0 <= p.first - t.last <= STITCH_GAP:   # t ends, p starts
                    (ax, ay, _), (bx, by, _) = t.points[t.last], p.points[p.first]
                    if math.hypot(ax - bx, ay - by) <= STITCH_DIST and lonely(t.id, o, t.last, ax, ay):
                        fits.add(lab[0])
            if len(fits) == 1:
                labels[t.id] = (fits.pop(), "stitch", 0.0)
                changed = True
    return labels


def _speed(p, i, q, j) -> float:
    """Implied speed (px/s) to get from point p at frame i to point q at frame j."""
    return math.hypot(q[0] - p[0], q[1] - p[1]) / max(abs(j - i) / SPOT_FPS, 0.1)


def mark_noise(tracks, reads, labels, alliance_of) -> set[int]:
    """Unnamed tracks that are spotter mistakes rather than robots.

    Only tracks without a single accepted bumper read qualify. A track is noise if, for
    most of its life, it either sits on another robot (a second dot, e.g. a blue dot on a
    red robot) or appears while all three teams of its alliance are named on tracks
    elsewhere -- a fourth robot of an alliance cannot exist.
    """
    has_reads = {r["track"] for r in reads}
    per_frame = frames_index(tracks)
    dup = dup_share(tracks)
    noise = set()
    for t in tracks.values():
        if t.id in labels or t.id in has_reads:
            continue
        side = t.alliance()[0]
        need = {team for team, s in alliance_of.items() if s == side}
        surplus = 0
        for i, (x, y, _) in t.points.items():
            elsewhere = {labels[o][0] for o, ox, oy in per_frame[i]
                         if o in labels and math.hypot(x - ox, y - oy) > DUP_DIST}
            surplus += need <= elsewhere
        if dup[t.id] > 0.5 or surplus / len(t.points) > 0.5:
            noise.add(t.id)
    return noise


# ── Output ───────────────────────────────────────────────────────────────────

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


def _ranges(frames, gap=int(SPOT_FPS)):
    """Sorted frame indices -> [(start, end)], joining gaps up to `gap` frames."""
    out = []
    for i in frames:
        if out and i - out[-1][1] <= gap:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def summarize(match_key, stats, tracks, n_noise, noise_frames, reads, labels, per_frame, teams,
              alliance_of, out_mp4):
    n_frames = stats["n_frames"]
    robot_frames = [tid for lst in per_frame.values() for tid, _, _ in lst if tid in tracks]
    how = Counter(labels[tid][1] if tid in labels else "unknown" for tid in robot_frames)
    total = max(1, len(robot_frames))
    disagree = [r for r in reads if r["track"] in labels and r["team"] != labels[r["track"]][0]]
    double = sum(1 for lst in per_frame.values()
                 for a in range(len(lst)) for b in range(a + 1, len(lst))
                 if lst[a][0] in labels and lst[b][0] in labels
                 and labels[lst[a][0]][0] == labels[lst[b][0]][0]
                 and math.hypot(lst[a][1] - lst[b][1], lst[a][2] - lst[b][2]) > DUP_DIST)
    angles = Counter(r["angle"] for r in reads)

    print(f"\n{match_key}: {n_frames} frames @ {SPOT_FPS:g} fps, {len(tracks)} tracks "
          f"({stats.get('jump_splits', '?')} jump splits, {stats.get('read_splits', '?')} read splits, "
          f"{stats.get('ambiguity_splits', '?')} ambiguity splits), "
          f"{len(reads)} accepted reads from {stats.get('n_crops', '?')} robot crops "
          f"({stats.get('minutes', '?')} min)")
    print(f"  spotter noise dropped: {n_noise} tracks, {noise_frames} dots (second dots on a robot, "
          f"or a 4th robot of an alliance)")
    print(f"  reads by crop rotation: " + ", ".join(f"{a}°: {n}" for a, n in sorted(angles.items())))
    print(f"  robot-frames labelled: {(total - how['unknown']) / total:.1%} "
          f"(by bumper reads {how['read'] / total:.1%}, by elimination {how['elim'] / total:.1%}, "
          f"bridged {how['bridge'] / total:.1%}, stitched {how['stitch'] / total:.1%}, "
          f"unknown {how['unknown'] / total:.1%})")
    print(f"  frames where two robots far apart share a label: {double}")
    print(f"  reads that disagree with their track's final label: {len(disagree)} of {len(reads)}")
    for r in sorted(disagree, key=lambda r: r["frame"]):
        print(f"    t={r['frame'] / SPOT_FPS:6.1f}s track {r['track']:>3} labelled "
              f"{labels[r['track']][0]:>5}: read {r['text']!r} -> {r['team']} "
              f"(per_char {r['per_char']:.2f})")

    print("  labelled time ranges (video seconds):")
    for team in teams:
        own = sorted((t for tid, t in tracks.items() if labels.get(tid, (None,))[0] == team),
                     key=lambda t: t.first)
        frames = sorted({i for t in own for i in t.points})  # a set: duplicate dots count once
        spans = ", ".join(f"{a / SPOT_FPS:.1f}-{b / SPOT_FPS:.1f}" for a, b in _ranges(frames))
        n_reads = sum(1 for r in reads if r["track"] in labels and labels[r["track"]][0] == team)
        pieces = " ".join(f"t{t.id}:{labels[t.id][1]}" for t in own)
        print(f"    {team:>5} {alliance_of[team]:>4} {len(frames) / n_frames:4.0%} of frames, "
              f"{n_reads:3} reads | {spans or '-'} | {pieces}")
    print(f"  video: {out_mp4}")


# ── Main ─────────────────────────────────────────────────────────────────────

def battery_problem() -> str | None:
    """A message if this is a laptop running on battery below MIN_BATTERY_PCT, else None."""
    try:
        import psutil
        b = psutil.sensors_battery()
    except Exception:
        return None  # no battery info: a desktop, or psutil missing
    if b is None or b.power_plugged or b.percent >= MIN_BATTERY_PCT:
        return None
    return (f"running on battery at {b.percent:.0f}% (minimum {MIN_BATTERY_PCT}% unplugged); "
            f"plug in the charger and run again")


def run_match(match_key, video, alliances, spotter, scorer):
    teams = alliances["red"] + alliances["blue"]
    alliance_of = {t: side for side, ts in alliances.items() for t in ts}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    layout = layout_for(match_key)
    tracker, reads = Tracker(), []
    start, n_frames, n_crops = time.perf_counter(), 0, 0
    for i, (t, frame) in enumerate(sample_frames(video, SPOT_FPS)):
        n_frames += 1
        if n_frames % BATTERY_CHECK_EVERY == 0 and (problem := battery_problem()):
            sys.exit(f"{match_key}: stopped at {t:.0f}s of video, {problem}")
        h, w = frame.shape[:2]
        dots = [d for d in spotter(frame) if layout.keeps(d[0], d[1], w, h)]
        tracked = tracker.update(i, dots)
        if i % OCR_EVERY == 0 and tracked:
            n_crops += len(tracked)
            reads += [{**r, "frame": i} for r in read_bumpers(scorer, frame, tracked, teams, layout)]
        if n_frames % 300 == 0:
            print(f"  {match_key}: {n_frames} frames ({t:.0f}s), {len(reads)} reads", flush=True)

    read_splits = split_by_reads(tracker, reads)
    tracks = {tid: t for tid, t in tracker.tracks.items() if len(t.points) >= MIN_TRACK_LEN}
    reads = [r for r in reads if r["track"] in tracks]
    stats = dict(n_crops=n_crops, jump_splits=tracker.jump_splits, read_splits=read_splits,
                 minutes=round((time.perf_counter() - start) / 60, 1))
    finish(match_key, video, alliances, tracks, reads, stats)


def finish(match_key, video, alliances, tracks, reads, stats, offset=0, limit=None):
    """Name the tracks, drop spotter noise, render the video, save evidence, print the summary."""
    teams = alliances["red"] + alliances["blue"]
    alliance_of = {t: side for side, ts in alliances.items() for t in ts}
    layout = layout_for(match_key)
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    # Tracks saved before the current field outline: drop those that live off the field.
    outside = {tid for tid, t in tracks.items()
               if sum(not layout.in_field(x, y, w, h) for x, y, _ in t.points.values()) > len(t.points) / 2}
    tracks_all = tracks
    tracks = {tid: t for tid, t in tracks_all.items() if tid not in outside}
    stats["ambiguity_splits"] = split_ambiguous(tracks, reads, next_id=max(tracks_all) + 1)
    labels = assign_teams(tracks, reads, teams, alliance_of)
    noise = mark_noise(tracks, reads, labels, alliance_of) | outside
    tracks_all.update(tracks)  # keep the pieces split_ambiguous created
    tracks = tracks_all
    shown = {tid: t for tid, t in tracks.items() if tid not in noise}
    per_frame = frames_index(shown)
    out_mp4, stats["n_frames"] = render(video, match_key, per_frame, shown, labels, reads,
                                        alliance_of, layout, offset, limit)

    (OUT_DIR / f"{match_key}.json").write_text(json.dumps({
        "teams": alliances,
        "layout": layout.name,
        "video": str(video),
        "offset": offset,
        "stats": stats,
        "labels": {str(tid): list(lab) for tid, lab in labels.items()},
        "noise": sorted(noise),
        "tracks": {str(tid): dict(first=t.first, last=t.last, points=len(t.points),
                                  alliance=t.alliance()[0], share=round(t.alliance()[1], 2),
                                  xy={i: [round(x), round(y), s] for i, (x, y, s) in t.points.items()})
                   for tid, t in tracks.items()},
        "reads": reads,
    }, indent=1))
    noise_frames = sum(len(tracks[tid].points) for tid in noise)
    summarize(match_key, stats, shown, len(noise), noise_frames, reads, labels, per_frame,
              teams, alliance_of, out_mp4)


def relabel(match_key):
    """Redo naming, noise filtering and the video from a saved run, without any OCR."""
    d = json.loads((OUT_DIR / f"{match_key}.json").read_text())
    tracks = {}
    for tid, td in d["tracks"].items():
        t = Track(int(tid), td["first"])
        for i, (x, y, side) in sorted(td["xy"].items(), key=lambda kv: int(kv[0])):
            t.add(int(i), x, y, side)
        tracks[t.id] = t
    if "video" in d:
        video = Path(d["video"])
    else:
        videos = {m["match_key"]: m["videos"] for m in json.loads(MATCHES_JSON.read_text(encoding="utf-8"))}
        video = download(videos[match_key][0])
    stats = d.get("stats", {})
    offset = d.get("offset", 0)
    limit = stats.pop("n_frames", None)  # keep the original run's length (partial runs too)
    finish(match_key, video, d["teams"], tracks, d["reads"], stats, offset, limit)


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["--relabel"]:
        for key in args[1:]:
            relabel(key)
        return

    if problem := battery_problem():
        sys.exit(f"Not starting: {problem}")

    from scoring import TeamScorer
    from spotter import RobotSpotter

    keys = args or [m["match_key"] for m in pick_matches()]
    videos = {m["match_key"]: m["videos"] for m in json.loads(MATCHES_JSON.read_text(encoding="utf-8"))}
    session = tba_session()
    spotter, scorer = RobotSpotter(), TeamScorer()
    for key in keys:
        if not videos.get(key):
            sys.exit(f"{key}: no video listed in {MATCHES_JSON}")
        alliances = match_teams(session, key)
        print(f"{key}: red {alliances['red']}  blue {alliances['blue']}")
        run_match(key, download(videos[key][0]), alliances, spotter, scorer)


if __name__ == "__main__":
    main()
