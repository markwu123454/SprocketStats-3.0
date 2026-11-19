"""Cutting tracks after association, where one track may have changed robots.

The tracker (tracker.py) refuses the jumps it can see frame to frame. Two kinds of
identity change get past it, and are cut here so each piece answers for one robot:
  - split_by_reads: confident bumper reads attached to the track switch team partway.
  - split_ambiguous: geometry alone says the track could have swapped onto another
    robot of its alliance (resuming next to one, or merging with one and splitting).
A needless cut costs little, since naming can still give both pieces the same team; a
missed swap names the whole track after the wrong robot. Both functions re-point the
reads ({"track", "frame", ...}) that fall in a piece that was cut off.
"""

import math
from collections import defaultdict

from mtmct.run.tracker import DUP_DIST, FPS, STITCH_GAP, Tracker, cut_track, frames_index

AMBIG_MIN_MISS = 3        # a track resuming after this many missed frames...
AMBIG_DIST = 150          # ...near another same-alliance robot (px)...
AMBIG_LOOKBACK = 10       # ...seen during the gap or this many frames before it, is cut there
MERGE_MAX_S = 5.0         # a robot vanishing into a track's dot and one emerging from it within
                          # this many s makes the track's identity after that a coin flip: cut


def split_by_reads(tracker: Tracker, reads, is_confident) -> int:
    """Split tracks where confident reads switch team. `is_confident(read)` comes from ocr,
    which owns what a confident read is (ocr/run/read_bumpers.py).

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
        confident = sorted((r for r in rs if is_confident(r)),
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
    merge_frames = int(MERGE_MAX_S * FPS)
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
