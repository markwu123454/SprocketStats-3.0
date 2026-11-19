"""Frame-to-frame robot tracking: link per-frame robot dots into tracks.

Greedy nearest-neighbour association with one hard rule: a track never jumps. After
missed frames it may only pick its robot back up near where it was lost; a dot farther
away starts a new track. A jump would carry one robot's identity onto another, while a
broken track is only two pieces that ocr's naming step can still join (bridge/stitch).

Cutting tracks after association is split_tracks.py; naming them is ocr's job
(ocr/run/assign_teams.py). This file only ever sees dots.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# The gates below are pixels per frame at this rate (MAX_MISSES = 20 is 2 s). Feed the
# tracker frames at another rate and they all need retuning.
FPS = 10.0

MIN_TRACK_LEN = 5         # shorter tracks are spotter noise
GATE_PX = 70              # max move between consecutive frames (px)
REACQUIRE_DIST = 100      # after missed frames, a dot farther than this from where the track
                          # was lost is a new track, never a jump
OLD_GATE_PER_MISS = 35    # the old gate's growth per missed frame; only used to spot would-be jumps
OTHER_ALLIANCE_PENALTY = 30  # px added to the matching cost when colours disagree
MAX_MISSES = 20           # frames a track can go unseen and still be picked back up (2 s)
DUP_DIST = 90             # two dots closer than this (px) may be the same robot
STITCH_GAP = 10           # frames around one piece's end and the next one's start that count
                          # as a hand-over (ambiguity cuts in split_tracks, stitching in naming)


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
        return (self.last - self.first + 1) / FPS


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


def cut_track(track: Track, frame: int, new_id: int) -> Track:
    """Move `track`'s points from `frame` on into a new track with id `new_id`."""
    tail = Track(new_id, frame)
    for i in sorted(k for k in track.points if k >= frame):
        x, y, side = track.points.pop(i)
        tail.add(i, x, y, side)
    track.last = max(track.points)
    track.votes = Counter(p[2] for p in track.points.values())
    return tail


def frames_index(tracks):
    """frame index -> [(track id, x, y)]"""
    per_frame = defaultdict(list)
    for t in tracks.values():
        for i, (x, y, _) in t.points.items():
            per_frame[i].append((t.id, x, y))
    return per_frame
