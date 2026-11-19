"""Naming tracks: give each finished track a team number, leave it unknown, or drop it as
spotter noise. That is ocr's whole job in the pipeline; the tracks, and where they are
cut, come from mtmct.

A track's team comes from its bumper reads (read_bumpers.py), summed along the track,
under two constraints: one robot per team at any moment, and the alliance colour the
spotter saw. A track without a decisive read can still be named by elimination (the only
team of its alliance not seen elsewhere at the time), bridging (a short piece between two
pieces of the same team) or stitching (a piece that starts where a named one ended).
"""

import math
from collections import defaultdict

import numpy as np

from mtmct.run.split_tracks import AMBIG_DIST
from mtmct.run.tracker import DUP_DIST, FPS as SPOT_FPS, STITCH_GAP, Track, frames_index

READ_FLOOR = -8.0         # cap how far a single read can count against a team
MIN_MARGIN = 2.0          # log-evidence lead a track's team needs over the runner-up (~7x)
ALLIANCE_SHARE = 0.7      # trust the spotter's colour for a track if >= 70% of its dots agree
MIN_APART = 5             # tracks are different robots only if seen this many frames far apart
ELIM_MIN_SECONDS = 3.0    # elimination only names tracks at least this long...
ELIM_MAX_DUP_SHARE = 0.5  # ...that don't sit on another track's robot for most of their life
BRIDGE_GAP_S = 4.0        # a piece is bridged if the same team ends/starts within this many s
MAX_SPEED_PX = 600        # px/s; faster implied motion between pieces is not the same robot
STITCH_DIST = 100         # a piece starting within STITCH_GAP frames of a named piece's end, this
                          # close (px) to where it ended, with no rival nearby, continues it


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
