"""Track every robot through a match and name it with its team number (broadcast top view).

The end-to-end pass, one step per module:
  1. Robot spotter (hrnet/run/spotter.py): one dot per robot + alliance colour, 10 fps.
  2. Tracker (tracker.py): links dots frame to frame, so each robot becomes one track. A
     track never jumps: after missed frames it may only pick its robot back up near where
     it was lost.
  3. Bumper reads (ocr/run/read_bumpers.py), 2 fps: bumper text scored against the six
     teams, attached to the track it was read on.
  4. Cutting (split_tracks.py): where confident reads switch team, and where a track could
     have swapped onto another robot of its alliance.
  5. Naming (ocr/run/assign_teams.py): each track gets a team, stays unknown, or is dropped
     as spotter noise.
This file owns the order and the saved result; each step owns its own rules. The labelled
video is a debug view of the saved result: python -m mtmct.debug.render_tracks <match_key>.

Run from the repo root:
    .venv\\Scripts\\python -m mtmct.run.track_robots [match_key ...]
    .venv\\Scripts\\python -m mtmct.run.track_robots --relabel <match_key>   # redo cuts + naming from saved JSON
Writes mtmct/data/tracks/<match_key>.json (tracks, reads, names), then prints a summary.
"""

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

from better_sampling.match_data import (MATCHES_JSON, download, match_teams, pick_matches,
                                        sample_frames, tba_session)
from broadcast.run.layouts import layout_for
from mtmct.run.split_tracks import split_ambiguous, split_by_reads
from mtmct.run.tracker import FPS as SPOT_FPS  # spotter + tracker rate; the tracker's gates assume it
from mtmct.run.tracker import DUP_DIST, MIN_TRACK_LEN, Track, Tracker, frames_index
from ocr.run.assign_teams import assign_teams, mark_noise
from ocr.run.read_bumpers import OCR_EVERY, is_confident, read_bumpers

# The OCR and spotter models are imported where they are used, so --relabel (cuts and
# naming only, from saved results) runs without loading torch or PaddleOCR.

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "tracks"

# Power: a run keeps the GPU busy for minutes and drains a laptop battery fast (one run
# ended in a Windows critical-battery shutdown). Unplugged below this, don't start or go on.
MIN_BATTERY_PCT = 25
BATTERY_CHECK_EVERY = 50  # frames


def tracks_from_saved(d) -> dict[int, Track]:
    """Rebuild Track objects from a saved run's JSON (relabel, and the debug renderer)."""
    tracks = {}
    for tid, td in d["tracks"].items():
        t = Track(int(tid), td["first"])
        for i, (x, y, side) in sorted(td["xy"].items(), key=lambda kv: int(kv[0])):
            t.add(int(i), x, y, side)
        tracks[t.id] = t
    return tracks


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
              alliance_of):
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
    print(f"  video: python -m mtmct.debug.render_tracks {match_key}")


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

    read_splits = split_by_reads(tracker, reads, is_confident)
    tracks = {tid: t for tid, t in tracker.tracks.items() if len(t.points) >= MIN_TRACK_LEN}
    reads = [r for r in reads if r["track"] in tracks]
    stats = dict(n_crops=n_crops, jump_splits=tracker.jump_splits, read_splits=read_splits,
                 minutes=round((time.perf_counter() - start) / 60, 1), n_frames=n_frames)
    finish(match_key, video, alliances, tracks, reads, stats)


def finish(match_key, video, alliances, tracks, reads, stats, offset=0):
    """Cut ambiguous tracks, name them, drop spotter noise, save the result, print the summary."""
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
              teams, alliance_of)


def relabel(match_key):
    """Redo ambiguity cuts, naming and noise filtering from a saved run, without any OCR."""
    d = json.loads((OUT_DIR / f"{match_key}.json").read_text())
    tracks = tracks_from_saved(d)
    if "video" in d:
        video = Path(d["video"])
    else:
        videos = {m["match_key"]: m["videos"] for m in json.loads(MATCHES_JSON.read_text(encoding="utf-8"))}
        video = download(videos[match_key][0])
    stats = d.get("stats", {})
    offset = d.get("offset", 0)
    finish(match_key, video, d["teams"], tracks, d["reads"], stats, offset)


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["--relabel"]:
        for key in args[1:]:
            relabel(key)
        return

    if problem := battery_problem():
        sys.exit(f"Not starting: {problem}")

    from hrnet.run.spotter import RobotSpotter
    from ocr.run.team_scorer import TeamScorer

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
