"""Tests for tracking and cutting tracks. Each test pins down a failure seen on real matches.

Run from the repo root:
    .venv\\Scripts\\python -m unittest mtmct.debug.test_tracks
"""

import unittest

import numpy as np

from mtmct.debug.render_tracks import draw_tag
from mtmct.run.split_tracks import split_ambiguous, split_by_reads
from mtmct.run.track_robots import battery_problem
from mtmct.run.tracker import Track, Tracker
from ocr.run.read_bumpers import is_confident

RED, BLUE = ["1111", "2222", "3333"], ["4444", "5555", "6666"]
TEAMS = RED + BLUE
ALLIANCE_OF = {t: "red" for t in RED} | {t: "blue" for t in BLUE}


def track(tid, side, frames, xy):
    """A track seen at `frames`, at a fixed point or at xy(frame)."""
    t = Track(tid, frames[0])
    for i in frames:
        x, y = xy(i) if callable(xy) else xy
        t.add(i, x, y, side)
    return t


def read(tid, frame, team, per_char=-0.05):
    logp = [-20.0] * len(TEAMS)
    logp[TEAMS.index(team)] = per_char * len(team)
    return dict(track=tid, frame=frame, team=team, per_char=per_char, logp=logp)


class Tracking(unittest.TestCase):
    def test_follows_normal_motion(self):
        tr = Tracker()
        tr.update(0, [(100, 100, "red", 0.9)])
        tr.update(1, [(150, 100, "red", 0.9)])
        self.assertEqual(len(tr.tracks), 1)

    def test_never_jumps_after_missed_frames(self):
        # qm46 47.6 s: a track lost 5347 and jumped 256 px onto 1735.
        tr = Tracker()
        tr.update(0, [(100, 100, "red", 0.9)])
        tr.update(4, [(250, 100, "red", 0.9)])   # 150 px away after 3 missed frames
        self.assertEqual(len(tr.tracks), 2)
        self.assertEqual(list(tr.tracks[0].points), [0])
        self.assertTrue(tr.tracks[0].closed)     # the would-be jump ends the old track

    def test_confident_reads_split_a_track(self):
        tr = Tracker()
        tr.tracks[0] = track(0, "red", range(0, 41), (100, 100))
        tr._next = 1
        reads = [read(0, f, "1111") for f in (0, 5, 10)] + [read(0, f, "2222") for f in (20, 25)]
        self.assertEqual(split_by_reads(tr, reads, is_confident), 1)
        self.assertEqual(max(tr.tracks[0].points), 19)
        self.assertEqual({r["track"] for r in reads if r["team"] == "2222"}, {1})

    def test_one_stray_read_does_not_split(self):
        tr = Tracker()
        tr.tracks[0] = track(0, "red", range(0, 41), (100, 100))
        tr._next = 1
        reads = [read(0, f, "1111") for f in (0, 5, 15, 20)] + [read(0, 10, "2222")]
        self.assertEqual(split_by_reads(tr, reads, is_confident), 0)

    def test_ambiguous_resume_is_cut(self):
        # qm46 71-73.5 s: two hidden blue robots traded identities under the score bar.
        tracks = {0: track(0, "red", list(range(0, 11)) + list(range(20, 30)), (110, 100)),
                  1: track(1, "red", range(14, 17), (150, 100))}
        self.assertEqual(split_ambiguous(tracks, []), 1)
        self.assertEqual(max(tracks[0].points), 10)

    def test_merge_then_split_is_cut(self):
        # 2026mimar_qm50 42 s: 7597 drove behind 9183, one dot for 2 s, then two again;
        # 9183's track took 7597's dot without missing a frame.
        tracks = {0: track(0, "red", range(0, 60), (100, 100)),
                  1: track(1, "red", range(0, 20), (130, 90)),     # vanishes into track 0
                  2: track(2, "red", range(40, 60), (110, 125))}   # emerges from it
        self.assertEqual(split_ambiguous(tracks, []), 2)
        self.assertEqual(max(tracks[0].points), 19)                # before the merge
        self.assertEqual(sorted(tracks[3].points)[::19], [20, 39])  # the merged dot alone

    def test_merge_then_track_ends_cuts_off_the_merged_dot(self):
        # 2026mimar_qm50 81-83 s: 9183 merged into 7597's dot, which then sat on 9183's
        # bumper; the track ended and both robots restarted as new tracks.
        tracks = {0: track(0, "red", range(0, 40), (100, 100)),
                  1: track(1, "red", range(0, 20), (130, 90)),
                  2: track(2, "red", range(41, 60), (90, 110)),
                  3: track(3, "red", range(42, 60), (140, 120))}
        self.assertEqual(split_ambiguous(tracks, []), 1)
        self.assertEqual(max(tracks[0].points), 19)

    def test_robot_driving_past_without_merging_is_not_cut(self):
        tracks = {0: track(0, "red", range(0, 60), (100, 100)),
                  2: track(2, "red", range(40, 60), (110, 125))}   # appears, nothing vanished
        self.assertEqual(split_ambiguous(tracks, []), 0)

    def test_resume_far_from_other_alliance_is_not_cut(self):
        tracks = {0: track(0, "red", list(range(0, 11)) + list(range(20, 30)), (110, 100)),
                  1: track(1, "blue", range(14, 17), (150, 100))}
        self.assertEqual(split_ambiguous(tracks, []), 0)


class Power(unittest.TestCase):
    def check(self, battery):
        from unittest import mock
        with mock.patch("psutil.sensors_battery", return_value=battery):
            return battery_problem()

    def test_battery_guard(self):
        # A run once ended in a Windows critical-battery shutdown.
        from collections import namedtuple
        B = namedtuple("B", "percent secsleft power_plugged")
        self.assertIsNotNone(self.check(B(10, 600, False)))   # unplugged and low: stop
        self.assertIsNone(self.check(B(10, -2, True)))        # low but charging: fine
        self.assertIsNone(self.check(B(80, 3600, False)))     # unplugged, well charged: fine
        self.assertIsNone(self.check(None))                   # desktop, no battery


class Drawing(unittest.TestCase):
    def test_labels_do_not_overlap(self):
        img = np.zeros((600, 800, 3), np.uint8)
        placed = []
        draw_tag(img, 400, 300, "10063", (0, 0, 255), False, [], placed)
        draw_tag(img, 410, 315, "1768", (255, 0, 0), False, [], placed)
        (a0, b0, a1, b1), (c0, d0, c1, d1) = placed
        self.assertFalse(a0 < c1 and c0 < a1 and b0 < d1 and d0 < b1)


if __name__ == "__main__":
    unittest.main()
