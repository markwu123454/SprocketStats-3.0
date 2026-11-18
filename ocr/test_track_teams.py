"""Tests for the team-tracking rules. Each test pins down a failure seen on real matches.

Run from the repo root:
    .venv\\Scripts\\python -m unittest discover -s ocr -p "test_*.py"
"""

import unittest

import numpy as np

import track_teams as tt
from quick_test import match_team

RED, BLUE = ["1111", "2222", "3333"], ["4444", "5555", "6666"]
TEAMS = RED + BLUE
ALLIANCE_OF = {t: "red" for t in RED} | {t: "blue" for t in BLUE}


def track(tid, side, frames, xy):
    """A track seen at `frames`, at a fixed point or at xy(frame)."""
    t = tt.Track(tid, frames[0])
    for i in frames:
        x, y = xy(i) if callable(xy) else xy
        t.add(i, x, y, side)
    return t


def read(tid, frame, team, per_char=-0.05):
    logp = [-20.0] * len(TEAMS)
    logp[TEAMS.index(team)] = per_char * len(team)
    return dict(track=tid, frame=frame, team=team, per_char=per_char, logp=logp)


class MatchTeam(unittest.TestCase):
    def test_fuzzy_match(self):
        teams = ["1071", "8013", "9729", "1735", "5347", "10393"]
        self.assertEqual(match_team("534", teams), "5347")     # dropped digit
        self.assertEqual(match_team("10309", teams), None)     # two edits away
        self.assertEqual(match_team("17", teams), None)        # too short to trust


class Tracking(unittest.TestCase):
    def test_follows_normal_motion(self):
        tr = tt.Tracker()
        tr.update(0, [(100, 100, "red", 0.9)])
        tr.update(1, [(150, 100, "red", 0.9)])
        self.assertEqual(len(tr.tracks), 1)

    def test_never_jumps_after_missed_frames(self):
        # qm46 47.6 s: a track lost 5347 and jumped 256 px onto 1735.
        tr = tt.Tracker()
        tr.update(0, [(100, 100, "red", 0.9)])
        tr.update(4, [(250, 100, "red", 0.9)])   # 150 px away after 3 missed frames
        self.assertEqual(len(tr.tracks), 2)
        self.assertEqual(list(tr.tracks[0].points), [0])
        self.assertTrue(tr.tracks[0].closed)     # the would-be jump ends the old track

    def test_confident_reads_split_a_track(self):
        tr = tt.Tracker()
        tr.tracks[0] = track(0, "red", range(0, 41), (100, 100))
        tr._next = 1
        reads = [read(0, f, "1111") for f in (0, 5, 10)] + [read(0, f, "2222") for f in (20, 25)]
        self.assertEqual(tt.split_by_reads(tr, reads), 1)
        self.assertEqual(max(tr.tracks[0].points), 19)
        self.assertEqual({r["track"] for r in reads if r["team"] == "2222"}, {1})

    def test_one_stray_read_does_not_split(self):
        tr = tt.Tracker()
        tr.tracks[0] = track(0, "red", range(0, 41), (100, 100))
        tr._next = 1
        reads = [read(0, f, "1111") for f in (0, 5, 15, 20)] + [read(0, 10, "2222")]
        self.assertEqual(tt.split_by_reads(tr, reads), 0)

    def test_ambiguous_resume_is_cut(self):
        # qm46 71-73.5 s: two hidden blue robots traded identities under the score bar.
        tracks = {0: track(0, "red", list(range(0, 11)) + list(range(20, 30)), (110, 100)),
                  1: track(1, "red", range(14, 17), (150, 100))}
        self.assertEqual(tt.split_ambiguous(tracks, []), 1)
        self.assertEqual(max(tracks[0].points), 10)

    def test_merge_then_split_is_cut(self):
        # 2026mimar_qm50 42 s: 7597 drove behind 9183, one dot for 2 s, then two again;
        # 9183's track took 7597's dot without missing a frame.
        tracks = {0: track(0, "red", range(0, 60), (100, 100)),
                  1: track(1, "red", range(0, 20), (130, 90)),     # vanishes into track 0
                  2: track(2, "red", range(40, 60), (110, 125))}   # emerges from it
        self.assertEqual(tt.split_ambiguous(tracks, []), 2)
        self.assertEqual(max(tracks[0].points), 19)                # before the merge
        self.assertEqual(sorted(tracks[3].points)[::19], [20, 39])  # the merged dot alone

    def test_merge_then_track_ends_cuts_off_the_merged_dot(self):
        # 2026mimar_qm50 81-83 s: 9183 merged into 7597's dot, which then sat on 9183's
        # bumper; the track ended and both robots restarted as new tracks.
        tracks = {0: track(0, "red", range(0, 40), (100, 100)),
                  1: track(1, "red", range(0, 20), (130, 90)),
                  2: track(2, "red", range(41, 60), (90, 110)),
                  3: track(3, "red", range(42, 60), (140, 120))}
        self.assertEqual(tt.split_ambiguous(tracks, []), 1)
        self.assertEqual(max(tracks[0].points), 19)

    def test_robot_driving_past_without_merging_is_not_cut(self):
        tracks = {0: track(0, "red", range(0, 60), (100, 100)),
                  2: track(2, "red", range(40, 60), (110, 125))}   # appears, nothing vanished
        self.assertEqual(tt.split_ambiguous(tracks, []), 0)

    def test_resume_far_from_other_alliance_is_not_cut(self):
        tracks = {0: track(0, "red", list(range(0, 11)) + list(range(20, 30)), (110, 100)),
                  1: track(1, "blue", range(14, 17), (150, 100))}
        self.assertEqual(tt.split_ambiguous(tracks, []), 0)


class Naming(unittest.TestCase):
    def base(self):
        """Two red robots named by reads and visible all along, far from the action."""
        tracks = {1: track(1, "red", range(0, 101), (900, 500)),
                  2: track(2, "red", range(0, 101), (900, 300)),
                  3: track(3, "red", range(0, 41), (100, 100))}
        reads = [read(1, 0, "2222"), read(2, 0, "3333"), read(3, 0, "1111")]
        return tracks, reads

    def test_elimination_names_the_last_team(self):
        tracks, reads = self.base()
        tracks[4] = track(4, "red", range(45, 101), (150, 100))
        labels = tt.assign_teams(tracks, reads, TEAMS, ALLIANCE_OF)
        self.assertEqual(labels[4][:2], ("1111", "elim"))

    @unittest.expectedFailure  # known bug until solid_enough_to_eliminate() gets a real rule
    def test_elimination_skips_a_flickering_dot(self):
        # 2026milak_qm73 130 s: a camera operator's dot, 14 dots over 3.0 s, was named 3875.
        tracks, reads = self.base()
        tracks[4] = track(4, "red", list(range(60, 86, 2)) + [89], (400, 100))   # too far to stitch
        self.assertNotIn(4, tt.assign_teams(tracks, reads, TEAMS, ALLIANCE_OF))

    def test_elimination_keeps_a_patchy_real_robot(self):
        # 2026milak_qm73 110 s: 9215 along the far wall, 31 dots over 4.9 s, named correctly.
        tracks, reads = self.base()
        tracks[4] = track(4, "red", sorted(set(range(60, 81)) | set(range(83, 108, 3)) | {108}), (400, 100))
        self.assertEqual(tt.assign_teams(tracks, reads, TEAMS, ALLIANCE_OF)[4][:2], ("1111", "elim"))

    def test_elimination_needs_a_believable_drive(self):
        # qm18 171 s: a dot on a person behind the glass was named 5962 by elimination.
        tracks, reads = self.base()
        tracks[4] = track(4, "red", range(45, 101), (1800, 100))  # 1700 px in 0.5 s
        labels = tt.assign_teams(tracks, reads, TEAMS, ALLIANCE_OF)
        self.assertNotIn(4, labels)

    def test_stitch_continues_a_restarted_track(self):
        # qm52 171 s: 5347's track restarted 74 px away with no reads.
        tracks, reads = self.base()
        tracks[4] = track(4, "red", range(41, 51), (120, 100))
        self.assertEqual(tt.assign_teams(tracks, reads, TEAMS, ALLIANCE_OF)[4][:2], ("1111", "stitch"))

    def test_no_stitch_when_a_hidden_rival_is_near(self):
        tracks, reads = self.base()
        tracks[4] = track(4, "red", range(41, 51), (120, 100))
        tracks[5] = track(5, "red", range(34, 39), (160, 100))     # hidden at the hand-over
        reads.append(read(5, 34, "2222", per_char=-0.5))
        self.assertNotEqual(tt.assign_teams(tracks, reads, TEAMS, ALLIANCE_OF).get(4, ("",))[0], "1111")

    def test_stitch_when_the_rival_is_visibly_elsewhere(self):
        tracks, reads = self.base()
        tracks[4] = track(4, "red", range(41, 51), (120, 100))
        tracks[5] = track(5, "red", range(34, 47), (230, 100))     # seen 110 px away at frame 41
        reads.append(read(5, 34, "2222", per_char=-0.5))
        self.assertEqual(tt.assign_teams(tracks, reads, TEAMS, ALLIANCE_OF)[4][0], "1111")

    def test_fourth_robot_of_an_alliance_is_noise(self):
        tracks, reads = self.base()
        tracks[3] = track(3, "red", range(0, 101), (100, 100))
        tracks[4] = track(4, "red", range(10, 31), (500, 500))
        labels = tt.assign_teams(tracks, reads, TEAMS, ALLIANCE_OF)
        self.assertIn(4, tt.mark_noise(tracks, reads, labels, ALLIANCE_OF))

    def test_track_with_a_read_is_never_noise(self):
        tracks, reads = self.base()
        tracks[3] = track(3, "red", range(0, 101), (100, 100))
        tracks[4] = track(4, "red", range(10, 31), (500, 500))
        reads.append(read(4, 10, "1111", per_char=-0.9))
        labels = tt.assign_teams(tracks, reads, TEAMS, ALLIANCE_OF)
        self.assertNotIn(4, tt.mark_noise(tracks, reads, labels, ALLIANCE_OF))


class Power(unittest.TestCase):
    def check(self, battery):
        from unittest import mock
        with mock.patch("psutil.sensors_battery", return_value=battery):
            return tt.battery_problem()

    def test_battery_guard(self):
        # A run once ended in a Windows critical-battery shutdown.
        from collections import namedtuple
        B = namedtuple("B", "percent secsleft power_plugged")
        self.assertIsNotNone(self.check(B(10, 600, False)))   # unplugged and low: stop
        self.assertIsNone(self.check(B(10, -2, True)))        # low but charging: fine
        self.assertIsNone(self.check(B(80, 3600, False)))     # unplugged, well charged: fine
        self.assertIsNone(self.check(None))                   # desktop, no battery


class Layout(unittest.TestCase):
    def test_field_outlines(self):
        from layouts import MABIL, MILAK, MIMAR, layout_for
        self.assertFalse(MABIL.keeps(10, 10, 1920, 1080))     # above the far wall
        self.assertFalse(MABIL.keeps(60, 140, 1920, 1080))    # drive-station corner (qm18 false dot)
        self.assertFalse(MABIL.keeps(190, 66, 1920, 1080))    # fuel-counter box
        self.assertTrue(MABIL.keeps(960, 400, 1920, 1080))
        self.assertFalse(MABIL.keeps(960, 900, 1920, 1080))   # close-up cameras below the view
        self.assertTrue(MILAK.keeps(960, 820, 1920, 1080))    # side view: field in the lower band
        self.assertFalse(MILAK.keeps(960, 400, 1920, 1080))   # crowd in the stands
        self.assertTrue(MIMAR.keeps(960, 600, 1920, 1080))
        self.assertFalse(MIMAR.keeps(960, 900, 1920, 1080))   # officials' tables below the field
        self.assertIs(layout_for("2026milak_qm73"), MILAK)
        with self.assertRaises(SystemExit):
            layout_for("2026xxxx_qm1")                         # unknown event: fail loudly

    def test_labels_do_not_overlap(self):
        img = np.zeros((600, 800, 3), np.uint8)
        placed = []
        tt.draw_tag(img, 400, 300, "10063", (0, 0, 255), False, [], placed)
        tt.draw_tag(img, 410, 315, "1768", (255, 0, 0), False, [], placed)
        (a0, b0, a1, b1), (c0, d0, c1, d1) = placed
        self.assertFalse(a0 < c1 and c0 < a1 and b0 < d1 and d0 < b1)


if __name__ == "__main__":
    unittest.main()
