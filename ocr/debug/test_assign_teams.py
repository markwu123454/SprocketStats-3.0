"""Tests for naming tracks. Each test pins down a failure seen on real matches.

Run from the repo root:
    .venv\\Scripts\\python -m unittest ocr.debug.test_assign_teams
"""

import unittest

from mtmct.run.tracker import Track
from ocr.debug.probe_bumper_reads import match_team
from ocr.run.assign_teams import assign_teams, mark_noise

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


class MatchTeam(unittest.TestCase):
    def test_fuzzy_match(self):
        teams = ["1071", "8013", "9729", "1735", "5347", "10393"]
        self.assertEqual(match_team("534", teams), "5347")     # dropped digit
        self.assertEqual(match_team("10309", teams), None)     # two edits away
        self.assertEqual(match_team("17", teams), None)        # too short to trust


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
        labels = assign_teams(tracks, reads, TEAMS, ALLIANCE_OF)
        self.assertEqual(labels[4][:2], ("1111", "elim"))

    @unittest.expectedFailure  # known bug until solid_enough_to_eliminate() gets a real rule
    def test_elimination_skips_a_flickering_dot(self):
        # 2026milak_qm73 130 s: a camera operator's dot, 14 dots over 3.0 s, was named 3875.
        tracks, reads = self.base()
        tracks[4] = track(4, "red", list(range(60, 86, 2)) + [89], (400, 100))   # too far to stitch
        self.assertNotIn(4, assign_teams(tracks, reads, TEAMS, ALLIANCE_OF))

    def test_elimination_keeps_a_patchy_real_robot(self):
        # 2026milak_qm73 110 s: 9215 along the far wall, 31 dots over 4.9 s, named correctly.
        tracks, reads = self.base()
        tracks[4] = track(4, "red", sorted(set(range(60, 81)) | set(range(83, 108, 3)) | {108}), (400, 100))
        self.assertEqual(assign_teams(tracks, reads, TEAMS, ALLIANCE_OF)[4][:2], ("1111", "elim"))

    def test_elimination_needs_a_believable_drive(self):
        # qm18 171 s: a dot on a person behind the glass was named 5962 by elimination.
        tracks, reads = self.base()
        tracks[4] = track(4, "red", range(45, 101), (1800, 100))  # 1700 px in 0.5 s
        labels = assign_teams(tracks, reads, TEAMS, ALLIANCE_OF)
        self.assertNotIn(4, labels)

    def test_stitch_continues_a_restarted_track(self):
        # qm52 171 s: 5347's track restarted 74 px away with no reads.
        tracks, reads = self.base()
        tracks[4] = track(4, "red", range(41, 51), (120, 100))
        self.assertEqual(assign_teams(tracks, reads, TEAMS, ALLIANCE_OF)[4][:2], ("1111", "stitch"))

    def test_no_stitch_when_a_hidden_rival_is_near(self):
        tracks, reads = self.base()
        tracks[4] = track(4, "red", range(41, 51), (120, 100))
        tracks[5] = track(5, "red", range(34, 39), (160, 100))     # hidden at the hand-over
        reads.append(read(5, 34, "2222", per_char=-0.5))
        self.assertNotEqual(assign_teams(tracks, reads, TEAMS, ALLIANCE_OF).get(4, ("",))[0], "1111")

    def test_stitch_when_the_rival_is_visibly_elsewhere(self):
        tracks, reads = self.base()
        tracks[4] = track(4, "red", range(41, 51), (120, 100))
        tracks[5] = track(5, "red", range(34, 47), (230, 100))     # seen 110 px away at frame 41
        reads.append(read(5, 34, "2222", per_char=-0.5))
        self.assertEqual(assign_teams(tracks, reads, TEAMS, ALLIANCE_OF)[4][0], "1111")

    def test_fourth_robot_of_an_alliance_is_noise(self):
        tracks, reads = self.base()
        tracks[3] = track(3, "red", range(0, 101), (100, 100))
        tracks[4] = track(4, "red", range(10, 31), (500, 500))
        labels = assign_teams(tracks, reads, TEAMS, ALLIANCE_OF)
        self.assertIn(4, mark_noise(tracks, reads, labels, ALLIANCE_OF))

    def test_track_with_a_read_is_never_noise(self):
        tracks, reads = self.base()
        tracks[3] = track(3, "red", range(0, 101), (100, 100))
        tracks[4] = track(4, "red", range(10, 31), (500, 500))
        reads.append(read(4, 10, "1111", per_char=-0.9))
        labels = assign_teams(tracks, reads, TEAMS, ALLIANCE_OF)
        self.assertNotIn(4, mark_noise(tracks, reads, labels, ALLIANCE_OF))


if __name__ == "__main__":
    unittest.main()
