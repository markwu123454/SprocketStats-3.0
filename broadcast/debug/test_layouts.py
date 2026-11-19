"""Tests for the per-event broadcast layouts. Each case is a dot seen on a real match.

Run from the repo root:
    .venv\\Scripts\\python -m unittest broadcast.debug.test_layouts
"""

import unittest


class Layout(unittest.TestCase):
    def test_field_outlines(self):
        from broadcast.run.layouts import MABIL, MILAK, MIMAR, layout_for
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


if __name__ == "__main__":
    unittest.main()
