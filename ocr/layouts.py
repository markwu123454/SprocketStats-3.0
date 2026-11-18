"""Broadcast layouts per event: where the field is on screen and which overlays to ignore.

Each event's video crew places cameras differently, so these are measured per event from a
frame of its broadcast. All coordinates are fractions of frame width/height (0..1).

    field      polygon around the field carpet; robot dots outside it are ignored
    overlays   boxes holding printed digits/team numbers (score bar, captions): greyed out
               before OCR and kept clear of labels
    drop       boxes where spotter dots are ignored even inside the field
    view       (top, bottom) band of the frame drawn in the labelled video
"""

from dataclasses import dataclass, field as _field

import cv2
import numpy as np


@dataclass(frozen=True)
class Layout:
    name: str
    field: tuple
    overlays: tuple = ()
    drop: tuple = ()
    view: tuple = (0.0, 1.0)
    notes: str = ""

    def _poly(self, w, h):
        return np.array([(x * w, y * h) for x, y in self.field], np.float32)

    def in_field(self, x, y, w, h) -> bool:
        return cv2.pointPolygonTest(self._poly(w, h), (float(x), float(y)), False) >= 0

    def in_drop(self, x, y, w, h) -> bool:
        return any(x0 * w <= x <= x1 * w and y0 * h <= y <= y1 * h for x0, y0, x1, y1 in self.drop)

    def keeps(self, x, y, w, h) -> bool:
        """Should a spotter dot at (x, y) be treated as a possible robot?"""
        return self.in_field(x, y, w, h) and not self.in_drop(x, y, w, h)

    def overlay_px(self, w, h):
        return [(int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)) for x0, y0, x1, y1 in self.overlays]

    def view_px(self, h):
        return int(self.view[0] * h), int(self.view[1] * h)


# 2026 FIRST overlay, NE District broadcast (measured on 2026mabil): an overhead camera fills
# the top 65% of the frame, with close-up cameras below it. Score bar at the top centre,
# fuel counters and arrows in the top corners. Off-field: above the far wall and the
# drive-station corners, where the spotter put dots on people behind the glass.
_NE_VIEW = 700 / 1080  # kept exact so 2026mabil results stay identical to earlier runs
MABIL = Layout(
    name="ne-overhead",
    field=((0.05, 0.04), (0.95, 0.04), (0.95, 0.25), (1.0, 0.25), (1.0, _NE_VIEW),
           (0.0, _NE_VIEW), (0.0, 0.25), (0.05, 0.25)),
    overlays=((0.22, 0.0, 0.78, 0.16), (0.0, 0.05, 0.19, 0.115), (0.81, 0.05, 1.0, 0.115)),
    drop=((0.0, 0.05, 0.19, 0.115), (0.81, 0.05, 1.0, 0.115)),
    view=(0.0, _NE_VIEW),
    notes="overhead view; robots ~100 px; bumper digits 8-12 px",
)

# FiM (Michigan) broadcasts: one wide side camera from the stands, full-width score bar over
# the top 24%, event caption at the bottom centre. The field is a trapezoid (perspective).
_FIM_OVERLAYS = ((0.0, 0.0, 1.0, 0.24), (0.32, 0.93, 0.68, 0.995))

MILAK = Layout(  # 2026milak (Lakeview): low camera, field in the lower third
    name="fim-lakeview",
    field=((0.155, 0.62), (0.84, 0.62), (0.995, 0.86), (0.995, 0.905), (0.005, 0.905), (0.005, 0.86)),
    overlays=_FIM_OVERLAYS,
    view=(0.0, 0.93),
    notes="side view; robots ~60-90 px",
)

MIMAR = Layout(  # 2026mimar (Marysville): higher camera, field in the middle band
    name="fim-marysville",
    field=((0.19, 0.455), (0.85, 0.455), (0.975, 0.72), (0.02, 0.68)),
    overlays=_FIM_OVERLAYS,
    view=(0.0, 0.76),
    notes="side view; robots ~60-90 px",
)

LAYOUTS = {"2026mabil": MABIL, "2026milak": MILAK, "2026mimar": MIMAR}


def layout_for(match_key: str) -> Layout:
    event = match_key.split("_")[0]
    if event not in LAYOUTS:
        raise SystemExit(f"No broadcast layout for event {event!r}. Measure one from a frame of its "
                         f"video and add it to ocr/layouts.py (see the existing entries).")
    return LAYOUTS[event]
