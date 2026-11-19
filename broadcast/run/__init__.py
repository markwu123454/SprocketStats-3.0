"""Per-match broadcast steps, in order:

1. detect_phases    -- match phase boundaries from audio cues (data/<match>_phases.json)
2. split_views      -- the camera feeds embedded in the frame (data/<match>_views.json)
3. detect_overlay   -- overlay regions that change during the match (data/<match>_regions.json)
4. read_overlay     -- every region read on every settled frame (data/<match>_regions_timeline.json)
5. identify_fields  -- what each region means, cut to the match window (data/<match>_match.json)

split_views and detect_overlay are the two halves of one decomposition of the
frame by per-pixel temporal range: live camera regions vs the static scoreboard
around them. Neither depends on the other, and detect_phases is independent of
both; identify_fields uses phases as a cross-check when they exist.
homography/run/detect_tags.py reads split_views' output.

read_overlay needs the digit bank that train/harvest_digits.py +
train/build_digit_templates.py build.
"""
