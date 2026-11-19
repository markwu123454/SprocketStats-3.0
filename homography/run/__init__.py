"""Per-match registration steps, in order:

1. detect_tags   -- AprilTags per view (data/detections/<stem>_tags.json)
2. solve_pose    -- intrinsics + camera pose per view (data/detections/<stem>_poses.json)

The views come from broadcast/run/split_views.py, which has to run first: finding
the camera feeds inside a broadcast frame is a property of the broadcast
production, not of registration. Known gap: detect_tags still requires a views
profile, even for single-camera footage that has nothing to split.

calibrate_ui is the manual fallback for step 2 on views with too few tags to fit
intrinsics: its exported K/dist goes into data/calibration/<stem>_intrinsics.json
and solve_pose picks it up on the next run.
"""
