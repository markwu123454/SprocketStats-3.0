# Plan review: what you said vs what the AI added

**Applied 2026-11-19.** The READMEs, `IDEAS.md` and the AI's memory notes now follow
these marks. This file is kept as the record of where each claim came from.

Every design claim in the root README, `hmm/README.md`, the module READMEs and the AI's
memory notes, checked against the 1,045 messages you typed across all saved Claude Code
sessions (2026-07-12 → 2026-11-19, SprocketStats 2.0 and 3.0 folders).

**Evidence** quotes your own words where they exist. "No trace" means you never said it
in a saved session. It doesn't mean it's wrong, just that it came from an AI (or from a
chat outside Claude Code, which isn't searchable). The big README rewrite (commit
`bcf5bb3`, 2026-08-16, 114 → 582 lines) isn't in any saved session, so most of §2 and
§5–§14 has no trace.

## How to mark

Fill the **You** column with one letter:

- **K**: keep. It's your plan.
- **I**: idea only. Move it to an `IDEAS.md` labelled "unconfirmed", so no session treats it as the plan.
- **D**: delete.
- **F**: fix. The text is out of date with the code; rewrite it to match.

Rows where your own words back the claim are pre-filled **K**; change any you disagree
with. Everything else is blank for you to decide. Flags: ⚠ conflicts with something you
said or built · 🕰 stale against the current code · 🔧 describes code that exists.

When you're done, tell me and I'll apply it: rewrite the READMEs to keep only K/F items,
move I items into `IDEAS.md`, and fix the memory notes.

---

## Root README

### §1 Two programs

| #   | Claim                                                                                 | Evidence                                                                                                                                                                                                        | You |
|-----|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| R01 | Program A: official broadcast in, per-robot data out                                  | You've worked on broadcast footage throughout                                                                                                                                                                   | K   |
| R02 | Hard constraint: **no per-match manual input** in Program A                           | No trace. 08-12 you said "we can manually label team numbers"; 08-15 "ideally everything here is automatic" (about intrinsics)                                                                                  | I     |
| R03 | Program B: hand-filmed, multi-angle, unknown cameras, per-match manual calibration OK | Partly. 08-11: "we also will be capturing our own footage of the match", "we use old tablets to record, and during the match its sent to a more powerful desktop". Multi-angle and manual calibration: no trace |  K   |
| R04 | Both programs are offline batch, no realtime                                          | No trace. ⚠ Your 08-11 plan has a pit desktop running "basic ml and ai" during the match                                                                                                                       | I    |

### §2 Season archetypes

| #   | Claim                                                                       | Evidence                                                                                                                                                                                                                   | You |
|-----|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| R05 | Games alternate shooter / place; 2027 is probably a place game              | No trace                                                                                                                                                                                                                   | k    |
| R06 | One engine with two evidence backends (PLACE/SHOOT), chosen per action type | Partly. 08-12: "for shooting games we'll also train a yolo on the shooting objects, and for pick and place train whatever model thats able to detect game pieces on scoring locations". The backend architecture: no trace | if we can create the whole pipeline in 1 season, changing it for shooting the next season isn't a huge amount of work, so we can just assume placing right now and change next season instead of going out of our way and implementing a switching code that we will probably rewrite next year anyways    |
| R07 | Registration accuracy target ~20 cm                                         | No trace                                                                                                                                                                                                                   | d    |

### §3 Components and layout

| #   | Claim                                                                                       | Evidence                                                                                                                                                                                   | You |
|-----|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| R08 | Six modules + supporting code; train/run/debug; tests in debug; files named by what they do | 11-19: "i consider mtmct, ocr, yolo, hrnet, broadcast, and homography to be modules, the rest is supporting code"; "tests go into debug, and files can be renamed to just be what they do" | those are the existing ones, there will be more when we keep working on more modles that does more things   |
| R09 | uv + pyproject; `data/` synced to HF, immutable unless `--override`                         | 11-19 HF request ("a duplicate of all data from each data folder from each module"); immutability recorded as your choice                                                                  | K   |
| R10 | YOLO box is the persistent tracking anchor                                                  | 08-14: "use bbox for id and associate each robot with hrnet like that"                                                                                                                     | K   |

### §4 Registration

| #   | Claim                                                                                    | Evidence                                                                                                                                | You |
|-----|------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|-----|
| R11 | Split embedded camera views by per-pixel temporal range                                  | 🔧 Built. You asked for it (08-15: "we need a way to automatically crop them"); the method was the AI's                                 | K   |
| R12 | "Program A gets multi-view coverage for free" (use the secondary views)                  | ⚠ 08-17: "currently lets not do multi camera, only the main camera is top down enough"; 08-15: bottom views are "less important to us" | secondary view is so oblique that keypoint struggle to find the actual spot, since the more top down the easier to just place in the middle of the robot    |
| R13 | Main camera doesn't move → one pose per view per match                                   | 08-19: "ever since 2025 the broadcast camera seldom shift or cut"                                                                       | K   |
| R14 | Per-tag hybrid: corners for head-on tags, centroid otherwise                             | 🔧 Built (`solve_pose.py`)                                                                                                              | K   |
| R15 | "Intrinsics are searched as a ladder, not fitted jointly"                                | 🕰 `homography/docs/pose_calibration_research.md` #7: a joint bounded fit gated on held-out error replaced the ladder                   | F   |
| R16 | Reprojection error propagated into output confidence                                     | No trace; not built                                                                                                                     |  d   |
| R17 | Program B calibration: tags or clicking 4+ points; handheld KLT+RANSAC with re-anchoring | No trace                                                                                                                                |  d   |

### §5 Detection

| #   | Claim                                                                    | Evidence                                                                                               | You |
|-----|--------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|-----|
| R18 | "The box tracks, the keypoint locates"                                   | 08-14 (as R10); 08-17: "i think bboxes may be able to persist on robots better"                        | K   |
| R19 | HRNet: 2-channel heatmap, one channel per alliance                       | 🔧 Built; your model                                                                                   | K   |
| R20 | YOLO-pose was rejected because it needs box association                  | No trace. ⚠ A YOLO pose spotter was later trained anyway (`yolo/data/runs/version_26x_pose_polished`) | d    |
| R21 | A box's bottom edge is a bad ground point                                | No trace                                                                                               | k    |
| R22 | Keypoint→box matching by containment + alliance + predicted ground point | No trace. Your idea was pairing each box with an HRNet dot; this matching rule is the AI's             | d    |
| R23 | Alliance colour as a soft score with a balanced 3-red/3-blue assignment  | No trace                                                                                               | d    |

### §6 Tracking and identity

| #   | Claim                                                                                  | Evidence                                                                                                                        | You |
|-----|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-----|
| R24 | Track in field coordinates with a Kalman filter; Mahalanobis + IoU + colour, Hungarian | No trace; not built (the tracker is greedy, in pixels)                                                                          |  d   |
| R25 | Offline min-cost-flow stitching into exactly six slots                                 | No trace. 08-12 you wanted broken tracks joined and to "manually tell the code that these 2 tracks are actually the same robot" | d    |
| R26 | "Don't guess, break the tracklet"                                                      | 🔧 Built (`mtmct/run/split_tracks.py`); no trace of you saying it                                                               | k    |
| R27 | Team numbers: take the six teams from TBA and assign them, don't discover them         | 🔧 Built (teammate's PR #1). 08-17: "team number detection is another algorithm of its own ... ocr perhaps"                     | K   |
| R28 | Score the three candidates through the OCR's CTC lattice                               | 🔧 Built (teammate's PR #1)                                                                                                     | K   |
| R29 | Gate OCR on box size and blur; sum log-probs along the track                           | Partly built (reads are summed along each track); no trace                                                                      |  d   |
| R30 | Appearance gallery: recognise robots across an event by look                           | 08-17: "or even going across an event and looking at robot shapes"                                                              | K   |
| R31 | Everything timestamped in match-clock seconds; A reads the timer overlay               | 08-17: "audio and cg overlay processing ... determine when a match start and end, and record when an alliance's scores update"  | K   |
| R32 | Program B syncs cameras by GCC-PHAT on the horn                                        | No trace. Note: GCC-PHAT was tried on broadcast audio and lost to NCC (`detect_phases.py`)                                      | d    |
| R33 | Multi-view fusion: geometric cross-camera association, factor graph / RTS smoother     | No trace. ⚠ Same "not multi camera" conflict as R12                                                                            | d    |

### §7 Game pack

| #   | Claim                                                                                                                    | Evidence                                                                                                     | You |
|-----|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|-----|
| R34 | Per-year "game pack" plugin (YAML schema); everything else year-agnostic                                                 | No trace for the plugin. Your need (08-12): "any way we can train or code anything without knowing the game" |  d   |
| R35 | PLACE: detect pieces on scoring locations                                                                                | 08-12 (quoted in R06)                                                                                        | K   |
| R36 | PLACE details: site ROIs in field coords, median filter, change window, credit the nearby robot, "occlusion is evidence" | No trace                                                                                                     | k    |
| R37 | PLACE: **Viterbi** with the end state pinned to TBA's breakdown                                                          | No trace                                                                                                     |d    |
| R38 | PLACE metrics: level capability, alignment time                                                                          | No trace                                                                                                     | d    |
| R39 | SHOOT: YOLO on pieces in flight                                                                                          | 08-12 (quoted in R06)                                                                                        | K   |
| R40 | SHOOT details: 6-DOF ballistic arc fit, back-extrapolate for attribution, forward for make/miss, launch speed/angle      | No trace                                                                                                     | k    |
| R41 | SHOOT extra signals: goal ROI, launch streaks, score-tick matching, flywheel audio                                       | No trace (score updates from the overlay are yours, R31)                                                     | k    |
| R42 | Per-robot shooting accuracy is the most valuable stat; accuracy vs distance                                              | No trace                                                                                                     | k    |
| R43 | Heading is low value; read aim from the projectile                                                                       | No trace                                                                                                     | k    |
| R44 | Endgame: a climb shows up as a ground-plane violation near a climb structure                                             | No trace                                                                                                     |  d   |

### §8 Analytics

| #   | Claim                                                                                      | Evidence                                                                                                            | You |
|-----|--------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|-----|
| R45 | Analytics on positions only, no game knowledge, so it works before a new game is supported | 08-12 (quoted in R34). This is your motivation; the methods below are not                                           | K   |
| R46 | Smoothing by Savitzky-Golay or an RTS smoother                                             | You asked for smoothing (08-12: "the paths are jiggery ... smooth"); the methods: no trace                          | d    |
| R47 | 3-state HMM: dwell / transit / idle                                                        | No trace. ⚠ 11-19 you wanted the model to "learn states by itself"                                                 | d    |
| R48 | Discover field zones by clustering where robots stop                                       | No trace                                                                                                            |  d  |
| R49 | Cycles as round trips; report median/IQR plus a trend                                      | You listed cycle rate and cycle count as outputs (08-12); the method: no trace                                      | k    |
| R50 | Cycle decomposition + path efficiency                                                      | No trace                                                                                                            | k    |
| R51 | Drivetrain type from speed vs turn rate                                                    | No trace                                                                                                            | d   |
| R52 | Auto routines by trajectory clustering (DTW / Fréchet)                                     | You listed "auto path" as an output (08-12); the method: no trace                                                   | k    |
| R53 | Defence impact by regression, in seconds added per cycle                                   | defense/defended were in your frame-classification classes (07-27); the regression: no trace                        | k    |
| R54 | Change-point detection for "died at t=94s"                                                 | No trace                                                                                                            | d    |
| R55 | Heatmaps and alliance compatibility by overlap                                             | No trace                                                                                                            |  k   |
| R56 | "Skip OPR"                                                                                 | ⚠ You built `match_prediction/` with `OPRModel` and `ComponentOPRModel`                                            | d    |
| R57 | `action/` is a supervised per-frame classifier                                             | 07-27: "using rnn or some adjacent algorithm to do a frame classification of what each robot is doing"; 08-12 again | K   |
| R58 | HMM and zone clustering act as pre-labelers for `action/`                                  | No trace                                                                                                            | d    |

### §9–§14
del all except that one
| #   | Claim                                                                                                            | Evidence                                                                                                                                  | You |
|-----|------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|-----|
| R59 | 11-stage order incl. match-ID OCR, replay rejection via the timer, review UI                                     | No trace                                                                                                                                  |     |
| R60 | TBA-constrained reconciliation (integer program / Viterbi to hit known totals)                                   | No trace. 08-12 you said TBA "doesnt record per robot ... the data is there for vereification but its not detailed enough"                |     |
| R61 | Output: Parquet tracks at 20–30 Hz, events, per-robot metrics                                                    | No trace. Note: storage lives in another repo (Postgres + S3)                                                                             |     |
| R62 | Confidence on every output value                                                                                 | Partly. 08-13: a binary prediction carries "what is predicted, and at what confidence/probability", a score "comes with a mean and stdev" |     |
| R63 | Split by season or match, never by frame                                                                         | 🔧 Built into hrnet/yolo/action training                                                                                                  | K   |
| R64 | Validation: TBA totals P/R, hand-labelled matches, regress vs human scouting, B as ground truth for A            | No trace                                                                                                                                  |     |
| R65 | Synthetic registration training data from field CAD in Blender                                                   | No trace (registration uses AprilTags, not a trained model)                                                                               |     |
| R66 | Training data: label one frame per shot and propagate; synthetic motion blur; self-training on reconciled output | No trace (the Label Studio active-learning loop is yours and built)                                                                       |     |
| R67 | Filming protocol: two corner tripods, 4K30, a tag always visible, no zoom                                        | No trace. Your setup (08-11) is "old tablets to record"                                                                                   |     |
| R68 | Non-goals: mechanism pose, ground pieces, realtime, manual input in A, segmentation                              | No trace                                                                                                                                  |     |

---

## hmm/README.md

| #   | Claim                                                                                                               | Evidence                                                                                                                                   | You |
|-----|---------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|-----|
| H01 | An `hmm/` module                                                                                                    | 11-19: "add a hmm folder and discuss how should hmm be implemented"                                                                        | K   |
| H02 | The model learns its own states; the data decides how many                                                          | 11-19: "i want the model to learn states by itself(so it might decide theres some other state thats more important or seen more)"          | K   |
| H03 | hmm/ also does site-state decoding and count-constrained (TBA) decoding                                             | No trace. Inherited from README R37/R47/R60 ("The root README asks for an HMM in three places")                                            |  d   |
| H04 | Forward-backward + Viterbi engine                                                                                   | Standard for any HMM, so it comes with H01                                                                                                 | K   |
| H05 | Write the engine ourselves instead of hmmlearn (missing frames, confidence weights, pins, time-varying transitions) | No trace                                                                                                                                   |   k  |
| H06 | Inputs: log speed, acceleration, turn rate, position, heading, distances to others                                  | No trace                                                                                                                                   |  d   |
| H07 | Red mirrored into blue's frame; auto fitted separately; phases as masks                                             | No trace                                                                                                                                   |   k  |
| H08 | Variational-Bayes sticky HDP-HMM (~20 states max); check states survive refits                                      | No trace. This is the AI's way of doing H02                                                                                                |  k   |
| H09 | Fit per season, pooled across events, frozen; "state cards" with clips for you to name                              | No trace                                                                                                                                   |  d   |
| H10 | Possession (loaded/empty) as a second hidden variable, anchored by score ticks                                      | You asked (11-19) "can't it infer driving loaded and empty purely based on beahvior"; the README's answer (no, add a variable) is the AI's |  d   |
| H11 | Later: HMM over `action/`'s GRU outputs (NN-HMM hybrid)                                                             | No trace                                                                                                                                   |  d   |

---

## Module READMEs

| #   | Claim                                                                                                                              | Evidence                                                                                                              | You |
|-----|------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|-----|
| B01 | broadcast "Design notes": buzzer is "a short burst of a specific frequency"; crop region calibrated per event; aggregate ±3 frames | 🕰 Written as a stub before the code; the pipeline now finds overlay regions automatically and matches FMS audio cues | F   |
| B02 | broadcast: "penalty events", score jumps with no scoring action, as a foul signal                                                  | No trace; not built                                                                                                   |  d   |
| HR1 | hrnet: season-agnostic, "train on 2022–2026, deploy 2027+"                                                                         | 08-16: you wanted "a large dataset for all recent years so the models can generalize" plus "a small dataset for 2026" | K   |
| HR2 | hrnet: "Why not YOLO-pose"                                                                                                         | Same as R20                                                                                                           | d    |

The mtmct, ocr and action READMEs describe the code as it is. Nothing to review there.

---

## The AI's memory notes

Not in the repo; these are what future sessions read automatically.

| #   | Note: claim                                                                                 | Evidence                                                                                                                                  | You |
|-----|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|-----|
| M01 | pipeline plan: goals are match prediction (ensemble) and robot classification (3–4 metrics) | 08-12: "right now our goal is to do 2 things: match prediction, and robot classification"                                                 | K   |
| M02 | pipeline plan: "HRNet models are year-specific"                                             | ⚠ Misfiled. You said "year specific" about the state-classification model (08-12), and wanted detection datasets that generalize (08-16) |     |
| M03 | pipeline plan: 4-point homography, EMA smoothing, manual team labelling                     | 🕰 Now AprilTag PnP, OCR naming                                                                                                           | F   |
| M04 | pipeline plan: "what position data alone can derive" list                                   | No trace. The AI's answer to your "so what do we get"                                                                                     |     |
| M05 | pipeline plan: "State classification decision: don't need frame-level labels"               | ⚠ You proposed frame-level classification (07-27, 08-12) and built `action/`                                                             |     |
| M06 | pipeline plan: timeline (pull data after week 1, train shot/piece detectors)                | 08-12                                                                                                                                     | K   |

The other notes (broadcast, homography, tracking, HF sync, module layout, storage) record
measured results or choices you made explicitly. I spot-checked them and found no
invented plans.
