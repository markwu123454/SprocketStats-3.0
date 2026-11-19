# Unconfirmed ideas

**Nothing in this file is a plan.** An AI wrote almost all of it, and the owner has
not approved any of it. Don't build toward it, cite it as the design, or list it as
a next step or "not built yet" without asking first. If the owner approves an item,
move it into the README it belongs to and delete it here.

The text is quoted as it stood in the READMEs before the 2026-11-19 review
(`PLAN_REVIEW.md`).

---

## Marked "idea only" in the review

**R02. No per-match manual input in Program A** (root README §1)

> Hard constraint: **no per-match manual calibration or data entry.** Per-*year*
> game knowledge and configuration authored by a human is expected and fine.

Review evidence: no trace of the owner saying it. On 08-12 the owner said "we can
manually label team numbers"; on 08-15, "ideally everything here is automatic"
(about intrinsics).

**R04. Offline batch only** (root README §1)

> Both are **offline batch**. No realtime requirement. This is a significant
> freedom: it permits heavy models, bidirectional smoothing, and global
> optimization over the whole match.

Review evidence: no trace. It conflicts with the owner's 08-11 plan, where a pit
desktop runs "basic ml and ai" during the match.

---

## Not covered by the review

The audit missed these when it listed claims, so the owner never marked them. They
came out of the READMEs because nobody had approved them.

### From the root README

**§1, after the two programs.** The stage diagram that followed is dropped: it named
the game pack and multi-view fusion, both deleted in the review.

> They converge on the same middle — *pixels → robots localized on the field plane
> in match-clock time → events → metrics*. The difference is only how camera pose
> and time base are obtained.

**§2. Place vs shoot as a difference in distance.** This was the argument for the
place/shoot "evidence backends", which the owner replaced with "assume place now,
change next season".

> The only structural difference between the two is **the distance between where
> the robot is when it scores and where the point is registered**:
>
> - **Place**: distance ≈ 0. The robot's position *is* the evidence.
> - **Shoot**: distance 2–8m. Robot position says only that it was somewhere in a
>   large legal shooting region.

**§4, opening.**

> The foundation. Everything downstream is expressed in field coordinates, so
> registration quality is the ceiling on every other number the system produces.

**§5, what happens during a keypoint dropout.**

> **When no keypoint is available**, the track survives on the box alone;
> position is predicted from track history (and optionally the box's bottom edge
> as a weak fallback) with degraded confidence, rather than the track breaking.
>
> The payoff: **keypoint dropouts become interpolated positions with lowered
> confidence instead of identity breaks.**

The README keeps only "the track survives on the box alone rather than breaking",
which is the owner's point that boxes persist better.

**§6, identity swaps.**

> Two same-alliance robots in sustained contact will produce identity swaps;
> nothing fully prevents this.
>
> - Colour halves the problem — red-vs-blue contact is trivially resolvable, and
>   most contact is cross-alliance (defence).
> - Swaps degrade gracefully: heatmaps and alliance aggregates survive; only
>   per-robot event attribution corrupts, and only for the swapped interval.

**§7 PLACE, scoring height.**

> Level/height disambiguation requires full camera pose, not just a ground
> homography.

**§7 SHOOT, detecting pieces in flight.**

> Detection is hard — a ~35cm piece at 15m in 1080p is ~25px, motion-blurred, often
> against a crowd. Mitigations: tiled high-resolution inference over the field-and-
> above region only, triggered when a launch is suspected; synthetic motion-blur
> augmentation (the streak is a feature, not just noise); and the ballistic fit
> itself as the false-positive filter.

**§7 SHOOT, shooter years.** This leaned on dwell/transit segmentation (R47) and
possession inference (H10), both deleted in the review.

> **What breaks elsewhere in shooter years**: dwell/transit segmentation (§8)
> degrades, since ground intake happens while driving and shooting happens while
> moving — use launch events as the cycle delimiter instead of zone occupancy. And
> possession is *invisible* (the piece is inside the robot), so it must be inferred
> between pickup and launch.

### From hmm/README.md

**Minimum durations.**

> A plain HMM's segment length is geometric, with its mode at one frame, so even a
> strongly sticky self-transition leaves a trail of 1–2 frame segments. Splitting
> each state into *n* left-to-right substates sets a minimum duration of *n* frames
> and gives a negative-binomial length distribution. Model code merges the
> substates back before reporting. A full semi-Markov model (explicit durations) is
> the fallback if the chain shape fits badly.

**A fallback model form.**

> If the Gaussian states turn out to split on how the speed and turn-rate inputs
> were computed rather than on how robots move, the next form to try is an
> autoregressive HMM: each state is a linear rule for how position evolves, so it
> learns ways of moving directly from the track, with no hand-made motion inputs.

**Tracking artifacts showing up as states.**

> - Far-wall positions are noisier, so a robot stopped there looks faster than one
>   stopped near the camera. The model may split "stopped" by distance from the
>   camera.
> - Dropout frames carry constant-velocity predictions, which would form their own
>   tidy cluster.
>
> MoSeq (Datta lab) discovers mouse behaviour "syllables" from pose trajectories
> the same way, with a sticky HDP autoregressive HMM. Keypoint-MoSeq ran into the
> first problem: keypoint jitter caused spurious state switches until the
> measurement noise was modelled separately from the movement.

**Per-team states.**

> Teams that drive faster may get states of their own (a "fast transit" used mostly
> by a few teams). That is a finding about those teams, and how much each team uses
> each state is itself a per-team statistic.

**Output format.**

> `data/segments/<match_key>.json`, per team: a segment list
> `[{t0, t1, state, name, p_mean}]` in match-clock seconds, per-frame posteriors,
> and the parameter file it was decoded with.

**Tests.**

> - **Engine.** Brute-force enumeration of all Kᵀ paths for T ≤ 8, K ≤ 3 gives
>   exact posteriors and the exact best path. Tests cover masked frames, pinned
>   terminal states and time-varying transitions.
> - **Model.** Sample synthetic tracks from a known model with, say, five states
>   and check that the VB fit returns five and recovers their parameters.

The draft's planned-files table, state cards and per-season frozen fit are not
carried over: they rested on H09, which the review deleted.
