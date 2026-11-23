# HMM — learning robot states from tracks

Design notes only; nothing here is implemented yet. Kept to what the owner
approved in the 2026-11-19 review (`PLAN_REVIEW.md`); the rest of the earlier
draft is in the root `IDEAS.md`, labelled unconfirmed.

Asked for on 2026-11-19: a model that learns robot states by itself, so it
might decide there's some other state that's more important or seen more.

The module is one small engine (forward-backward and Viterbi) plus the model
that uses it. The engine knows nothing about robots; the model builds the
arrays the engine consumes.

## Why write the engine instead of using hmmlearn

The constraints this needs are the parts libraries don't support:

- **Missing observations.** Keypoint dropouts are routine (root §5). A dropped
  frame should contribute nothing to the likelihood while time and transitions
  keep running. hmmlearn rejects NaN input.
- **Confidence-weighted frames.** A tracker can report a position during a
  dropout, predicted from track history (mtmct doesn't today). Fed in at full
  weight, a robot that stopped behind another robot would look like it kept
  driving. Each frame's log-likelihood is multiplied by its confidence, so
  predicted positions barely count.
- **Pinned and clamped states.** A boolean mask of allowed states per frame.
  The match-phase masks below use it.
- **Time-varying transitions.** Behaviour differs between auto, teleop and
  endgame, and the 2026 phases file also has `shift_changes` every 25 s.
  Robots are disabled in the auto→teleop gap, and stopping there is not idling.

pomegranate was rewritten on torch for 1.0 with an incompatible API, and dynamax
needs JAX, which has no CUDA build for Windows. A match is about 1,600 frames at
the spotter's 10 fps with at most a few dozen states, so log-space numpy runs in
milliseconds per sequence, and numpy and `scipy.special.logsumexp` are already
in the environment. Writing it ourselves adds no dependency and removes no
capability we need.

### Planned engine API (`run/core.py`)

```python
# All arrays log-space. K = number of states, T = frames.
log_init:  (K,)
log_trans: (K, K) or (T-1, K, K)        # time-varying allowed
log_emit:  (T, K)                        # 0 on missing frames; already conf-weighted
allowed:   (T, K) bool | None           # pins, clamps, phase masks

forward_backward(...) -> posterior (T, K), expected transition counts (K, K), loglik
viterbi(...)          -> path (T,), log_prob
```

The variational Bayes fit below calls the same two functions; VB only changes
what the M-step hands back (expected log-parameters instead of point
estimates).

## Behaviour discovery

The states come from the data. We choose the inputs and the general form of
the model; the model chooses how many states there are and what each one is,
and it can find states nobody listed in advance, such as robots queueing
behind each other at a loading station.

**Needs field-space tracks.** mtmct's tracks are in image pixels from a single
view. In pixels a robot crossing the far wall moves a fraction of what the same
robot moves at the near wall, so states learned in pixels would split robots by
distance from the camera and would have to be thrown away. The engine and its
tests don't need field-space tracks and can be built first.

### Alliance and match phase

Position goes in alliance-relative: red robots are mapped into blue's frame
using the field's symmetry. Otherwise "at its own loading station" would be
found twice, once per alliance, with half the data each.

Match phase is a condition, not an input. The auto→teleop gap and time outside
the live match are masked out. Auto is fitted as its own model, because its
paths are scripted and root §8 (item 3) already clusters them.

### How many states

Each state is a full-covariance Gaussian over the standardized inputs, with
sticky transitions. The fit is variational Bayes with room for about 20
states and a prior that favours using few of them. States the data doesn't
need end up with almost no frames and are dropped, and the number left is the
model's answer. This is a finite approximation of the sticky HDP-HMM (Fox et
al., 2011), which was designed for discovering an unknown number of states in
a sequence, and it needs nothing beyond the engine.

That number still depends on how strongly the prior favours few states, so it
gets checked: refit with several prior strengths, several random starts, and
on different events. A state that appears in every fit is a real pattern. A
state that appears in some fits and not others is the model splitting or
merging something arbitrarily, and it shouldn't be given a name.

Stickiness needs a prior too. Without one, the cheapest way for the model to
explain noise is to switch state every few frames. A self-transition
probability around 0.95 or more at 10 fps gives states lasting a couple of
seconds on average.

## Open questions

- Which inputs? The earlier draft's list (speed, acceleration, turn rate,
  position, heading, distances to other robots) was dropped in the review.
- Is `hmm/` a pipeline module (train/run/debug, imported through `hmm.run`) or
  supporting code like `better_sampling/`? It is laid out as a module because
  other code will import the engine.
- Who builds the field projection of tracks, and when? Behaviour discovery
  waits on it; the engine doesn't.
