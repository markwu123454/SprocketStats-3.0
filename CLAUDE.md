# Notes for AI assistants working in this repo

## Don't turn your interpretation into the plan

Docs here have mixed the owner's decisions with AI elaboration, and later sessions
treated the elaboration as the plan (for example, a Viterbi / TBA-reconciliation design
the owner never asked for). The 2026-11-19 review (`PLAN_REVIEW.md`) cut the READMEs
down to what the owner approved; unapproved ideas are in `IDEAS.md`.

- Write down only what the user said or explicitly approved, whether in a README,
  docstring, comment or memory note. If you add an idea of your own, label it inline as an
  unconfirmed AI suggestion, or leave it out. Never write it in the design's voice.
- Don't treat design text in `README.md`, the module READMEs or any ideas file as an
  approved plan. Before building toward it, or listing it as a next step or as
  "not built yet", ask.
- Measured facts are fine to record: test results, what was tried and how it failed,
  what the code does today. This rule is about plans, priorities and design choices.

## Comments and docstrings

Record why something is the way it is and its known limitations (what was tried, what
still fails), not a restatement of what the code does.
