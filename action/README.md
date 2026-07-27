# Per-robot action classification

Frame classification of what each robot is doing (traveling / intake /
scoring / defense / defended / idle / ...), given finished tracks from
`../mtmct/`. No image/vision dependency — only consumes
`(x_frac, y_frac, label)` sequences per robot per frame.

- `01_train.py` — dataset (ego + relational features), GRU model, training loop.
- `02_eval.py` — frame-level P/R/F1 + confusion matrix, and segment-level F1@IoU.

See `01_train.py`'s module docstring for the assumed manifest format — the
adapter boundary to swap once `../mtmct/`'s real output format lands.
