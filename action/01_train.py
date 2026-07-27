#!/usr/bin/env python3
"""
Step 1 — Train the per-robot action-classification model.

Input: per-robot track sequences (positions over time, at 10-30fps) coming
out of the MTMCT tracker (../mtmct/). This file does NOT do tracking or
re-identification — it consumes finished tracks and classifies, per
timestep, what each robot is doing (traveling / intake / scoring / defense
/ defended / idle / ...). It has no dependency on HRNet or any vision
backbone, which is why it lives in its own folder/config/checkpoints.

Design:
  - Ego-centric, not a fixed 6-robot slot vector. A fixed slot layout (e.g.
    "blue_1, blue_2, ..., red_3") breaks the moment a robot is occluded or a
    track ID gets reassigned, and forces the model to learn an arbitrary
    identity ordering. Instead each robot's input is its OWN trajectory plus
    relational features to the other 5, sorted by distance (closest-K
    teammates, closest-K opponents) with a present/absent mask. That's still
    "the model sees all 6 robots' positions" — "defended" is legible because
    a nearby opponent lands in the opp_1 slot with small distance — but it's
    permutation-invariant and robust to a missing robot.
  - One shared-weight GRU is applied per-robot. Batches are therefore
    (B*6, T, F), not some exotic joint-multi-agent architecture. Simpler to
    train on limited labeled data, and scales to any number of robots on
    field (doesn't hardcode 3v3).
  - Bidirectional GRU: this is offline post-match analysis (stats, like the
    rest of the pipeline), not a live overlay, so future frames are free
    signal — a robot only classifies as "defended" once you can see the
    opponent arrive AND leave.
  - Many-to-many sequence labeling: loss is per-timestep cross-entropy over
    a window, with an ignore_index for frames that fall outside any labeled
    segment (human annotators label segments, not every single frame).
  - Match-level holdout, same reasoning as HRNet-W32/02_train.py: frames
    next to each other in time are near-duplicates, so a random frame split
    leaks. Held out whole matches instead.

MANIFEST FORMAT (placeholder — this is the one function to change,
`load_matches()`, once the real MTMCT output format is finalized):

  One JSON object per line, one line per match:
  {
    "match": "2026arc_qm12",
    "fps": 20,
    "tracks": {"blue_1": "blue", "blue_2": "blue", "blue_3": "blue",
               "red_1": "red", "red_2": "red", "red_3": "red"},
    "frames": [
      {
        "t": 0,
        "robots": {
          "blue_1": {"x_frac": 0.31, "y_frac": 0.52, "label": "traveling"},
          "blue_2": {"x_frac": 0.40, "y_frac": 0.61, "label": "defense"},
          "red_1":  {"x_frac": 0.44, "y_frac": 0.60, "label": "defended"}
          # a track missing from "robots" this frame == occluded/off-field
        }
      },
      ...
    ]
  }

  x_frac/y_frac are field-fraction coords (same convention as the
  HRNet-W32 06/07 field-calibration tools), NOT image pixels — this is what
  makes the model camera- and match-independent. "label" is null/absent for
  frames nobody has hand-labeled yet; those are excluded from the loss via
  ignore_index, not counted as a real class.
"""
import json, math, pathlib, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CFG_PATH = pathlib.Path(__file__).parent / "config.yaml"
import yaml
CFG = yaml.safe_load(open(CFG_PATH))
A = CFG["action"]; P = CFG["paths"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = pathlib.Path(__file__).parent
def _p(path_str):
    p = pathlib.Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)

CLASSES = A["classes"]
CLS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
IGNORE_LABEL = A["ignore_label"]
IGNORE_ID = -100  # torch's CrossEntropyLoss default ignore_index

K_TEAM = A["relational_k_teammates"]
K_OPP = A["relational_k_opponents"]
# ego: x, y, vx, vy, self_alliance(0/1) = 5
# relational: (K_TEAM + K_OPP) slots x (dx, dy, dist, present) = 4 each
IN_DIM = 5 + (K_TEAM + K_OPP) * 4
MISSING_DIST = 1.5  # > max possible field-frac distance (sqrt(2)); flags "not present"


# ----------------------------------------------------------------------
# Manifest loading -> per-track feature/label sequences
# ----------------------------------------------------------------------
def load_matches(manifest_path):
    matches = []
    for line in open(manifest_path):
        line = line.strip()
        if line:
            matches.append(json.loads(line))
    return matches


def build_track_sequences(match):
    """One match -> list of (track_id, alliance, feats[T,IN_DIM], labels[T])."""
    fps = match["fps"]
    dt = 1.0 / fps
    frames = match["frames"]
    track_ids = list(match["tracks"].keys())
    alliance_of = match["tracks"]
    T = len(frames)

    # raw per-track position (nan where absent) and label id
    pos = {tid: np.full((T, 2), np.nan, dtype=np.float32) for tid in track_ids}
    lab = {tid: np.full((T,), IGNORE_ID, dtype=np.int64) for tid in track_ids}
    for t, fr in enumerate(frames):
        for tid, r in fr.get("robots", {}).items():
            if tid not in pos:
                continue
            pos[tid][t] = (r["x_frac"], r["y_frac"])
            lbl = r.get("label")
            if lbl and lbl != IGNORE_LABEL:
                lab[tid][t] = CLS_TO_ID[lbl]

    # velocity via finite differences on present frames only (nan-safe)
    vel = {}
    for tid in track_ids:
        p = pos[tid]
        v = np.zeros_like(p)
        present = ~np.isnan(p[:, 0])
        for t in range(1, T):
            if present[t] and present[t - 1]:
                v[t] = (p[t] - p[t - 1]) / dt
        vel[tid] = v

    out = []
    for tid in track_ids:
        alliance = alliance_of[tid]
        feats = np.zeros((T, IN_DIM), dtype=np.float32)
        p_self, v_self = pos[tid], vel[tid]
        for t in range(T):
            if np.isnan(p_self[t, 0]):
                continue  # this robot absent this frame -> leave as zero + ignore label
            x, y = p_self[t]
            vx, vy = v_self[t]
            feats[t, 0:5] = (x, y, vx, vy, 0.0 if alliance == "blue" else 1.0)

            # gather other robots present at t, split teammate/opponent, sort by dist
            teammates, opponents = [], []
            for other in track_ids:
                if other == tid or np.isnan(pos[other][t, 0]):
                    continue
                dx, dy = pos[other][t, 0] - x, pos[other][t, 1] - y
                dist = math.hypot(dx, dy)
                bucket = teammates if alliance_of[other] == alliance else opponents
                bucket.append((dist, dx, dy))
            teammates.sort(key=lambda z: z[0])
            opponents.sort(key=lambda z: z[0])

            off = 5
            for k in range(K_TEAM):
                if k < len(teammates):
                    dist, dx, dy = teammates[k]
                    feats[t, off:off + 4] = (dx, dy, dist, 1.0)
                else:
                    feats[t, off:off + 4] = (0.0, 0.0, MISSING_DIST, 0.0)
                off += 4
            for k in range(K_OPP):
                if k < len(opponents):
                    dist, dx, dy = opponents[k]
                    feats[t, off:off + 4] = (dx, dy, dist, 1.0)
                else:
                    feats[t, off:off + 4] = (0.0, 0.0, MISSING_DIST, 0.0)
                off += 4

        out.append((tid, alliance, feats, lab[tid]))
    return out


# ----------------------------------------------------------------------
# Windowed dataset (many-to-many: predict a label per frame in the window)
# ----------------------------------------------------------------------
class ActionWindowDataset(Dataset):
    def __init__(self, matches, fps, window_sec, stride_sec):
        self.window = int(round(window_sec * fps))
        stride = max(1, int(round(stride_sec * fps)))
        self.samples = []  # (feats[W,IN_DIM], labels[W])
        for match in matches:
            for tid, alliance, feats, labels in build_track_sequences(match):
                T = feats.shape[0]
                if T < self.window:
                    continue
                for start in range(0, T - self.window + 1, stride):
                    end = start + self.window
                    lbl_win = labels[start:end]
                    if (lbl_win != IGNORE_ID).sum() == 0:
                        continue  # nothing labeled in this window, skip
                    self.samples.append((feats[start:end], lbl_win))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, labels = self.samples[idx]
        return torch.from_numpy(feats), torch.from_numpy(labels)


def match_split(matches, val_frac=0.15, seed=42):
    """Hold out whole matches for validation (same reasoning as HRNet-W32/02_train.py)."""
    m = list(matches)
    rng = random.Random(seed)
    rng.shuffle(m)
    if len(m) < 2:
        print("[warn] only one match present; validation will be optimistic.")
        return m, m
    n_val = max(1, int(len(m) * val_frac))
    return m[n_val:], m[:n_val]


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class ActionGRU(nn.Module):
    def __init__(self, in_dim, hidden, layers, num_classes, bidirectional, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.LayerNorm(hidden))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=bidirectional,
                          dropout=dropout if layers > 1 else 0.0)
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes))

    def forward(self, x):
        h = self.encoder(x)
        out, _ = self.gru(h)
        return self.head(out)  # (B, T, num_classes) logits


# ----------------------------------------------------------------------
# Class weights (traveling/idle will dominate; upweight the rare actions)
# ----------------------------------------------------------------------
def compute_class_weights(dataset):
    counts = np.zeros(len(CLASSES), dtype=np.float64)
    for _, labels in dataset.samples:
        valid = labels[labels != IGNORE_ID]
        if len(valid):
            counts += np.bincount(valid, minlength=len(CLASSES))
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def main():
    manifest = _p(P["manifest"])
    matches = load_matches(manifest)
    torch.manual_seed(A["seed"]); np.random.seed(A["seed"])

    train_matches, val_matches = match_split(matches, A["val_frac"], A["seed"])
    tr = ActionWindowDataset(train_matches, A["fps"], A["window_sec"], A["stride_sec"])
    va = ActionWindowDataset(val_matches, A["fps"], A["eval_window_sec"], A["eval_window_sec"])
    print(f"[data] {len(matches)} matches -> train windows={len(tr)} val windows={len(va)}")

    MIN_TRAIN = 50
    if len(tr) < MIN_TRAIN:
        raise SystemExit(
            f"[abort] not enough labeled action data: {len(tr)} windows (need >={MIN_TRAIN}). "
            f"Label more action segments first.")

    tl = DataLoader(tr, batch_size=A["batch_size"], shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=A["batch_size"], shuffle=False)

    model = ActionGRU(IN_DIM, A["hidden_size"], A["num_layers"], len(CLASSES),
                      A["bidirectional"], A["dropout"]).to(DEVICE)
    class_weights = compute_class_weights(tr).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_ID)
    opt = torch.optim.AdamW(model.parameters(), lr=A["base_lr"], weight_decay=A["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, A["epochs"])

    ckpt_dir = _p(P["checkpoints"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(A["epochs"]):
        model.train()
        running = 0.0
        for feats, labels in tl:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            logits = model(feats)  # (B, T, C)
            loss = criterion(logits.reshape(-1, len(CLASSES)), labels.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()

        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for feats, labels in vl:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                logits = model(feats)
                val_loss += criterion(logits.reshape(-1, len(CLASSES)), labels.reshape(-1)).item()
                n += 1
        val_loss /= max(1, n)
        print(f"[epoch {epoch}] train_loss={running/len(tl):.4f} val_loss={val_loss:.4f}")

        torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": CFG},
                   ckpt_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": CFG},
                       ckpt_dir / "best.pt")
            print(f"  * new best val_loss={best_val_loss:.4f}")

    print(f"[done] best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
