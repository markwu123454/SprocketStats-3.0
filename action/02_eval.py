#!/usr/bin/env python3
"""
Step 2 — Measure the action-classification model.

Loads checkpoints/best.pt, runs it on held-out (validation-match) tracks,
and reports two levels of metric:

  - Frame-level: per-class precision/recall/F1 + confusion matrix. Standard,
    but a bit misleading here — most frames are "traveling", so it rewards
    the model for the boring class and can hide it missing short events.
  - Segment-level ("segmental F1@IoU", the standard action-segmentation
    metric): group each contiguous run of one label into a segment, and
    count a predicted segment as a match if it's the same class and
    overlaps a not-yet-matched GT segment with IoU >= threshold. This is
    what actually answers "did it catch that this robot scored at 1:32-1:34"
    rather than "did it get most frames right."

Inference stitches the track in overlapping windows (window_sec /
eval_overlap_sec from config) and averages softmax probabilities in the
overlap region, since the model was trained many-to-many on fixed windows
but tracks run far longer than one window.

Module name starts with a digit, so 01_train can't be `import`ed
normally — load it by path instead.
"""
import argparse, importlib.util, pathlib
import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).parent

def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "action_train", PROJECT_ROOT / "01_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

T = _load_train_module()

CFG = T.CFG; A = T.A; P = T.P
CLASSES = T.CLASSES
IGNORE_ID = T.IGNORE_ID
DEVICE = T.DEVICE


def _p(path_str):
    p = pathlib.Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_model(ckpt_name="best.pt"):
    ckpt = torch.load(_p(P["checkpoints"]) / ckpt_name, map_location=DEVICE)
    model = T.ActionGRU(T.IN_DIM, A["hidden_size"], A["num_layers"], len(CLASSES),
                        A["bidirectional"], A["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[model] loaded epoch {ckpt['epoch']} from {ckpt_name}")
    return model


@torch.no_grad()
def predict_track(model, feats, window_sec, overlap_sec, fps):
    """feats: (T, IN_DIM) -> predicted class ids (T,), via overlap-averaged
    softmax over sliding windows. Short tracks (< 1 window) run in one shot."""
    Tlen = feats.shape[0]
    window = int(round(window_sec * fps))
    if Tlen <= window:
        x = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
        probs = torch.softmax(model(x), dim=-1)[0].cpu().numpy()
        return probs.argmax(axis=-1)

    step = max(1, window - int(round(overlap_sec * fps)))
    prob_sum = np.zeros((Tlen, len(CLASSES)), dtype=np.float64)
    count = np.zeros((Tlen,), dtype=np.int32)
    for start in range(0, Tlen - window + 1, step):
        end = start + window
        x = torch.from_numpy(feats[start:end]).unsqueeze(0).to(DEVICE)
        probs = torch.softmax(model(x), dim=-1)[0].cpu().numpy()
        prob_sum[start:end] += probs
        count[start:end] += 1
    # tail: make sure the last `window` frames are covered even if step
    # doesn't land exactly on Tlen - window
    if count[-1] == 0:
        start = Tlen - window
        x = torch.from_numpy(feats[start:]).unsqueeze(0).to(DEVICE)
        probs = torch.softmax(model(x), dim=-1)[0].cpu().numpy()
        prob_sum[start:] += probs
        count[start:] += 1
    count = np.maximum(count, 1)[:, None]
    return (prob_sum / count).argmax(axis=-1)


# ----------------------------------------------------------------------
# Frame-level metrics
# ----------------------------------------------------------------------
def frame_level_report(all_pred, all_gt):
    n = len(CLASSES)
    conf = np.zeros((n, n), dtype=np.int64)  # rows=gt, cols=pred
    for p, g in zip(all_pred, all_gt):
        if g == IGNORE_ID:
            continue
        conf[g, p] += 1

    print("\n[frame-level]")
    print(f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for c, name in enumerate(CLASSES):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        support = conf[c, :].sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        print(f"{name:<12}{prec:>10.3f}{rec:>10.3f}{f1:>10.3f}{support:>10d}")

    print("\nconfusion matrix (rows=gt, cols=pred):")
    header = "".join(f"{c[:8]:>10}" for c in CLASSES)
    print(" " * 12 + header)
    for c, name in enumerate(CLASSES):
        row = "".join(f"{v:>10d}" for v in conf[c])
        print(f"{name:<12}{row}")
    return conf


# ----------------------------------------------------------------------
# Segment-level metric: F1 at an IoU threshold, per class, then averaged.
# ----------------------------------------------------------------------
def to_segments(labels):
    """labels: (T,) class ids, IGNORE_ID allowed -> list of (cls, start, end)
    exclusive end, skipping ignored runs."""
    segs = []
    start = 0
    for t in range(1, len(labels) + 1):
        if t == len(labels) or labels[t] != labels[start]:
            if labels[start] != IGNORE_ID:
                segs.append((labels[start], start, t))
            start = t
    return segs


def segment_iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[1], b[1]))
    union = max(a[2], b[2]) - min(a[1], b[1])
    return inter / union if union > 0 else 0.0


def segmental_f1(pred_tracks, gt_tracks, iou_thresh=0.5):
    """pred_tracks/gt_tracks: list of per-track label arrays, same order."""
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(len(CLASSES))}
    for pred, gt in zip(pred_tracks, gt_tracks):
        p_segs = to_segments(pred)
        g_segs = to_segments(gt)
        matched_gt = set()
        for pc, ps, pe in p_segs:
            best_iou, best_j = 0.0, -1
            for j, (gc, gs, ge) in enumerate(g_segs):
                if gc != pc or j in matched_gt:
                    continue
                iou = segment_iou((pc, ps, pe), (gc, gs, ge))
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh:
                per_class[pc]["tp"] += 1
                matched_gt.add(best_j)
            else:
                per_class[pc]["fp"] += 1
        for j, (gc, gs, ge) in enumerate(g_segs):
            if j not in matched_gt:
                per_class[gc]["fn"] += 1

    print(f"\n[segment-level F1@IoU={iou_thresh}]")
    print(f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'#gt segs':>10}")
    f1s = []
    for c, name in enumerate(CLASSES):
        s = per_class[c]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        f1s.append(f1)
        print(f"{name:<12}{prec:>10.3f}{rec:>10.3f}{f1:>10.3f}{tp+fn:>10d}")
    print(f"{'mean F1':<12}{'':>20}{np.mean(f1s):>10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    args = ap.parse_args()

    matches = T.load_matches(_p(P["manifest"]))
    _, val_matches = T.match_split(matches, A["val_frac"], A["seed"])
    print(f"[data] evaluating on {len(val_matches)} held-out matches")

    model = load_model(args.ckpt)

    all_pred_frames, all_gt_frames = [], []
    pred_tracks, gt_tracks = [], []
    for match in val_matches:
        for tid, alliance, feats, labels in T.build_track_sequences(match):
            if (labels != IGNORE_ID).sum() == 0:
                continue  # nothing labeled on this track, skip
            pred = predict_track(model, feats, A["eval_window_sec"],
                                 A["eval_overlap_sec"], A["fps"])
            pred_tracks.append(pred)
            gt_tracks.append(labels)
            all_pred_frames.extend(pred.tolist())
            all_gt_frames.extend(labels.tolist())

    if not gt_tracks:
        raise SystemExit("[abort] no labeled tracks in the held-out matches.")

    frame_level_report(np.array(all_pred_frames), np.array(all_gt_frames))
    segmental_f1(pred_tracks, gt_tracks, iou_thresh=args.iou_thresh)


if __name__ == "__main__":
    main()
