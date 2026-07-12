#!/usr/bin/env python3
"""
Step 2 — Train the heatmap model.

Key choices baked in:
  - Validation holds out an ENTIRE season (config train.val_season). Random
    frame splits leak, because 0.5fps frames from one match are near-dupes.
    Season holdout is the honest proxy for "works on a 2027 game never seen".
  - Metric: PCK-style. A predicted peak is correct if it's within `tol` px
    (in original-image scale) of a same-alliance GT point, one-to-one matched.
"""
import os
# Reduce allocator fragmentation on the 16GB card (must be set before CUDA init).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import pathlib, yaml, math
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

from model import (HeatmapNet, FRCKeypointDataset, heatmap_loss,
                   decode_peaks, CLASSES)

CFG = yaml.safe_load(open(pathlib.Path(__file__).parent / "config.yaml"))
T = CFG["train"]; HM = CFG["heatmap"]; P = CFG["paths"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = pathlib.Path(__file__).parent
def _p(path_str):
    p = pathlib.Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)



def load_records(manifest):
    import json
    recs = []
    for line in open(manifest):
        recs.append(json.loads(line))
    return recs


def match_split(records, val_frac=0.15, seed=42):
    """Hold out whole MATCHES for validation. All frames of a match go to the
    same side, so 0.5fps near-duplicate frames can't leak across the split.
    Falls back to a per-frame split only if there's just one match."""
    import random
    by_match = {}
    for r in records:
        key = r.get("match") or r.get("event") or "unknown"
        by_match.setdefault(key, []).append(r)

    matches = sorted(by_match)
    rng = random.Random(seed)
    rng.shuffle(matches)

    if len(matches) < 2:
        # Only one match available (e.g. early labeling). Split frames instead,
        # with a warning — this WILL leak near-duplicate frames and inflate val.
        print("[warn] only one match present; falling back to a frame-level "
              "split. Val score will be optimistic until you label >=2 matches.")
        frames = list(records); rng.shuffle(frames)
        n_val = max(1, int(len(frames) * val_frac))
        return frames[n_val:], frames[:n_val]

    n_val = max(1, int(len(matches) * val_frac))
    val_matches = set(matches[:n_val])
    train, val = [], []
    for mkey, items in by_match.items():
        (val if mkey in val_matches else train).extend(items)
    print(f"[data] {len(matches)} matches -> {len(val_matches)} held out for val")
    print(f"[data] val matches: {sorted(val_matches)}")
    return train, val


@torch.no_grad()
def evaluate(model, loader, out_stride, tol_frac=0.02):
    """PCK: fraction of GT points matched by a same-class pred within tol.
    tol is tol_frac of the image diagonal (scale-invariant)."""
    model.eval()
    tp = fp = fn = 0
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        dets = decode_peaks(logits, HM["peak_threshold"], HM["nms_kernel"],
                            HM["max_instances"])
        # recover GT peaks from target heatmaps (argmax clusters ~ use the
        # same decode on the target so eval is apples-to-apples)
        gt = decode_peaks(torch.logit(targets.clamp(1e-4, 1 - 1e-4)).to(DEVICE),
                          0.5, HM["nms_kernel"], HM["max_instances"])
        h = imgs.shape[2] // out_stride
        diag = math.hypot(h, h)
        tol = tol_frac * diag
        for pd, gd in zip(dets, gt):
            for c in range(len(CLASSES)):
                P_ = [(x, y) for x, y, cc, s in pd if cc == c]
                G_ = [(x, y) for x, y, cc, s in gd if cc == c]
                if not P_ and not G_:
                    continue
                if not P_:
                    fn += len(G_); continue
                if not G_:
                    fp += len(P_); continue
                C_ = np.zeros((len(G_), len(P_)))
                for i, g in enumerate(G_):
                    for j, p in enumerate(P_):
                        C_[i, j] = math.hypot(g[0]-p[0], g[1]-p[1])
                ri, ci = linear_sum_assignment(C_)
                matched = 0
                for i, j in zip(ri, ci):
                    if C_[i, j] <= tol:
                        matched += 1
                tp += matched
                fn += len(G_) - matched
                fp += len(P_) - matched
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1}


def main():
    manifest = _p(P["export_dir"]) / "manifest.jsonl"
    records = load_records(manifest)
    torch.manual_seed(T["seed"]); np.random.seed(T["seed"])

    train_recs, val_recs = match_split(
        records, val_frac=T.get("val_frac", 0.15), seed=T["seed"])

    tr = FRCKeypointDataset(records=train_recs, input_size=tuple(HM["input_size"]),
                            out_stride=HM["output_stride"], sigma=HM["sigma"], train=True)
    va = FRCKeypointDataset(records=val_recs, input_size=tuple(HM["input_size"]),
                            out_stride=HM["output_stride"], sigma=HM["sigma"], train=False)
    print(f"[data] train={len(tr)} val={len(va)}")

    # Guard: refuse to run on an unusably small set (produces meaningless F1=0).
    MIN_TRAIN, MIN_VAL = 20, 2
    if len(tr) < MIN_TRAIN or len(va) < MIN_VAL:
        raise SystemExit(
            f"[abort] not enough labeled data: train={len(tr)} (need >={MIN_TRAIN}), "
            f"val={len(va)} (need >={MIN_VAL}). Label more frames first, or for a "
            f"quick smoke test point the dataset at manifest_debug.jsonl and lower "
            f"these thresholds. Also check the val_season actually has labels."
        )

    tl = DataLoader(tr, batch_size=T["batch_size"], shuffle=True,
                    num_workers=T["num_workers"], pin_memory=True, drop_last=True)
    vl = DataLoader(va, batch_size=T["batch_size"], shuffle=False,
                    num_workers=T["num_workers"], pin_memory=True)

    model = HeatmapNet(T["backbone"], len(CLASSES), HM["output_stride"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=T["base_lr"],
                            weight_decay=T["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T["epochs"])
    use_amp = T["amp"] and DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = _p(P["checkpoints"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    accum = max(1, T.get("grad_accum", 1))
    for epoch in range(T["epochs"]):
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for i, (imgs, targets) in enumerate(tl):
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
                # scale down so accumulated grads match a single big batch
                loss = heatmap_loss(logits, targets) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
            running += loss.item() * accum          # report un-scaled loss
            if (i + 1) % 100 == 0:
                print(f"  e{epoch} it{i+1}/{len(tl)} loss={running/(i+1):.5f}")
        # flush a trailing partial accumulation window (drop_last keeps batches
        # uniform, but the epoch length may not be a multiple of `accum`)
        if len(tl) % accum != 0:
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
        sched.step()

        m = evaluate(model, vl, HM["output_stride"])
        print(f"[val] epoch {epoch}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
        torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": CFG},
                   ckpt_dir / "last.pt")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": CFG},
                       ckpt_dir / "best.pt")
            print(f"  * new best F1={best_f1:.3f}")

    print(f"[done] best val F1={best_f1:.3f}")


if __name__ == "__main__":
    main()
