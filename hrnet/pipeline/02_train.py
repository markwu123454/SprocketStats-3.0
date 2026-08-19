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
from tqdm import tqdm

import sys as _sys
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_sys.path.insert(0, str(PROJECT_ROOT))

from model import (HeatmapNet, FRCKeypointDataset, heatmap_loss, class_names,
                   decode_peaks, CLASSES)

CFG = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
T = CFG["train"]; HM = CFG["heatmap"]; P = CFG["paths"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def _p(path_str):
    p = pathlib.Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)



def load_records(manifest):
    import json
    recs = []
    for line in open(manifest):
        recs.append(json.loads(line))
    return recs


def load_all(manifest_names, val_from):
    """Load one or more manifests, dedup by image, tag validation eligibility.

    Manifests are labeling ROUNDS, and rounds are not interchangeable. A round
    labeled in confidence-ascending order is a biased sample of its pool (it
    over-represents whatever the model was unsure about, including empty CGI
    frames), so it is fine to TRAIN on but will produce a misleading val score.
    --val-from names the manifests a val set may be drawn from; everything else
    is forced to train.
    """
    export_dir = _p(P["export_dir"])
    eligible_src = set(val_from) if val_from else set(manifest_names)
    seen, order = {}, []
    for name in manifest_names:
        path = export_dir / name
        if not path.exists():
            raise SystemExit(f"[abort] no manifest at {path}")
        recs = load_records(path)
        n_new = n_dup = 0
        for r in recs:
            key = r.get("image")
            r["_val_ok"] = name in eligible_src
            if key in seen:
                # Keep the first occurrence. A duplicate must never be able to
                # land on the opposite side of the split from its twin, and if
                # either copy came from a train-only round, neither is val-safe.
                seen[key]["_val_ok"] = seen[key]["_val_ok"] and r["_val_ok"]
                n_dup += 1
                continue
            seen[key] = r; order.append(key); n_new += 1
        pos = sum(1 for r in recs if r.get("points"))
        print(f"[data] {name}: {len(recs)} records ({pos} positive, "
              f"{len(recs)-pos} negative) | +{n_new} new, {n_dup} dup"
              f"{'' if name in eligible_src else '  [train-only]'}")
    return [seen[k] for k in order]


def match_split(records, val_frac=0.15, seed=42):
    """Hold out whole MATCHES for validation. All frames of a match go to the
    same side, so near-duplicate frames can't leak across the split.
    Falls back to a per-frame split only if there's just one match."""
    import random
    by_match = {}
    for r in records:
        key = r.get("match") or r.get("event") or "unknown"
        by_match.setdefault(key, []).append(r)

    # A match may only be held out if EVERY one of its frames came from a
    # val-eligible manifest — otherwise validating on it would either leak
    # (same match on both sides) or pull biased frames into the val set.
    matches = sorted(m for m, items in by_match.items()
                     if all(r.get("_val_ok", True) for r in items))
    n_ineligible = len(by_match) - len(matches)
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

    n_val = min(len(matches), max(1, int(len(by_match) * val_frac)))
    val_matches = set(matches[:n_val])
    train, val = [], []
    for mkey, items in by_match.items():
        (val if mkey in val_matches else train).extend(items)
    print(f"[data] {len(by_match)} matches ({len(matches)} val-eligible, "
          f"{n_ineligible} train-only) -> {len(val_matches)} held out for val")
    return train, val


def _score_pairs(dets, gts, n_cls, tol):
    """One-to-one Hungarian match within each channel. Returns (tp, fp, fn)."""
    tp = fp = fn = 0
    for pd, gd in zip(dets, gts):
        for c in range(n_cls):
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
                    C_[i, j] = math.hypot(g[0] - p[0], g[1] - p[1])
            ri, ci = linear_sum_assignment(C_)
            matched = sum(1 for i, j in zip(ri, ci) if C_[i, j] <= tol)
            tp += matched
            fn += len(G_) - matched
            fp += len(P_) - matched
    return tp, fp, fn


def _prf(tp, fp, fn):
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return prec, rec, 2 * prec * rec / (prec + rec + 1e-9)


@torch.no_grad()
def evaluate(model, loader, out_stride, n_classes, tol_frac=0.02):
    """PCK: fraction of GT points matched by a same-channel pred within tol.
    tol is tol_frac of the image diagonal (scale-invariant).

    Also reports CLASS-AGNOSTIC metrics, with the alliance channels max-pooled
    into one before decoding. This is the only way a 2-channel run and a merged
    1-channel run are comparable: in 2-channel mode a blue/red mix-up costs both
    a false positive and a false negative, a penalty the merged model cannot
    incur. Compare merged `f1` against 2-channel `agn_f1`, never against its
    per-alliance `f1`.
    """
    model.eval()
    tp = fp = fn = 0
    atp = afp = afn = 0
    for imgs, targets, _ignore in tqdm(loader, desc="val", unit="bt",
                                        leave=False, dynamic_ncols=True):
        imgs = imgs.to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(imgs)
        # recover GT peaks from target heatmaps (use the same decode on the
        # target so eval is apples-to-apples)
        tgt_logits = torch.logit(targets.clamp(1e-4, 1 - 1e-4))

        h = imgs.shape[2] // out_stride
        tol = tol_frac * math.hypot(h, h)

        dets = decode_peaks(logits, HM["peak_threshold"], HM["nms_kernel"],
                            HM["max_instances"])
        gt = decode_peaks(tgt_logits, 0.5, HM["nms_kernel"], HM["max_instances"])
        a, b, c = _score_pairs(dets, gt, n_classes, tol)
        tp += a; fp += b; fn += c

        # class-agnostic: collapse channels by max, decode as a single channel
        dets_a = decode_peaks(logits.max(1, keepdim=True).values,
                              HM["peak_threshold"], HM["nms_kernel"],
                              HM["max_instances"])
        gt_a = decode_peaks(tgt_logits.max(1, keepdim=True).values, 0.5,
                            HM["nms_kernel"], HM["max_instances"])
        a, b, c = _score_pairs(dets_a, gt_a, 1, tol)
        atp += a; afp += b; afn += c

    prec, rec, f1 = _prf(tp, fp, fn)
    aprec, arec, af1 = _prf(atp, afp, afn)
    return {"precision": prec, "recall": rec, "f1": f1,
            "agn_precision": aprec, "agn_recall": arec, "agn_f1": af1}


def _atomic_save(obj, dst):
    """Write to a temp file then rename — avoids Windows file-lock errors on last.pt."""
    import pathlib
    dst = pathlib.Path(dst)
    tmp = dst.with_suffix(".tmp")
    torch.save(obj, tmp)
    if dst.exists():
        dst.unlink()
    tmp.rename(dst)


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Train the heatmap keypoint model.")
    ap.add_argument("--merge-alliances", action="store_true",
                    help="Train ONE class-agnostic 'robot' channel instead of "
                         "separate blue/red channels. Uses the same labels — no "
                         "re-pull needed. Checkpoints get a _merged suffix so "
                         "they do not clobber the two-channel run.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override train.epochs from config (for quick trials).")
    ap.add_argument("--tag", type=str, default=None,
                    help="Extra checkpoint filename suffix.")
    ap.add_argument("--manifest", nargs="+", default=["manifest.jsonl"],
                    help="One or more manifest filenames under paths.export_dir. "
                         "Merged and deduped by image path.")
    ap.add_argument("--val-frac", type=float, default=None,
                    help="Override train.val_frac. Note this is a fraction of "
                         "ALL matches, so when most are train-only it consumes "
                         "a large share of the val-eligible ones.")
    ap.add_argument("--val-from", nargs="+", default=None,
                    help="Subset of --manifest that validation may be drawn "
                         "from. Use this to keep a biased labeling round "
                         "(e.g. one labeled in confidence order) out of val. "
                         "Default: all of them.")
    return ap.parse_args()


def main(args):
    merge = args.merge_alliances or bool(T.get("merge_alliances", False))
    classes = class_names(merge)
    n_ch = len(classes)
    epochs = args.epochs if args.epochs is not None else T["epochs"]
    tag = ("_merged" if merge else "") + (f"_{args.tag}" if args.tag else "")
    print(f"[mode] channels={n_ch} ({', '.join(classes)}) | epochs={epochs}"
          + (f" | ckpt suffix '{tag}'" if tag else ""))

    unknown = set(args.val_from or []) - set(args.manifest)
    if unknown:
        raise SystemExit(f"[abort] --val-from names manifests not in "
                         f"--manifest: {sorted(unknown)}")

    records = load_all(args.manifest, args.val_from)
    torch.manual_seed(T["seed"]); np.random.seed(T["seed"])

    train_recs, val_recs = match_split(
        records, val_frac=T.get("val_frac", 0.15), seed=T["seed"])

    tr = FRCKeypointDataset(records=train_recs, input_size=tuple(HM["input_size"]),
                            out_stride=HM["output_stride"], sigma=HM["sigma"],
                            train=True, merge_alliances=merge)
    va = FRCKeypointDataset(records=val_recs, input_size=tuple(HM["input_size"]),
                            out_stride=HM["output_stride"], sigma=HM["sigma"],
                            train=False, merge_alliances=merge)
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
                    num_workers=T["num_workers"], pin_memory=True, drop_last=True,
                persistent_workers=T["num_workers"] > 0)
    vl = DataLoader(va, batch_size=T["batch_size"], shuffle=False,
                    num_workers=T["num_workers"], pin_memory=True,
                persistent_workers=T["num_workers"] > 0)

    model = HeatmapNet(T["backbone"], n_ch, HM["output_stride"]).to(DEVICE)
    freeze_epochs = T.get("freeze_backbone_epochs", 0)

    def _set_backbone_grad(requires_grad: bool):
        for p in model.backbone.parameters():
            p.requires_grad_(requires_grad)

    backbone_lr_scale = T.get("backbone_lr_scale", 0.2)  # backbone fine-tunes slower

    if freeze_epochs > 0:
        _set_backbone_grad(False)
        print(f"[train] backbone frozen for first {freeze_epochs} epochs")

    def _make_optimizer():
        head_params = [p for p in model.head.parameters() if p.requires_grad]
        bb_params   = [p for p in model.backbone.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr": T["base_lr"]}]
        if bb_params:
            groups.append({"params": bb_params, "lr": T["base_lr"] * backbone_lr_scale})
        return torch.optim.AdamW(groups, weight_decay=T["weight_decay"])

    opt = _make_optimizer()
    # ReduceLROnPlateau reacts to val F1 plateaus — far better than fixed cosine
    # for small datasets where the optimum arrives at an unpredictable epoch.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=8, min_lr=1e-6)
    use_amp = T["amp"] and DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = _p(P["checkpoints"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    accum = max(1, T.get("grad_accum", 1))
    for epoch in range(epochs):
        if freeze_epochs > 0 and epoch == freeze_epochs:
            _set_backbone_grad(True)
            # Rebuild optimizer with two param groups: head at full LR, backbone at scaled LR.
            # Reset scheduler so patience counter starts fresh from the unfrozen baseline.
            opt = _make_optimizer()
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max", factor=0.5, patience=8, min_lr=1e-6)
            print(f"[train] backbone unfrozen at epoch {epoch} "
                  f"(head_lr={T['base_lr']:.2e}, bb_lr={T['base_lr']*backbone_lr_scale:.2e})")
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(tl, desc=f"e{epoch:03d}", unit="bt", leave=False,
                    dynamic_ncols=True)
        for i, (imgs, targets, ignore_masks) in enumerate(pbar):
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            ignore_masks = ignore_masks.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
                # scale down so accumulated grads match a single big batch
                loss = heatmap_loss(logits, targets, ignore_masks) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
            running += loss.item() * accum          # report un-scaled loss
            pbar.set_postfix(loss=f"{running/(i+1):.5f}")
        # flush a trailing partial accumulation window (drop_last keeps batches
        # uniform, but the epoch length may not be a multiple of `accum`)
        if len(tl) % accum != 0:
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
        m = evaluate(model, vl, HM["output_stride"], n_ch)
        sched.step(m["agn_f1"])   # agnostic F1 is less noisy: doesn't penalize alliance swap
        lrs = [pg["lr"] for pg in opt.param_groups]
        lr_str = "/".join(f"{lr:.2e}" for lr in lrs)
        print(f"[val] epoch {epoch}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} | agnostic P={m['agn_precision']:.3f} "
              f"R={m['agn_recall']:.3f} F1={m['agn_f1']:.3f}  lr={lr_str}")
        meta = {"model": model.state_dict(), "epoch": epoch, "cfg": CFG,
                "merge_alliances": merge, "classes": classes}
        _atomic_save(meta, ckpt_dir / f"last{tag}.pt")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_agn = m["agn_f1"]
            _atomic_save(meta, ckpt_dir / f"best{tag}.pt")
            print(f"  * new best F1={best_f1:.3f} (agnostic {best_agn:.3f})")

    print(f"[done] best val F1={best_f1:.3f}")
    print("[compare] merged runs report 1 channel, so their F1 IS the agnostic "
          "number. Compare it against the two-channel run's 'agnostic F1', not "
          "its per-alliance F1.")


if __name__ == "__main__":
    main(parse_args())
