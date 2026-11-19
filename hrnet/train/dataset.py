#!/usr/bin/env python3
"""
Dataset, heatmap targets and loss for training the alliance-keypoint model
(the network itself and peak decoding are in run/model.py).
"""
import json, math, pathlib  # math used in _augment (random erasing)
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from hrnet.run.model import CLS_TO_CH, class_names


# ----------------------------------------------------------------------
# Target generation: place a 2D Gaussian at each keypoint in its channel
# ----------------------------------------------------------------------
def draw_gaussian(hm, cx, cy, sigma):
    """In-place add a Gaussian peak to a single-channel heatmap (H,W)."""
    H, W = hm.shape
    tmp = 3 * sigma
    x0, x1 = int(cx - tmp), int(cx + tmp + 1)
    y0, y1 = int(cy - tmp), int(cy + tmp + 1)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    if x0c >= x1c or y0c >= y1c:
        return
    ys = np.arange(y0c, y1c)[:, None]
    xs = np.arange(x0c, x1c)[None, :]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    # keep the max where peaks overlap (nearby robots)
    hm[y0c:y1c, x0c:x1c] = np.maximum(hm[y0c:y1c, x0c:x1c], g)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class FRCKeypointDataset(Dataset):
    def __init__(self, records, input_size=(768, 768), out_stride=4,
                 sigma=2.0, train=True, merge_alliances=False):
        self.items = list(records)
        self.in_h, self.in_w = input_size
        self.out_stride = out_stride
        self.out_h, self.out_w = self.in_h // out_stride, self.in_w // out_stride
        self.sigma = sigma
        self.train = train
        # merge_alliances collapses blue/red into a single "robot" channel.
        # Labels on disk are unchanged — this only affects target construction,
        # so switching modes needs no re-pull and no re-annotation.
        self.merge_alliances = merge_alliances
        self.classes = class_names(merge_alliances)
        self.n_channels = len(self.classes)

    def __len__(self):
        return len(self.items)

    def _channel_for(self, cls):
        """Heatmap channel for a labeled point, or None if not a positive target.

          blue / red  -> their alliance channel, or channel 0 when merged.
          unknown     -> position is good, alliance unknown; positive in merged
                         mode, ignored (not hard-negative) in two-channel mode.
          climbing    -> robot is off-ground; position not a carpet point.
                         Becomes an ignore region, never a positive or negative.
        """
        if cls in ("climbing", "unknown") and not self.merge_alliances:
            return None
        if cls == "climbing":
            return None
        if cls == "unknown":
            return 0   # merge mode: treat as robot
        if self.merge_alliances:
            return 0
        return CLS_TO_CH.get(cls)

    def _is_ignore(self, cls):
        """True for points that should contribute zero gradient (neither positive
        nor negative). Climbing robots are visible but not carpet-plane targets;
        training the model to suppress them actively hurts recall."""
        if cls == "climbing":
            return True
        if cls == "unknown" and not self.merge_alliances:
            return True
        return False

    def _augment(self, img, pts):
        # horizontal flip
        if np.random.rand() < 0.5:
            img = img[:, ::-1, :]
            W = img.shape[1]
            for p in pts:
                p["x_px"] = W - 1 - p["x_px"]

        # small rotation (camera is mostly overhead but not perfectly level)
        if np.random.rand() < 0.5:
            H, W = img.shape[:2]
            angle = np.random.uniform(-12, 12)
            cx, cy = W / 2, H / 2
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            ones = np.ones((len(pts), 1))
            coords = np.array([[p["x_px"], p["y_px"]] for p in pts]) if pts else np.zeros((0, 2))
            if len(coords):
                rot = (M[:, :2] @ coords.T + M[:, 2:]).T
                for p, (rx, ry) in zip(pts, rot):
                    p["x_px"], p["y_px"] = float(rx), float(ry)
            pts = [p for p in pts if 0 <= p["x_px"] < W and 0 <= p["y_px"] < H]

        # random scale + crop — wider range helps small dataset generalise across zoom levels
        if np.random.rand() < 0.75:
            H, W = img.shape[:2]
            scale = np.random.uniform(0.60, 1.0)
            ch, cw = int(H * scale), int(W * scale)
            y0 = np.random.randint(0, H - ch + 1)
            x0 = np.random.randint(0, W - cw + 1)
            img = img[y0:y0 + ch, x0:x0 + cw]
            for p in pts:
                p["x_px"] -= x0
                p["y_px"] -= y0
            pts = [p for p in pts if 0 <= p["x_px"] < cw and 0 <= p["y_px"] < ch]

        # color: brightness + contrast
        if np.random.rand() < 0.6:
            a = 1.0 + np.random.uniform(-0.3, 0.3)
            b = np.random.uniform(-20, 20)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

        # HSV: saturation + hue jitter (hue matters for blue/red discrimination)
        if np.random.rand() < 0.6:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= np.random.uniform(0.6, 1.4)           # saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-15, 15)) % 180  # hue ±15°
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Gaussian noise — simulates sensor noise and compression artefacts
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, np.random.uniform(3, 12), img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Random erasing — forces model to rely on partial observations
        if np.random.rand() < 0.4:
            H, W = img.shape[:2]
            area = H * W
            erase_area = np.random.uniform(0.02, 0.15) * area
            ar = np.random.uniform(0.3, 3.3)
            eh = min(H, int(math.sqrt(erase_area * ar)))
            ew = min(W, int(math.sqrt(erase_area / ar)))
            if eh > 0 and ew > 0:
                ey = np.random.randint(0, H - eh + 1)
                ex = np.random.randint(0, W - ew + 1)
                img[ey:ey+eh, ex:ex+ew] = np.random.randint(0, 256, (eh, ew, 3), dtype=np.uint8)

        return img, pts

    def __getitem__(self, idx):
        rec = self.items[idx]
        img = cv2.imread(rec["image"])          # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H0, W0 = img.shape[:2]

        pts = [dict(p) for p in rec["points"]]  # copy
        if self.train:
            img, pts = self._augment(img, pts)
            H0, W0 = img.shape[:2]

        # resize to network input, scale points accordingly
        img_r = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        sx, sy = self.in_w / W0, self.in_h / H0

        target = np.zeros((self.n_channels, self.out_h, self.out_w), dtype=np.float32)
        # ignore_mask: regions that contribute zero gradient (climbing, unknown-2ch).
        # Shape (1, H, W) so it broadcasts over all output channels.
        ignore = np.zeros((1, self.out_h, self.out_w), dtype=np.float32)
        for p in pts:
            hx = p["x_px"] * sx / self.out_stride
            hy = p["y_px"] * sy / self.out_stride
            if not (0 <= hx < self.out_w and 0 <= hy < self.out_h):
                continue
            if self._is_ignore(p["cls"]):
                # Wider blob so the whole robot footprint is masked, not just peak
                draw_gaussian(ignore[0], hx, hy, self.sigma * 2.5)
            else:
                ch = self._channel_for(p["cls"])
                if ch is not None:
                    draw_gaussian(target[ch], hx, hy, self.sigma)

        img_t = torch.from_numpy(img_r).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) \
                / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return img_t, torch.from_numpy(target), torch.from_numpy(ignore)


# ----------------------------------------------------------------------
# Loss: weighted MSE (foreground pixels upweighted so sparse peaks matter)
# ----------------------------------------------------------------------
def heatmap_loss(pred_logits, target, ignore_mask=None, fg_weight=20.0):
    pred = torch.sigmoid(pred_logits)
    weight = 1.0 + fg_weight * target          # emphasize peak neighborhoods
    loss = weight * (pred - target) ** 2
    if ignore_mask is not None:
        # ignore_mask (B,1,H,W): zero gradient at climbing/unknown regions
        loss = loss * (1.0 - ignore_mask.clamp(0.0, 1.0))
    return loss.mean()
