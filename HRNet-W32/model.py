#!/usr/bin/env python3
"""
Model + dataset + heatmap targets for box-free alliance-keypoint detection.

Design:
  - Backbone: HRNet-W32 (via timm), keeps high-res features -> good for
    precise, small keypoints across huge scale variation.
  - Head: 1x1 conv to 2 channels (ch0=blue, ch1=red), each a Gaussian
    peak map at stride 4. No boxes, no anchors, no NMS-of-boxes.
  - Loss: per-pixel MSE on the heatmaps with foreground weighting.
    (This is the CenterNet/pose-heatmap formulation.)

Why not YOLO-pose: it needs box<->keypoint association. Our points can sit
outside any robot box (airborne climb) and robots overlap heavily, so the
box relationship is unreliable. Heatmap peak-finding sidesteps all of that.
"""
import json, math, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
import timm

CLASSES = ["blue", "red"]
CLS_TO_CH = {c: i for i, c in enumerate(CLASSES)}


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class HeatmapNet(nn.Module):
    def __init__(self, backbone="hrnet_w32", num_classes=2, out_stride=4):
        super().__init__()
        # features_only returns a pyramid at strides [2, 4, 8, 16, 32]. Pick the
        # pyramid level whose stride == out_stride (index 1 == stride 4). NOTE:
        # HRNet's finest level is stride 2 (index 0), NOT stride 4 — using index
        # 0 emits a stride-2 heatmap that mismatches the stride-4 targets.
        stride_to_idx = {2: 0, 4: 1, 8: 2, 16: 3, 32: 4}
        if out_stride not in stride_to_idx:
            raise ValueError(f"out_stride must be one of {sorted(stride_to_idx)}, "
                             f"got {out_stride}")
        idx = stride_to_idx[out_stride]
        self.backbone = timm.create_model(
            backbone, features_only=True, pretrained=True, out_indices=(idx,)
        )
        feat_ch = self.backbone.feature_info.channels()[0]
        self.head = nn.Sequential(
            nn.Conv2d(feat_ch, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )
        self.out_stride = out_stride

    def forward(self, x):
        f = self.backbone(x)[0]              # (B, C, H/4, W/4)
        h = self.head(f)                     # (B, 2, H/4, W/4)
        return h                             # raw logits; sigmoid at loss/decode


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
                 sigma=2.0, train=True):
        self.items = list(records)
        self.in_h, self.in_w = input_size
        self.out_stride = out_stride
        self.out_h, self.out_w = self.in_h // out_stride, self.in_w // out_stride
        self.sigma = sigma
        self.train = train

    def __len__(self):
        return len(self.items)

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

        # saturation jitter (convert to HSV, perturb S channel)
        if np.random.rand() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= np.random.uniform(0.6, 1.4)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

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

        target = np.zeros((len(CLASSES), self.out_h, self.out_w), dtype=np.float32)
        for p in pts:
            hx = p["x_px"] * sx / self.out_stride
            hy = p["y_px"] * sy / self.out_stride
            if 0 <= hx < self.out_w and 0 <= hy < self.out_h:
                draw_gaussian(target[CLS_TO_CH[p["cls"]]], hx, hy, self.sigma)

        img_t = torch.from_numpy(img_r).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) \
                / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return img_t, torch.from_numpy(target)


# ----------------------------------------------------------------------
# Loss: weighted MSE (foreground pixels upweighted so sparse peaks matter)
# ----------------------------------------------------------------------
def heatmap_loss(pred_logits, target, fg_weight=20.0):
    pred = torch.sigmoid(pred_logits)
    weight = 1.0 + fg_weight * target          # emphasize peak neighborhoods
    return (weight * (pred - target) ** 2).mean()


# ----------------------------------------------------------------------
# Decode: sigmoid -> local-max NMS -> peaks above threshold
# ----------------------------------------------------------------------
def decode_peaks(pred_logits, threshold=0.3, nms_kernel=5, max_instances=12):
    """Returns list per image: [(x_hm, y_hm, cls_idx, score), ...] in heatmap px."""
    prob = torch.sigmoid(pred_logits)          # (B, 2, h, w)
    pad = nms_kernel // 2
    pooled = F.max_pool2d(prob, nms_kernel, stride=1, padding=pad)
    keep = (pooled == prob) & (prob >= threshold)
    out = []
    B, C, h, w = prob.shape
    for b in range(B):
        dets = []
        for c in range(C):
            ys, xs = torch.where(keep[b, c])
            for y, x in zip(ys.tolist(), xs.tolist()):
                dets.append((x, y, c, float(prob[b, c, y, x])))
        dets.sort(key=lambda d: -d[3])
        out.append(dets[:max_instances])
    return out
