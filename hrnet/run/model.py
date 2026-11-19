#!/usr/bin/env python3
"""
Model + peak decoding for box-free alliance-keypoint detection.

The dataset, heatmap targets and loss are in train/dataset.py: inference only
needs the network and the decoder, and the channel layout below (CLASSES /
CLS_TO_CH) stays here because the decoder and the targets must agree on it.

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
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

CLASSES = ["blue", "red"]
CLS_TO_CH = {c: i for i, c in enumerate(CLASSES)}

# Alliance-merged mode: one class-agnostic "robot" channel instead of blue/red.
#
# Detection ("is there a robot here") and alliance classification ("what colour
# are its bumpers") are separable problems, and the second one is unanswerable
# from pixels when the bumpers are occluded. Splitting the output by alliance
# forces the detector to solve both at once, so a robot whose bumpers are hidden
# has no channel it clearly belongs to. Merging tests whether that split is
# costing detection recall. Alliance can be resolved downstream instead, where
# the exactly-3-red/3-blue constraint over tracks makes it much easier.
MERGED_CLASSES = ["robot"]


def class_names(merge_alliances: bool):
    """Channel names for the active mode."""
    return list(MERGED_CLASSES if merge_alliances else CLASSES)


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
