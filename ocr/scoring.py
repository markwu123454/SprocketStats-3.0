"""Bumper text: find it in a robot crop, then score the known team numbers against it.

"Score candidates, don't decode" (README section 6). PP-OCRv6's recognizer is a CTC model:
for each horizontal slice of a text crop it outputs a probability for every character.
Free-form decoding keeps only the single best path through those probabilities, which is
how a blurry "8013" comes out as "8913". Instead we compute log P(team | crop) for each
candidate team by summing over *every* path that spells it -- exactly what torch's
ctc_loss computes. Among a handful of known teams that stays decisive on crops far too
blurry to decode.
"""

from check_model import GPU_ENGINE  # also points the model cache at ocr/.cache

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from paddleocr import TextDetection, TextRecognition  # noqa: E402
from paddlex.inference.models.text_recognition import predictor as _recp  # noqa: E402

# paddlex decodes the recognizer's per-timestep outputs to a string and discards them.
# Wrap its postprocess step to keep a copy (and the model's character list).
_CAPTURED: list = []
_CHARS: list[str] = []
_orig_postprocess = _recp.TextRecTransformersPredictor.postprocess


def _capturing_postprocess(self, outputs, **kwargs):
    if not _CHARS:
        chars = getattr(self.image_processor, "character_list", None) or \
            self.model_config.get("PostProcess", {}).get("character_dict")
        _CHARS.extend(chars)
    _CAPTURED.append(outputs)
    return _orig_postprocess(self, outputs, **kwargs)


_recp.TextRecTransformersPredictor.postprocess = _capturing_postprocess


def _log_probs(outputs) -> torch.Tensor:
    """Recognizer outputs -> [batch, time, chars] log-probabilities on the CPU."""
    t = getattr(outputs, "logits", None)
    if t is None:
        if isinstance(outputs, dict):
            t = outputs.get("logits", next(iter(outputs.values())))
        else:
            t = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    t = torch.as_tensor(t).float().cpu()
    if t.dim() == 2:
        t = t[None]
    is_prob = bool(t.min() >= 0) and torch.allclose(t.sum(-1), torch.ones(t.shape[:2]), atol=1e-2)
    return t.clamp_min(1e-12).log() if is_prob else F.log_softmax(t, -1)


def _char_index(ch: str) -> int:
    # CTC blank is class 0; the character list may or may not include it.
    return _CHARS.index(ch) if _CHARS[0] == "blank" else _CHARS.index(ch) + 1


def warp_quad(img, quad):
    """Straighten a 4-point text box (tl, tr, br, bl) into a horizontal crop, or None."""
    q = np.asarray(quad, dtype=np.float32)
    w = int(round(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2]))))
    h = int(round(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]))))
    if w < 6 or h < 6:
        return None
    m = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    out = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Vertical box (e.g. a side bumper seen from above): turn it on its side.
    return np.ascontiguousarray(np.rot90(out)) if h > 1.5 * w else out


class TeamScorer:
    def __init__(self):
        # Robot crops are small, so cap the long side at 960 rather than shrinking them.
        self.det = TextDetection(model_name="PP-OCRv6_medium_det",
                                 limit_side_len=960, limit_type="max", **GPU_ENGINE)
        self.rec = TextRecognition(model_name="PP-OCRv6_medium_rec", **GPU_ENGINE)

    def detect(self, crops) -> list[np.ndarray]:
        """Text boxes (N x 4 x 2 quads) for each crop."""
        if not crops:
            return []
        return [np.asarray(r["dt_polys"], dtype=float).reshape(-1, 4, 2) for r in self.det.predict(crops)]

    def score(self, text_crops, teams: list[str]) -> list[tuple[str, float, np.ndarray]]:
        """For each straightened text crop: (free_read, free_score, log P(team | crop) per team).

        Upside-down text is handled by also scoring the crop rotated 180 degrees and
        keeping the better orientation per team.
        """
        if not text_crops:
            return []
        imgs = []
        for c in text_crops:
            imgs += [c, np.ascontiguousarray(np.rot90(c, 2))]
        _CAPTURED.clear()
        results = list(self.rec.predict(imgs))
        lps = [lp for out in _CAPTURED for lp in _log_probs(out)]
        assert len(lps) == len(imgs) == len(results), (len(lps), len(imgs), len(results))

        targets = [torch.tensor([_char_index(ch) for ch in t]) for t in teams]
        flat, lens = torch.cat(targets), torch.tensor([len(t) for t in teams])
        scores = []
        for lp in lps:
            T, C = lp.shape
            nll = F.ctc_loss(lp[:, None, :].expand(T, len(teams), C).contiguous(), flat,
                             torch.full((len(teams),), T), lens, blank=0, reduction="none")
            scores.append(-nll.numpy())  # impossible alignments give -inf, never a free pass

        out = []
        for k in range(len(text_crops)):
            a, b = scores[2 * k], scores[2 * k + 1]
            best_side = 2 * k if a.max() >= b.max() else 2 * k + 1
            r = results[best_side]
            out.append((r["rec_text"], float(r["rec_score"]), np.maximum(a, b)))
        return out
