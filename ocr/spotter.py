"""Robot spotter: the repo's HRNet keypoint model (hrnet/), used read-only.

Gives one dot per robot, on the carpet under the bumper centre, labelled blue or red.
See hrnet/README.md for how the model was trained.
"""

import importlib.util
from pathlib import Path

import cv2
import timm
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
HRNET_MODEL_PY = REPO_ROOT / "hrnet" / "model.py"
DEFAULT_CKPT = REPO_ROOT / "hrnet" / "data" / "frc" / "checkpoints" / "best.pt"


def _load_hrnet_module():
    # Load by path under a unique name: "model" is too generic to put hrnet/ on sys.path.
    spec = importlib.util.spec_from_file_location("hrnet_model", HRNET_MODEL_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RobotSpotter:
    def __init__(self, ckpt: Path = DEFAULT_CKPT, device: str = "cuda"):
        hr = _load_hrnet_module()
        ck = torch.load(ckpt, map_location=device, weights_only=False)  # the repo's own checkpoint
        cfg = ck["cfg"]  # trust the checkpoint's saved config over hrnet/config.yaml (they can drift)
        self.hm = cfg["heatmap"]
        self.classes = list(hr.CLASSES)
        self._decode = hr.decode_peaks

        # HeatmapNet asks timm for ImageNet-pretrained weights (an ~80 MB download) that
        # load_state_dict overwrites immediately, so build it without them.
        create = timm.create_model
        timm.create_model = lambda *a, **kw: create(*a, **{**kw, "pretrained": False})
        try:
            net = hr.HeatmapNet(cfg["train"]["backbone"], len(self.classes), self.hm["output_stride"])
        finally:
            timm.create_model = create
        net.load_state_dict(ck["model"])
        self.net = net.to(device).eval()

        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        # Leave cudnn.benchmark off: it re-tunes kernels for every new input shape, and the
        # OCR models running alongside see many shapes (one first frame took ~2 min with it on).
        torch.backends.cudnn.benchmark = False

    @torch.no_grad()
    def __call__(self, frame_bgr) -> list[tuple[float, float, str, float]]:
        """Robots in one frame as [(x, y, alliance, score)] in frame pixels."""
        H, W = frame_bgr.shape[:2]
        ih, iw = self.hm["input_size"]
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (iw, ih)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float() / 255.0
        x = (x - self.mean) / self.std
        dets = self._decode(self.net(x[None]), self.hm["peak_threshold"],
                            self.hm["nms_kernel"], self.hm["max_instances"])[0]
        s = self.hm["output_stride"]
        return [(xh * s * W / iw, yh * s * H / ih, self.classes[c], p) for xh, yh, c, p in dets]
