"""PaddleOCR PP-OCRv6 (medium) on the Transformers engine, GPU 0: shared setup.

Importing this module points the model caches at ocr/.cache (gitignored) instead of
~/.paddlex and ~/.cache/huggingface. paddleocr/paddlex read those variables when they
are first imported, so this module must be imported before them anywhere in the
process -- run/team_scorer.py and the debug scripts import it first for that reason.
"""

import os
import sys
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(CACHE_DIR / "paddlex"))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))

import torch  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402


# Transformers engine on GPU 0. For non-Paddle engines, engine_config is passed through
# as-is and validated against paddlex's TransformersEngineConfig (extra keys are rejected).
GPU_ENGINE = dict(
    engine="transformers",
    device="gpu:0",
    engine_config={"device_type": "gpu", "device_id": 0},
)


def build_ocr() -> PaddleOCR:
    """PP-OCRv6_medium det+rec on the Transformers engine, GPU 0. Shared by other scripts."""
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available to torch; refusing to fall back to CPU.")

    # With no lang/ocr_version, PaddleOCR defaults to PP-OCRv6_medium_det + _rec.
    return PaddleOCR(
        **GPU_ENGINE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
