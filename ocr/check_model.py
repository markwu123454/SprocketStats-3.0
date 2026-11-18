"""Smoke test: PaddleOCR PP-OCRv6 (medium) on the Transformers engine, on the GPU.

Run from the repo root:
    .venv\\Scripts\\python ocr\\check_model.py [image_path_or_url]
"""

import os
import sys
from pathlib import Path

# Keep model weights inside the repo (ocr/.cache, gitignored) instead of ~/.paddlex
# and ~/.cache/huggingface. Must be set before paddleocr/paddlex are imported.
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(CACHE_DIR / "paddlex"))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))

import torch  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402

DEFAULT_IMAGE = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
)


def model_devices() -> set[str]:
    """Devices holding the weights of every torch module the pipeline loaded."""
    import gc

    devices = set()
    for obj in gc.get_objects():
        if isinstance(obj, torch.nn.Module):
            for p in obj.parameters(recurse=False):
                devices.add(str(p.device))
    return devices


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


def main() -> None:
    image = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    ocr = build_ocr()

    results = ocr.predict(image)

    for res in results:
        for text, score in zip(res["rec_texts"], res["rec_scores"]):
            print(f"{score:.4f}  {text}")

    print()
    print("Model weights on:", ", ".join(sorted(model_devices())) or "(none found)")
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
