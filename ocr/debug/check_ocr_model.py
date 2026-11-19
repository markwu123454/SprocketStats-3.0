"""Smoke test: PaddleOCR PP-OCRv6 (medium) on the Transformers engine, on the GPU.

Run from the repo root:
    .venv\\Scripts\\python -m ocr.debug.check_ocr_model [image_path_or_url]
"""

import sys

# ocr_engine sets the ocr/.cache env vars before paddleocr is imported.
from ocr.run.ocr_engine import build_ocr

import torch  # noqa: E402

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
