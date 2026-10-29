from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


def generate_samples(output_dir: Path | None = None) -> Path:
    output_dir = output_dir or Path(__file__).resolve().parent / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = np.asarray(
        Image.open(PROJECT_ROOT / "demo_clean_reference.png").convert("RGB")
    )
    height, width = clean.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    save_rgb(output_dir / "clean_document.png", clean)

    soft_distance = ((xx - width * 0.68) / (width * 0.48)) ** 2 + (
        (yy - height * 0.48) / (height * 0.62)
    ) ** 2
    soft_field = 1.0 - 0.48 * np.exp(-soft_distance * 1.5)
    save_rgb(output_dir / "large_soft_shadow.png", clean * soft_field[..., None])

    polygon = np.array(
        [
            [int(width * 0.43), 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [int(width * 0.67), height - 1],
        ],
        dtype=np.int32,
    )
    hard_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(hard_mask, [polygon], 255)
    hard_mask = cv2.GaussianBlur(hard_mask, (0, 0), sigmaX=1.2).astype(np.float32) / 255
    hard_field = 1.0 - 0.43 * hard_mask
    save_rgb(output_dir / "hard_edged_shadow.png", clean * hard_field[..., None])

    uneven_field = 0.82 + 0.13 * (xx / max(width - 1, 1)) + 0.05 * (
        yy / max(height - 1, 1)
    )
    save_rgb(output_dir / "uneven_lighting.png", clean * uneven_field[..., None])

    text_page = np.full((height, width, 3), 247, dtype=np.uint8)
    lines = [
        "Thin text: i l ! . , : ; 1",
        "Fine strokes remain visible",
        "Small type 0123456789",
        "Punctuation: A, B. C; D!",
        "The quick brown fox jumps",
        "Document edges and details",
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            text_page,
            text,
            (20, 90 + index * 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
    save_rgb(output_dir / "small_thin_text.png", text_page)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate deterministic document test images.")
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args()
    print(generate_samples(arguments.output_dir))
