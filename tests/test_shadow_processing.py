from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from shadow_processing import paired_metrics, process_document, read_rgb
from tests.generate_samples import generate_samples


SAMPLES = generate_samples()
DEFAULTS = {
    "method": "Shadow-map compensation",
    "strength": 0.85,
    "sensitivity": 6,
    "scale": 0.10,
    "preserve_text": True,
    "denoise": 0,
    "clahe": 0.0,
    "sharpen": False,
    "binary": False,
    "block": 31,
    "offset": 10,
}


def load_sample(name: str) -> np.ndarray:
    return np.asarray(Image.open(SAMPLES / name).convert("RGB"))


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def mse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean((gray(first).astype(np.float64) - gray(second)) ** 2))


class ShadowProcessingTests(unittest.TestCase):
    def test_clean_document_is_not_needlessly_corrected(self) -> None:
        clean = load_sample("clean_document.png")
        result = process_document(clean, DEFAULTS)
        self.assertFalse(result["shadow_detected"])
        self.assertEqual(result["coverage"], 0.0)
        self.assertTrue(np.array_equal(result["output"], clean))

    def test_supplied_shadowed_photo_improves_against_clean_reference(self) -> None:
        clean = np.asarray(
            Image.open(PROJECT_ROOT / "demo_clean_reference.png").convert("RGB")
        )
        shadowed = np.asarray(
            Image.open(PROJECT_ROOT / "demo_shadowed_document.png").convert("RGB")
        )
        result = process_document(shadowed, DEFAULTS)
        before = paired_metrics(shadowed, clean)
        after = paired_metrics(result["output"], clean)
        self.assertLess(after["MSE"], before["MSE"])
        self.assertGreater(after["PSNR_dB"], before["PSNR_dB"])
        if before["SSIM"] is not None and after["SSIM"] is not None:
            self.assertGreater(after["SSIM"], before["SSIM"])

    def test_soft_and_hard_shadows_improve_against_clean_reference(self) -> None:
        clean = load_sample("clean_document.png")
        for filename in ("large_soft_shadow.png", "hard_edged_shadow.png"):
            with self.subTest(sample=filename):
                shadowed = load_sample(filename)
                result = process_document(shadowed, DEFAULTS)
                self.assertLess(
                    mse(result["output"], clean),
                    mse(shadowed, clean) * 0.90,
                )

    def test_synthetic_shadow_improves_reference_metrics(self) -> None:
        clean = load_sample("clean_document.png")
        shadowed = load_sample("large_soft_shadow.png")
        result = process_document(shadowed, DEFAULTS)

        before = paired_metrics(shadowed, clean)
        after = paired_metrics(result["output"], clean)

        self.assertTrue(result["shadow_detected"])
        self.assertLess(after["MSE"], before["MSE"])
        self.assertGreater(after["PSNR_dB"], before["PSNR_dB"])
        if before["SSIM"] is not None and after["SSIM"] is not None:
            self.assertGreater(after["SSIM"], before["SSIM"])

    def test_strength_and_sensitivity_change_processing(self) -> None:
        shadowed = load_sample("large_soft_shadow.png")
        no_correction = process_document(
            shadowed, {**DEFAULTS, "strength": 0.0}
        )
        stronger_correction = process_document(
            shadowed, {**DEFAULTS, "strength": 1.0}
        )
        low_sensitivity = process_document(
            shadowed, {**DEFAULTS, "sensitivity": 1}
        )
        high_sensitivity = process_document(
            shadowed, {**DEFAULTS, "sensitivity": 10}
        )

        self.assertFalse(
            np.array_equal(no_correction["output"], stronger_correction["output"])
        )
        self.assertFalse(
            np.array_equal(low_sensitivity["output"], high_sensitivity["output"])
        )
        self.assertLess(
            stronger_correction["illumination_spread_after"],
            no_correction["illumination_spread_after"],
        )
        self.assertGreater(high_sensitivity["coverage"], low_sensitivity["coverage"])

    def test_mild_uneven_lighting_does_not_get_worse(self) -> None:
        clean = load_sample("clean_document.png")
        uneven = load_sample("uneven_lighting.png")
        result = process_document(uneven, DEFAULTS)
        self.assertLess(mse(result["output"], clean), mse(uneven, clean))

    def test_thin_text_edges_remain_readable_after_shadow_correction(self) -> None:
        page = load_sample("small_thin_text.png")
        height, width = page.shape[:2]
        shadow_field = np.tile(np.linspace(0.62, 1.0, width), (height, 1))
        shadowed = np.clip(page * shadow_field[..., None], 0, 255).astype(np.uint8)
        preserved = process_document(shadowed, DEFAULTS)
        unprotected = process_document(
            shadowed, {**DEFAULTS, "preserve_text": False}
        )
        roi = (slice(65, 370), slice(10, width - 10))
        input_edges = cv2.Laplacian(gray(shadowed)[roi], cv2.CV_32F)
        preserved_edges = cv2.Laplacian(gray(preserved["output"])[roi], cv2.CV_32F)
        unprotected_edges = cv2.Laplacian(gray(unprotected["output"])[roi], cv2.CV_32F)
        preserved_energy = float(np.mean(np.abs(preserved_edges)))
        unprotected_energy = float(np.mean(np.abs(unprotected_edges)))
        clean_gray = gray(page)
        corrected_gray = gray(preserved["output"])
        text_pixels = clean_gray < 180
        paper_pixels = clean_gray > 220
        text_contrast = float(
            np.median(corrected_gray[paper_pixels])
            - np.median(corrected_gray[text_pixels])
        )

        self.assertGreater(preserved_energy, 0.0)
        self.assertGreaterEqual(preserved_energy, float(np.mean(np.abs(input_edges))) * 0.75)
        self.assertGreaterEqual(preserved_energy, unprotected_energy * 0.95)
        self.assertGreater(text_contrast, 60.0)
        self.assertGreater(float(np.mean(np.abs(input_edges))), 0.0)

    def test_methods_metrics_and_file_reader(self) -> None:
        sample = load_sample("large_soft_shadow.png")
        clean = load_sample("clean_document.png")
        for method in (
            "Illumination normalization",
            "Background subtraction",
            "Log-domain Retinex",
            "Shadow-map compensation",
        ):
            with self.subTest(method=method):
                result = process_document(sample, {**DEFAULTS, "method": method})
                self.assertEqual(result["output"].shape, sample.shape)
                self.assertEqual(result["output"].dtype, np.uint8)
                self.assertTrue(np.isfinite(result["output"]).all())

        metrics = paired_metrics(sample, clean)
        self.assertGreaterEqual(metrics["MSE"], 0.0)
        self.assertTrue(np.isfinite(metrics["PSNR_dB"]))
        if metrics["SSIM"] is not None:
            self.assertTrue(-1.0 <= metrics["SSIM"] <= 1.0)
        baseline_mse = mse(sample, clean)
        for method in (
            "Illumination normalization",
            "Background subtraction",
            "Log-domain Retinex",
            "Shadow-map compensation",
        ):
            with self.subTest(method=method):
                result = process_document(sample, {**DEFAULTS, "method": method})
                self.assertLess(mse(result["output"], clean), baseline_mse)
        loaded = read_rgb(SAMPLES / "clean_document.png")
        self.assertEqual(loaded.shape, clean.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
