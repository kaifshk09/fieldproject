from __future__ import annotations

import io
import time

import cv2
import numpy as np
from PIL import Image, ImageOps


MAX_SIDE = 2200


def read_rgb(upload) -> np.ndarray:
    with Image.open(upload) as image:
        oriented = ImageOps.exif_transpose(image)
        if min(oriented.size) < 32:
            raise ValueError("Image must be at least 32 × 32 pixels.")
        oriented = oriented.convert("RGB")
        oriented.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        rgb = np.asarray(oriented)
    if rgb.ndim != 3 or rgb.shape[0] < 32 or rgb.shape[1] < 32:
        raise ValueError("Image must be at least 32 × 32 pixels.")
    return rgb


def limit_size(rgb: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1:
        rgb = cv2.resize(
            rgb,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return rgb


def odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def estimate_illumination(luma: np.ndarray, scale: float) -> np.ndarray:
    """Close printed strokes, then smooth the document's low-frequency field."""
    height, width = luma.shape
    short_side = min(height, width)
    field_scale = float(np.clip(scale, 0.04, 0.30))
    kernel_size = odd(short_side * max(field_scale, 0.08))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    background = cv2.morphologyEx(luma, cv2.MORPH_CLOSE, kernel)
    sigma = max(2.0, short_side * max(field_scale, 0.08) * 0.45)
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(background, 1, 255).astype(np.uint8)


def _shadow_confidence(
    illumination: np.ndarray, sensitivity: int
) -> tuple[np.ndarray, float, float]:
    field = illumination.astype(np.float32)
    # The upper field percentile samples bright page background while staying
    # robust to a few glare pixels and to text missed by the closing filter.
    low, high = np.percentile(field, [10, 98])
    target = float(high)
    variation = float((high - low) / max(target, 1.0))
    threshold = float(np.interp(np.clip(sensitivity, 1, 10), [1, 10], [0.16, 0.025]))

    if variation <= max(0.025, threshold * 0.45):
        return np.zeros_like(field), target, variation

    relative_dimming = np.clip((target - field) / max(target, 1.0), 0.0, 1.0)
    confidence = np.clip(
        (relative_dimming - threshold) / max(0.04, 0.28 - threshold),
        0.0,
        1.0,
    )
    short_side = min(illumination.shape)
    confidence = cv2.GaussianBlur(
        confidence, (0, 0), sigmaX=max(2.0, short_side * 0.018)
    )
    return np.clip(confidence, 0.0, 1.0), target, variation


def estimate_shadow_mask(
    illumination: np.ndarray, sensitivity: int
) -> np.ndarray:
    confidence, _, _ = _shadow_confidence(illumination, sensitivity)
    return np.clip(confidence * 255.0, 0, 255).astype(np.uint8)


def correct_luminance(
    luma: np.ndarray,
    illumination: np.ndarray,
    method: str,
    strength: float,
    sensitivity: int,
    preserve_text: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    source = luma.astype(np.float32)
    field = np.maximum(illumination.astype(np.float32), 20.0)
    confidence, target, _ = _shadow_confidence(illumination, sensitivity)
    gain_limit = 1.8
    gain = np.clip(target / field, 1.0, gain_limit)
    if preserve_text:
        correction_source = cv2.GaussianBlur(
            source, (0, 0), sigmaX=1.0, sigmaY=1.0
        )
        detail = source - correction_source
    else:
        correction_source = source
        detail = 0.0

    correction_delta = np.clip(target - field, 0.0, 110.0)
    if method == "Illumination normalization":
        candidate = correction_source * gain
    elif method == "Background subtraction":
        candidate = correction_source + correction_delta
    elif method == "Log-domain Retinex":
        candidate = np.expm1(
            np.log1p(correction_source) - np.log1p(field) + np.log1p(target)
        )
    else:
        candidate = correction_source + field * (gain - 1.0)

    alpha = np.clip(confidence * float(np.clip(strength, 0.0, 1.0)), 0.0, 1.0)
    if preserve_text:
        corrected = correction_source * (1.0 - alpha) + candidate * alpha + detail
    else:
        corrected = source * (1.0 - alpha) + candidate * alpha
    # Bound only newly corrected dark areas to the estimated bright-page
    # reference; retain any input highlight that was already brighter.
    ceiling = np.maximum(source, min(250.0, target + 2.0))
    corrected = np.minimum(corrected, ceiling)
    return (
        np.clip(corrected, 0, 255).astype(np.uint8),
        np.clip(alpha * 255.0, 0, 255).astype(np.uint8),
    )


def process_document(source: np.ndarray, settings: dict) -> dict:
    start = time.perf_counter()
    rgb = limit_size(source)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luma, a_channel, b_channel = cv2.split(lab)
    detection_luma = luma

    denoise = int(settings.get("denoise", 0))
    if denoise > 0:
        detection_luma = cv2.bilateralFilter(
            luma, 7, denoise * 3, denoise * 3
        )

    illumination = estimate_illumination(
        detection_luma, float(settings.get("scale", 0.10))
    )
    mask = estimate_shadow_mask(
        illumination, int(settings.get("sensitivity", 6))
    )
    corrected_luma, alpha = correct_luminance(
        detection_luma,
        illumination,
        str(settings.get("method", "Shadow-map compensation")),
        float(settings.get("strength", 0.75)),
        int(settings.get("sensitivity", 6)),
        bool(settings.get("preserve_text", True)),
    )

    clahe = float(settings.get("clahe", 0.0))
    if clahe > 0:
        corrected_luma = cv2.createCLAHE(
            clipLimit=clahe, tileGridSize=(8, 8)
        ).apply(corrected_luma)
    if settings.get("sharpen", False):
        blurred = cv2.GaussianBlur(corrected_luma, (0, 0), 1.0)
        corrected_luma = cv2.addWeighted(corrected_luma, 1.12, blurred, -0.12, 0)

    if (
        not np.any(alpha)
        and denoise == 0
        and clahe == 0
        and not settings.get("sharpen", False)
    ):
        corrected_rgb = rgb.copy()
    else:
        corrected_lab = cv2.merge([corrected_luma, a_channel, b_channel])
        corrected_rgb = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB)
    if settings.get("binary", False):
        block = min(odd(int(settings.get("block", 31))), min(corrected_luma.shape) - 1)
        if block % 2 == 0:
            block -= 1
        block = max(3, block)
        output = cv2.adaptiveThreshold(
            corrected_luma,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            int(settings.get("offset", 10)),
        )
    else:
        output = corrected_rgb

    confidence, _, variation = _shadow_confidence(
        illumination, int(settings.get("sensitivity", 6))
    )
    after_illumination = estimate_illumination(
        corrected_luma, float(settings.get("scale", 0.10))
    )
    after_spread = float(
        np.percentile(after_illumination, 90)
        - np.percentile(after_illumination, 10)
    )
    before_spread = float(
        np.percentile(illumination, 90) - np.percentile(illumination, 10)
    )
    label = (
        "Low variation"
        if variation < 0.06
        else "Moderate variation"
        if variation < 0.18
        else "High variation"
    )
    return {
        "original": rgb,
        "working_luma": detection_luma,
        "illumination": illumination,
        "mask": mask,
        "alpha": alpha,
        "corrected_luma": corrected_luma,
        "corrected_rgb": corrected_rgb,
        "output": output,
        "coverage": float(np.mean(confidence > 0.15)),
        "severity": variation,
        "label": label,
        "illumination_spread_before": before_spread,
        "illumination_spread_after": after_spread,
        "shadow_detected": bool(np.any(mask > 0)),
        "elapsed": time.perf_counter() - start,
    }


def paired_metrics(output: np.ndarray, reference: np.ndarray) -> dict:
    output_rgb = np.asarray(output)
    reference_rgb = np.asarray(reference)
    if output_rgb.ndim == 2:
        output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_GRAY2RGB)
    if reference_rgb.ndim == 2:
        reference_rgb = cv2.cvtColor(reference_rgb, cv2.COLOR_GRAY2RGB)
    if output_rgb.shape[:2] != reference_rgb.shape[:2]:
        reference_rgb = cv2.resize(
            reference_rgb,
            (output_rgb.shape[1], output_rgb.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    if output_rgb.shape[2] == 4:
        output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_RGBA2RGB)
    if reference_rgb.shape[2] == 4:
        reference_rgb = cv2.cvtColor(reference_rgb, cv2.COLOR_RGBA2RGB)

    output_gray = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2GRAY)
    reference_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    output_float = output_gray.astype(np.float64)
    reference_float = reference_gray.astype(np.float64)
    mse = float(np.mean((output_float - reference_float) ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255**2 / mse))
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        ssim = None
    else:
        ssim = float(
            structural_similarity(
                reference_gray, output_gray, data_range=255
            )
        )
    return {"MSE": mse, "PSNR_dB": psnr, "SSIM": ssim}


def as_png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
