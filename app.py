from __future__ import annotations

import json
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

import ui
from shadow_processing import (
    MAX_SIDE,
    as_png,
    limit_size,
    paired_metrics,
    process_document,
    read_rgb,
)

# --------------------------------------------------------------------------- #
# Page config + global styling (single stylesheet injected once per run)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="ShadowScout | Document Restoration",
    page_icon="🔎",
    layout="wide",
)
ui.inject_styles()

ALLOWED_TYPES = ["png", "jpg", "jpeg", "webp", "tif", "tiff"]

with st.sidebar:
    ui.brand_header()
    st.divider()

    ui.group("Processing", "Controls that drive shadow estimation and correction.")
    method = st.selectbox(
        "Correction algorithm",
        [
            "Illumination normalization",
            "Background subtraction",
            "Log-domain Retinex",
            "Shadow-map compensation",
        ],
        help="Classical correction model applied to the estimated illumination field.",
    )
    strength = st.slider("Correction strength", 0.0, 1.0, 0.75, 0.05,
                         help="How strongly the correction is blended into detected shadow regions.")
    sensitivity = st.slider("Shadow sensitivity", 1, 10, 6,
                            help="Higher values detect fainter, subtler illumination differences.")
    scale = st.slider("Illumination field scale", 0.04, 0.30, 0.10, 0.01,
                      help="Relative size of the smooth background estimate. Larger values model broader shading.")

    st.divider()
    ui.group("Image quality", "Optional post-processing of the corrected lightness.")
    denoise = st.slider("Denoising", 0, 10, 0,
                        help="Edge-preserving bilateral denoising applied before estimation.")
    clahe = st.slider("Contrast enhancement", 0.0, 3.0, 0.0, 0.25,
                      help="Local contrast (CLAHE). 0 disables it.")
    sharpen = st.toggle("Sharpen text", False,
                        help="Slight unsharp mask to make strokes crisper.")
    binary = st.toggle("Binarize to black & white", False,
                       help="Produce a high-contrast scan with adaptive thresholding.")

    st.divider()
    ui.group("Advanced", "Fine controls for text retention and binarization.")
    preserve_text = st.toggle("Preserve fine text detail", True,
                              help="Restores high-frequency stroke detail after correction.")
    block = st.select_slider("Binarization window", [15, 21, 31, 41, 51], value=31,
                             help="Neighbourhood size for adaptive thresholding.")
    offset = st.slider("Binarization offset", 0, 20, 10,
                       help="Threshold bias. Larger values keep more pixels white.")

    st.divider()
    st.caption("Tip: inspect the candidate mask before fixing the final settings. "
               "Stamps, dark text and page edges can still confuse this heuristic detection.")


# --------------------------------------------------------------------------- #
# Product header — compact
# --------------------------------------------------------------------------- #
ui.product_header(
    title="ShadowScout",
    subtitle="Document Shadow Removal & Quality Enhancement",
    description=(
        "A classical image-processing tool for unevenly lit document photos. ShadowScout estimates the "
        "page's smooth illumination field, flags candidate shadow regions, and selectively corrects "
        "luminance in the LAB lightness channel while retaining colour — then lets you inspect the "
        "result, compare against a clean reference, and export the output."
    ),
    badge="Research prototype",
)

# --------------------------------------------------------------------------- #
# Upload experience
# --------------------------------------------------------------------------- #
up_col, ref_col = st.columns([1.15, 1], gap="large")
with up_col:
    upload = st.file_uploader(
        "Upload shadowed document",
        type=ALLOWED_TYPES,
        help=f"Accepted: {', '.join(t.upper() for t in ALLOWED_TYPES)}. "
             f"Images are resized so the longest side is at most {MAX_SIDE}px to bound processing time.",
    )
    ui.caption(
        f"Input image · {', '.join(t.upper() for t in ALLOWED_TYPES)} · "
        f"processed up to {MAX_SIDE}px per side"
    )
with ref_col:
    reference_upload = st.file_uploader(
        "Optional clean reference",
        type=ALLOWED_TYPES,
        help="A matching shadow-free image of the same document is required for meaningful "
             "reference-based PSNR / SSIM / MSE evaluation.",
    )
    ui.caption("Reference image · optional · used only for evaluation metrics")

if upload is None:
    st.markdown("### How it works")
    ui.workflow_steps([
        ("1 · Preprocess", "Orient via EXIF and cap the image size for consistent processing."),
        ("2 · Estimate", "Build a smooth illumination field and a candidate shadow mask."),
        ("3 · Correct", "Apply bounded, spatially selective correction in the LAB lightness channel."),
        ("4 · Evaluate", "Inspect the output and compare against a clean reference if you have one."),
    ])
    st.info("Upload a document image above to run the processing pipeline.")
    st.stop()

try:
    source = read_rgb(upload)
    reference = read_rgb(reference_upload) if reference_upload else None
except (OSError, ValueError) as exc:
    st.error(f"Could not open image: {exc}")
    st.stop()


# --------------------------------------------------------------------------- #
# Processing (unchanged pipeline; settings map 1:1 to existing parameters)
# --------------------------------------------------------------------------- #
settings = {
    "method": method,
    "strength": strength,
    "sensitivity": sensitivity,
    "scale": scale,
    "preserve_text": preserve_text,
    "denoise": denoise,
    "clahe": clahe,
    "sharpen": sharpen,
    "binary": binary,
    "block": block,
    "offset": offset,
}

try:
    with st.spinner("Processing document…"):
        result = process_document(source, settings)
except Exception as exc:  # noqa: BLE001 - algorithm failure must be shown, not swallowed
    st.error(f"Processing failed for the selected algorithm: {exc}")
    st.stop()

# --------------------------------------------------------------------------- #
# Analysis cards
# --------------------------------------------------------------------------- #
st.markdown("### Image analysis")
c1, c2, c3, c4 = st.columns(4, gap="medium")

with c1:
    tone = "warn" if result["shadow_detected"] else "ok"
    note = "Candidate shadow regions found" if result["shadow_detected"] else "No broad shadow detected"
    ui.metric_card("Illumination variation", result["label"], note, tone=tone)
with c2:
    ui.metric_card(
        "Candidate mask coverage",
        f'{result["coverage"] * 100:.1f}%',
        "Diagnostic estimate — not % removed",
    )
with c3:
    ui.metric_card(
        "Variation indicator",
        f'{result["severity"]:.2f}',
        "Low-frequency spread ratio (P90−P10)",
    )
with c4:
    ui.metric_card("Processing time", f'{result["elapsed"]:.2f} s', "Full pipeline, one image")

if result["shadow_detected"]:
    st.info(
        "A broad illumination change was detected; the smooth correction map is limited to darker regions. "
        "Inspect the candidate mask and the output, because folds, graphics and page edges can still confuse "
        "this classical heuristic. These figures are diagnostics, not an accuracy score."
    )
else:
    st.warning(
        "No significant broad shadow was detected with the current settings; this does not prove the image is "
        "shadow-free, and no shadow-removal success is claimed. If a visible shadow remains, try adjusting "
        "sensitivity or illumination field scale and inspect the candidate map."
    )


# --------------------------------------------------------------------------- #
# Tabs — preserve Streamlit tabs and all existing content
# --------------------------------------------------------------------------- #
tab1, tab2, tab3, tab4 = st.tabs(["Before / After", "Shadow Analysis", "Evaluation", "Export"])

with tab1:
    a, b = st.columns([1, 1], gap="large")
    with a:
        ui.panel_label("Original")
        st.image(result["original"], use_container_width=True)
        h, w = result["original"].shape[:2]
        ui.caption(f"Input document · {w} × {h} px")
    with b:
        ui.panel_label("Processed result")
        st.image(result["output"], use_container_width=True)
        oh, ow = result["output"].shape[:2]
        ui.caption(f"{method} · {ow} × {oh} px")

    st.caption(
        "Both panels use matching display width and never stretch the document, so aspect ratio is preserved. "
        "Use the expander below for a larger single-image view to inspect small text."
    )
    with st.expander("View processed result at full size"):
        st.image(result["output"], caption="Processed output at full container width.",
                 use_container_width=True)
    with st.expander("View luminance-only intermediate"):
        st.image(result["corrected_luma"], caption="Corrected LAB lightness channel.",
                 use_container_width=True)

with tab2:
    ui.panel_label("Estimated candidate shadow map")
    st.image(result["mask"], use_container_width=True)
    ui.caption("A heuristic map of regions with lower relative luminance. It is an estimate, not ground truth.")

    st.divider()
    ui.panel_label("Estimated illumination field")
    st.image(result["illumination"], use_container_width=True)
    ui.caption("The smooth low-frequency background brightness estimated across the page.")

    st.divider()
    overlay = result["original"].copy()
    selected = result["mask"] > 80
    overlay[selected] = (0.58 * overlay[selected] + 0.42 * np.array([255, 50, 65])).astype(np.uint8)
    ui.panel_label("Candidate regions overlaid in red")
    st.image(overlay, use_container_width=True)

    st.markdown(
        """
**Legend**

- **Original document** — the uploaded image.
- **Red overlay / mask** — *estimated* candidate shadow regions (algorithmic, may include non-shadow areas).
- **Corrected output** — the result shown in the *Before / After* tab.

Shadow masks are algorithmic estimates, **not guaranteed ground truth**. Dark print, stamps, page edges
and graphics can be misread as shadows, so treat the mask as a diagnostic aid rather than an answer.
"""
    )

with tab3:
    ui.panel_label("Descriptive statistics of the corrected output")
    st.caption("These describe the produced image. They are not a shadow-removal accuracy score.")
    gray = cv2.cvtColor(result["corrected_rgb"], cv2.COLOR_RGB2GRAY)
    d1, d2, d3 = st.columns(3, gap="medium")
    with d1:
        ui.metric_card("Mean luminance", f"{np.mean(gray):.1f}", "Average brightness of the output")
    with d2:
        ui.metric_card("Luminance std. deviation", f"{np.std(gray):.1f}", "Overall contrast spread")
    with d3:
        ui.metric_card("Dark-pixel ratio", f"{np.mean(gray < 80) * 100:.1f}%", "Share of pixels below 80")

    st.divider()
    spread_before = result["illumination_spread_before"]
    spread_after = result["illumination_spread_after"]
    reduction = (spread_before - spread_after) / max(spread_before, 1.0) * 100.0
    ui.metric_card(
        "Estimated broad illumination spread (P90 − P10)",
        f"{spread_before:.1f} → {spread_after:.1f} levels",
        f"{reduction:.1f}% lower after correction" if reduction >= 0
        else f"{abs(reduction):.1f}% higher after correction",
        tone="ok" if reduction >= 0 else "warn",
    )
    st.caption(
        "This low-frequency spread is a practical diagnostic only; it is not a substitute for an aligned "
        "clean reference or a shadow-removal accuracy score."
    )

    st.divider()
    ui.panel_label("Reference-based metrics")
    if reference is not None:
        ref = limit_size(reference)
        if ref.shape[:2] != result["output"].shape[:2]:
            st.warning(
                "The reference dimensions differ from the processed image. It is resized for metric "
                "calculation; align the same document carefully, because resizing cannot correct "
                "perspective or framing differences."
            )
        scores = paired_metrics(result["output"], ref)
        m1, m2, m3 = st.columns(3, gap="medium")
        with m1:
            if np.isinf(scores["PSNR_dB"]):
                ui.metric_card("PSNR", "∞", "Identical to reference (no error)")
            else:
                ui.metric_card("PSNR", f'{scores["PSNR_dB"]:.2f} dB', "Higher is better")
        with m2:
            if scores["SSIM"] is None:
                ui.metric_card("SSIM", "Unavailable",
                               "Install scikit-image to compute", unavailable=True)
            else:
                ui.metric_card("SSIM", f'{scores["SSIM"]:.4f}', "Closer to 1 is more similar")
        with m3:
            ui.metric_card("MSE", f'{scores["MSE"]:.2f}', "Lower is better")
        st.caption(
            "Use only a clean reference of the same, aligned document; the metrics are meaningful only then."
        )
    else:
        st.info(
            "No clean reference uploaded. PSNR, SSIM and MSE are not reported because a valid target "
            "image is required for a meaningful reference-based comparison."
        )

with tab4:
    report = {
        "project": "ShadowScout",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_filename": upload.name,
        "algorithm": method,
        "settings": settings,
        "illumination_variation_label": result["label"],
        "candidate_mask_coverage": result["coverage"],
        "variation_indicator": result["severity"],
        "processing_seconds": round(result["elapsed"], 4),
        "illumination_spread_before": result["illumination_spread_before"],
        "illumination_spread_after": result["illumination_spread_after"],
        "reference_metrics_available": reference is not None,
        "limitations": "Classical heuristic image-processing prototype. Mask coverage is not accuracy. "
                       "Evaluate on a paired test dataset before making performance claims.",
    }
    if reference is not None:
        report["reference_metrics"] = paired_metrics(result["output"], limit_size(reference))

    ui.panel_label("Processed outputs")
    e1, e2 = st.columns(2, gap="medium")
    with e1:
        st.download_button(
            "Download processed image",
            as_png(result["output"]),
            "shadowscout_processed.png",
            "image/png",
            use_container_width=True,
            key="dl_processed",
        )
        ui.caption("The corrected/restored document as a PNG.")
    with e2:
        st.download_button(
            "Download shadow map",
            as_png(result["mask"]),
            "shadowscout_shadow_map.png",
            "image/png",
            use_container_width=True,
            key="dl_mask",
        )
        ui.caption("The estimated candidate shadow mask as a PNG (diagnostic).")

    ui.panel_label("Report")
    st.download_button(
        "Download JSON report",
        json.dumps(report, indent=2).encode(),
        "shadowscout_report.json",
        "application/json",
        use_container_width=True,
        key="dl_report",
    )
    ui.caption("A machine-readable record of the settings, diagnostics and (if available) reference metrics.")


ui.footer(
    "ShadowScout · Document Shadow Removal &amp; Quality Enhancement<br>"
    "Classical image-processing prototype inspired by the research survey; not a document-authentication system."
)
