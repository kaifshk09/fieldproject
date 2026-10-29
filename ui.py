"""Shared UI layer for ShadowScout.

Centralises the design tokens, the single global stylesheet and small reusable
components so styling is not duplicated across the app and stays consistent
across Streamlit reruns.

Design intent: premium, research-grade, restrained. Off-white surface, deep
navy text, muted slate secondary text, one indigo accent, thin neutral borders.
All colours were chosen for readable contrast (WCAG AA where practical).
"""
from __future__ import annotations

from typing import Optional, Sequence

import streamlit as st

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #
TOKENS = {
    # surfaces
    "bg": "#f6f7f9",          # soft cool off-white page background
    "surface": "#ffffff",      # cards / panels
    "surface_alt": "#f1f3f7",  # subtle inset panels
    "sidebar": "#0f1729",      # deep navy sidebar
    "sidebar_elev": "#16203a",
    # text
    "text": "#141b2d",         # deep navy, primary
    "muted": "#5a6478",        # muted slate, secondary text
    "on_dark": "#eef2fa",
    "on_dark_muted": "#a4b0c6",
    # accent + semantics
    "accent": "#3556e0",       # refined indigo / electric blue
    "accent_soft": "#eaefff",
    "success": "#1f8a52",
    "success_soft": "#e6f4ec",
    "warning": "#b3760a",
    "warning_soft": "#fdf3e0",
    "danger": "#c0392b",
    "danger_soft": "#fbeceb",
    # lines
    "border": "#e2e6ee",
    "border_strong": "#cfd6e3",
    "radius": "14px",
    "radius_sm": "10px",
}

# --------------------------------------------------------------------------- #
# Global stylesheet (single source of truth)
# --------------------------------------------------------------------------- #
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

/* ---- base ---------------------------------------------------------------- */
html, body, [class*="css"], .stApp {{
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
    color: {TOKENS['text']};
}}
.stApp {{ background: {TOKENS['bg']}; }}
.block-container {{
    max-width: 1300px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}}
h1, h2, h3, h4 {{ font-family: 'Manrope', 'Inter', sans-serif; color: {TOKENS['text']}; }}
h1 {{ letter-spacing: -.02em; }}
p, span, label, li, div {{ line-height: 1.55; }}

/* ---- sidebar ------------------------------------------------------------- */
section[data-testid="stSidebar"] {{
    background: {TOKENS['sidebar']};
    border-right: 1px solid #0a1120;
}}
section[data-testid="stSidebar"] * {{ color: {TOKENS['on_dark']}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}

/* Streamlit widget labels inside the sidebar: force readable, high-contrast */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] .stMarkdown p {{
    color: {TOKENS['on_dark']} !important;
    font-weight: 600;
    font-size: .82rem;
}}
/* slider numeric ticks + helper captions stay legible but subordinate */
section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {{
    color: {TOKENS['on_dark_muted']} !important;
}}
/* slider current-value bubble + thumbs use the accent so values pop */
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stThumbValue"],
section[data-testid="stSidebar"] [data-testid="stTickBarMinMax"] {{
    color: #c9d6ff !important;
    font-weight: 600;
}}
/* selection controls */
section[data-testid="stSidebar"] div[data-baseweb="select"] *,
section[data-testid="stSidebar"] div[data-baseweb="input"] input {{
    color: {TOKENS['on_dark']} !important;
    background: {TOKENS['sidebar_elev']} !important;
}}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
    border-color: #2b3554 !important;
}}
section[data-testid="stSidebar"] hr {{ border-color: #26314e; opacity: .7; }}

/* group headings in the sidebar */
.ss-group {{
    font-family: 'Manrope', sans-serif;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: #8fa2c9;
    margin: .4rem 0 .35rem;
}}
.ss-help {{
    color: #9aa8c4;
    font-size: .74rem;
    line-height: 1.5;
    margin: -.15rem 0 .5rem;
}}

/* ---- brand --------------------------------------------------------------- */
.ss-brand {{ display:flex; align-items:center; gap:.6rem; margin-bottom:.15rem; }}
.ss-brand .mark {{
    width: 30px; height: 30px; border-radius: 8px;
    background: linear-gradient(150deg, #3556e0, #2b3fb0);
    display:flex; align-items:center; justify-content:center;
    color:#fff; font-weight:800; font-size:16px;
    box-shadow: 0 2px 8px rgba(53,86,224,.35);
}}
.ss-brand .word {{
    font-family:'Manrope',sans-serif; font-weight:800; font-size:1.18rem;
    letter-spacing:-.02em; color:{TOKENS['on_dark']};
}}
.ss-sub {{
    color:{TOKENS['on_dark_muted']}; font-size:.66rem; font-weight:700;
    letter-spacing:.18em; text-transform:uppercase; margin-top:.15rem;
}}

/* ---- product header ------------------------------------------------------ */
.ss-header {{
    background:{TOKENS['surface']};
    border:1px solid {TOKENS['border']};
    border-radius:{TOKENS['radius']};
    padding:1.1rem 1.35rem;
    margin-bottom:1.1rem;
    box-shadow: 0 1px 2px rgba(20,27,45,.04);
}}
.ss-header .row {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; }}
.ss-header h1 {{ font-size:1.55rem; margin:0; letter-spacing:-.02em; }}
.ss-header .tag {{ color:{TOKENS['muted']}; font-size:.86rem; margin:.2rem 0 0; }}
.ss-header p.desc {{ color:{TOKENS['muted']}; font-size:.86rem; margin:.6rem 0 0; max-width:70ch; }}
.ss-badge {{
    display:inline-flex; align-items:center; gap:.4rem;
    background:{TOKENS['accent_soft']}; color:{TOKENS['accent']};
    border:1px solid #cfdaff; border-radius:999px;
    padding:.28rem .7rem; font-size:.72rem; font-weight:700; white-space:nowrap;
}}
.ss-badge .dot {{ width:6px; height:6px; border-radius:50%; background:{TOKENS['accent']}; }}

/* ---- cards --------------------------------------------------------------- */
.ss-card {{
    background:{TOKENS['surface']};
    border:1px solid {TOKENS['border']};
    border-radius:{TOKENS['radius_sm']};
    padding:.85rem 1rem;
    height:100%;
    box-shadow: 0 1px 2px rgba(20,27,45,.03);
}}
.ss-card .k {{
    font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
    color:{TOKENS['muted']};
}}
.ss-card .v {{
    font-family:'Manrope',sans-serif; font-size:1.5rem; font-weight:800;
    color:{TOKENS['text']}; margin:.25rem 0 .1rem; letter-spacing:-.02em;
}}
.ss-card .s {{ font-size:.78rem; color:{TOKENS['muted']}; }}
.ss-card .s.ok {{ color:{TOKENS['success']}; }}
.ss-card .s.warn {{ color:{TOKENS['warning']}; }}
.ss-card .s.bad {{ color:{TOKENS['danger']}; }}
.ss-card.na .v {{ color:{TOKENS['muted']}; font-size:1.1rem; font-weight:700; }}

/* small pill status */
.ss-pill {{
    display:inline-block; padding:.15rem .55rem; border-radius:999px;
    font-size:.72rem; font-weight:700;
}}
.ss-pill.ok {{ background:{TOKENS['success_soft']}; color:{TOKENS['success']}; }}
.ss-pill.warn {{ background:{TOKENS['warning_soft']}; color:{TOKENS['warning']}; }}
.ss-pill.neutral {{ background:{TOKENS['surface_alt']}; color:{TOKENS['muted']}; }}

/* section label above panels */
.ss-label {{
    font-size:.72rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase;
    color:{TOKENS['muted']}; margin:.1rem 0 .45rem;
}}
.ss-cap {{ font-size:.76rem; color:{TOKENS['muted']}; margin-top:.35rem; }}

/* ---- tabs ---------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {{
    gap:.35rem; border-bottom:1px solid {TOKENS['border']};
}}
.stTabs [data-baseweb="tab"] {{
    height:auto; padding:.55rem .95rem; border-radius:10px 10px 0 0;
    color:{TOKENS['muted']}; font-weight:600; font-size:.9rem;
}}
.stTabs [aria-selected="true"] {{
    color:{TOKENS['accent']} !important;
    background:{TOKENS['accent_soft']};
}}
.stTabs [data-baseweb="tab-highlight"] {{ background:{TOKENS['accent']}; }}
.stTabs [data-baseweb="tab-border"] {{ background:{TOKENS['border']}; }}

/* ---- uploaders ----------------------------------------------------------- */
[data-testid="stFileUploaderDropzone"] {{
    background:{TOKENS['surface']} !important;
    border:1.5px dashed {TOKENS['border_strong']} !important;
    border-radius:{TOKENS['radius_sm']} !important;
    padding:1rem !important;
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color:{TOKENS['accent']} !important; }}
/* the "Browse files" button + all uploader text must be readable */
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] label,
[data-testid="stFileUploaderDropzone"] div {{
    color:{TOKENS['text']} !important;
}}
[data-testid="stFileUploaderDropzone"] small {{ color:{TOKENS['muted']} !important; }}
[data-testid="stFileUploaderDropzone"] button {{
    border:1px solid {TOKENS['border_strong']} !important;
    color:{TOKENS['text']} !important; background:{TOKENS['surface_alt']} !important;
    border-radius:8px !important;
}}
/* main-area widget labels (upload label etc.) */
.stApp label, .stApp label p {{ color:{TOKENS['text']} !important; font-weight:600; }}
[data-testid="stWidgetLabel"] p {{ color:{TOKENS['text']} !important; font-weight:600; font-size:.9rem; }}
/* captions / helper text in main area */
.stApp [data-testid="stCaptionContainer"] *, .stApp .stCaption, .stApp small {{
    color:{TOKENS['muted']} !important;
}}

/* ---- images -------------------------------------------------------------- */
.ss-figure img {{ border:1px solid {TOKENS['border']}; border-radius:{TOKENS['radius_sm']}; }}

/* ---- buttons / downloads ------------------------------------------------- */
.stButton button, .stDownloadButton button {{
    border-radius:10px !important; font-weight:700 !important; min-height:2.6rem;
}}
.stDownloadButton button {{
    background:{TOKENS['accent']} !important; color:#fff !important; border:1px solid {TOKENS['accent']} !important;
}}
.stDownloadButton button:hover {{ background:#2b46c4 !important; }}

/* ---- alerts -------------------------------------------------------------- */
[data-testid="stAlert"] p, .stAlert p {{ color:{TOKENS['text']} !important; }}

/* ---- footer -------------------------------------------------------------- */
.ss-footer {{
    text-align:center; color:{TOKENS['muted']}; font-size:.78rem;
    padding:1.1rem 0 .4rem; border-top:1px solid {TOKENS['border']}; margin-top:1.4rem;
}}

/* ---- responsive ---------------------------------------------------------- */
@media (max-width: 900px) {{
    .block-container {{ padding-left:.8rem; padding-right:.8rem; }}
    .ss-header {{ padding:.9rem 1rem; }}
    .ss-header h1 {{ font-size:1.3rem; }}
    .ss-card .v {{ font-size:1.3rem; }}
}}
@media (max-width: 640px) {{
    .ss-card .v {{ font-size:1.2rem; }}
    .stTabs [data-baseweb="tab"] {{ padding:.5rem .6rem; font-size:.82rem; }}
}}
</style>
"""


def inject_styles() -> None:
    """Inject the single global stylesheet. Call once per run."""
    st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Reusable components
# --------------------------------------------------------------------------- #
def brand_header() -> None:
    """Sidebar wordmark + subtitle + divider."""
    st.markdown(
        """
        <div class="ss-brand">
            <div class="mark">S</div>
            <div>
                <div class="word">ShadowScout</div>
            </div>
        </div>
        <div class="ss-sub">Document Image Restoration</div>
        """,
        unsafe_allow_html=True,
    )


def group(title: str, help_text: Optional[str] = None) -> None:
    """Sidebar group heading with optional helper line."""
    html = f'<div class="ss-group">{title}</div>'
    if help_text:
        html += f'<div class="ss-help">{help_text}</div>'
    st.markdown(html, unsafe_allow_html=True)


def product_header(title: str, subtitle: str, description: str, badge: Optional[str] = None) -> None:
    """Compact product header card."""
    badge_html = (
        f'<span class="ss-badge"><span class="dot"></span>{badge}</span>'
        if badge
        else ""
    )
    st.markdown(
        f"""
        <div class="ss-header">
            <div class="row">
                <div>
                    <h1>{title}</h1>
                    <div class="tag">{subtitle}</div>
                </div>
                {badge_html}
            </div>
            <p class="desc">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(key: str, value: str, note: str = "", tone: str = "", unavailable: bool = False) -> None:
    """Compact metric card.

    tone: one of "", "ok", "warn", "bad" applied to the note text.
    """
    css = "ss-card na" if unavailable else "ss-card"
    note_html = f'<div class="s {tone}">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="{css}">
            <div class="k">{key}</div>
            <div class="v">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_label(text: str) -> None:
    st.markdown(f'<div class="ss-label">{text}</div>', unsafe_allow_html=True)


def caption(text: str) -> None:
    st.markdown(f'<div class="ss-cap">{text}</div>', unsafe_allow_html=True)


def footer(text: str) -> None:
    st.markdown(f'<div class="ss-footer">{text}</div>', unsafe_allow_html=True)


def workflow_steps(steps: Sequence[tuple[str, str]]) -> None:
    """Render the empty-state workflow as consistent, readable cards."""
    cols = st.columns(len(steps))
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="ss-card"><div class="k">{title}</div>'
                f'<div class="s" style="margin-top:.3rem">{desc}</div></div>',
                unsafe_allow_html=True,
            )
