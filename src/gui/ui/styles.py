# -*- coding: utf-8 -*-
"""CUSTOM_CSS — application-wide CSS overrides for the Gradio interface.

Kept as a pure string constant.  Import and pass to ``gr.Blocks(css=...)``.

Usage::

    from src.gui.ui.styles import CUSTOM_CSS
"""

CUSTOM_CSS: str = """
/* ================================================================
   Global Container & Typography
   ================================================================ */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
    font-family: 'Inter', 'Source Sans Pro', sans-serif !important;
}

.comparison-container {
    max-width: 1000px !important;
    margin: 0 auto;
}

body {
    background-color: #F7F8FA !important;
}

/* Remove default Gradio block borders to reduce visual noise */
.gradio-container .form {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Section titles */
.section-title h2, .section-title h3 {
    color: #1A202C !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #E2E8F0 !important;
    padding-bottom: 8px !important;
    margin-bottom: 16px !important;
}

/* ================================================================
   Header Section
   ================================================================ */
.header-section {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    margin-bottom: 24px;
    background: #FFFFFF;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.header-section h1 {
    font-size: 1.5rem !important;
    color: #1A202C !important;
    margin: 0 !important;
    font-weight: 700 !important;
}
.header-section p {
    color: #718096 !important;
    font-size: 0.9rem;
    margin: 4px 0 0 0 !important;
}

/* ================================================================
   Result Cards (Clean & Minimal)
   ================================================================ */
.result-card, .best-model-card, .kpi-card {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
}

/* Hero Prediction Card */
.hero-result-card {
    background: #FFFFFF !important;
    border: 2px solid #F05A28 !important; /* Deep Learning Orange Accent */
    border-radius: 12px !important;
    padding: 32px !important;
    box-shadow: 0 10px 15px -3px rgba(240, 90, 40, 0.1) !important;
    text-align: center;
}

/* KPI Summary Cards (Stripe inspired) */
.kpi-card {
    text-align: center;
    padding: 16px !important;
}
.kpi-title {
    color: #718096 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px !important;
}
.kpi-value {
    color: #1A202C !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

/* ================================================================
   Video Preview
   ================================================================ */
.video-preview video {
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    max-height: 400px;
    width: auto !important;
    margin: 0 auto;
}

/* ================================================================
   Primary Button
   ================================================================ */
.primary-btn {
    background: #F05A28 !important; /* Deep Learning Orange */
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
.primary-btn:hover {
    transform: translateY(-1px) !important;
    background: #EA580C !important; /* Slightly darker on hover */
    box-shadow: 0 4px 12px rgba(240, 90, 40, 0.2) !important;
}

/* ================================================================
   Model Info Box
   ================================================================ */
.model-info-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.6 !important;
    background: #FAFAFC !important;
    border: 2px solid #F05A28 !important; /* Deep Learning Orange */
    color: #4A5568 !important;
}

/* ================================================================
   Language Selector (Compact Toggle)
   ================================================================ */
.language-selector {
    background: #FAFAFC !important;
    border-radius: 6px !important;
    padding: 4px 8px !important;
    border: 1px solid #E2E8F0 !important;
}
.language-selector label {
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    color: #4A5568 !important;
}

/* ================================================================
   Loaders & Spinners
   ================================================================ */
.generating, .loading {
    color: #F05A28 !important;
}
.loader {
    border-color: #F05A28 !important;
    border-bottom-color: transparent !important;
}

/* ================================================================
   Gradio Toast Notifications (Errors/Info)
   ================================================================ */
.toast-wrap {
    top: auto !important;
    bottom: 30px !important;
    right: 30px !important;
    left: auto !important;
    z-index: 9999 !important;
}

.toast-wrap > div {
    background-color: #FFFFFF !important;
    color: #1A202C !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1) !important;
    padding: 16px 20px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

.toast-wrap .error {
    border-left: 4px solid #E53E3E !important;
}

/* ================================================================
   Tables (gr.Dataframe) & Code blocks (force light theme)
   ================================================================ */
.gradio-container table {
    background-color: #FFFFFF !important;
    color: #1A202C !important;
}
.gradio-container th, .gradio-container td {
    background-color: #FFFFFF !important;
    color: #1A202C !important;
    border-color: #E2E8F0 !important;
}
.gradio-container tr:nth-child(even) td {
    background-color: #F7F8FA !important;
}
.gradio-container tr:hover td {
    background-color: #EDF2F7 !important;
}
.gradio-container code, .prose code {
    background-color: #EDF2F7 !important;
    color: #E53E3E !important; /* Muted red for inline code */
    padding: 2px 6px !important;
    border-radius: 4px !important;
    border: 1px solid #E2E8F0 !important;
}

/* ================================================================
   Footer
   ================================================================ */
.academic-footer {
    text-align: center;
    padding: 32px 0;
    margin-top: 48px;
    border-top: 1px solid #E2E8F0;
    color: #718096;
    font-size: 0.85rem;
    line-height: 1.6;
}
.academic-footer strong {
    color: #4A5568;
    font-weight: 600;
}
"""
