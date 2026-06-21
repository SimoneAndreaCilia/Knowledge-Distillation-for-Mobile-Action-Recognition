# -*- coding: utf-8 -*-
"""CUSTOM_CSS — application-wide CSS overrides for the Gradio interface.

Kept as a pure string constant.  Import and pass to ``gr.Blocks(css=...)``.

Usage::

    from src.gui.ui.styles import CUSTOM_CSS
"""

CUSTOM_CSS: str = """
/* ================================================================
   Global Container
   ================================================================ */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
}

/* ================================================================
   Header Section
   ================================================================ */
.header-section {
    text-align: center;
    padding: 20px 0;
    margin-bottom: 10px;
    background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.header-section h1 {
    font-size: 2rem !important;
    background: linear-gradient(90deg, #6C63FF, #00D68F);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px !important;
}
.header-section p {
    color: #aaa !important;
    font-size: 1rem;
}

/* ================================================================
   Result Card
   ================================================================ */
.result-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
}

/* ================================================================
   Model Info Box
   ================================================================ */
.model-info-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.6 !important;
}

/* ================================================================
   Status Message
   ================================================================ */
.status-msg .prose {
    font-size: 1.05rem !important;
}

/* ================================================================
   Comparison Summary
   ================================================================ */
.comparison-summary .prose {
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
}

/* ================================================================
   Video Preview
   ================================================================ */
.video-preview video {
    border-radius: 12px;
    border: 2px solid rgba(108, 99, 255, 0.3);
}

/* ================================================================
   Primary Button
   ================================================================ */
.primary-btn {
    background: linear-gradient(135deg, #6C63FF, #00D68F) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}
.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(108, 99, 255, 0.35) !important;
}

/* ================================================================
   Tab Navigation
   ================================================================ */
.tabs .tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
"""
