import streamlit as st
import pandas as pd
import plotly.express as px

from services.claim_analysis import (
    extract_claim,
    analyze_evidence
)

from services.prediction_service import (
    predict_content
)

from services.evidence_service import (
    search_evidence
)

from services.credibility_service import (
    rank_sources
)

from services.final_decision_service import (
    make_final_decision
)

from services.history_service import (
    save_detection_history,
    load_detection_history
)

from services.media_analysis_service import (
    extract_text_from_image,
    transcribe_audio_video
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Fake Content Detection",
    page_icon="🔎",
    layout="wide"
)


# ==================================================
# THEME — "EVIDENCE DOSSIER" DESIGN SYSTEM
# ==================================================
#
# Palette:  ink #0B1120 (base) / panel #131D33 / panel-hi #1C2A48
#           gold #C9A227 (single accent, used sparingly)
#           verified #2F9E6D · contested #E5484D · unclear #C9A227
# Type:     Playfair Display for headlines (press/dossier authority),
#           Inter for body and data.
# Depth:    layered, navy-tinted shadows that vary by hierarchy —
#           not one repeated grey card shadow everywhere.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --ink: #0B1120;
    --panel: #131D33;
    --panel-hi: #1C2A48;
    --panel-line: rgba(201,162,39,0.14);
    --gold: #C9A227;
    --gold-soft: rgba(201,162,39,0.16);
    --verified: #2F9E6D;
    --contested: #E5484D;
    --text: #E7EAF3;
    --muted: #8B93AC;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, #142140 0%, transparent 45%),
        radial-gradient(circle at 100% 30%, #10182c 0%, transparent 55%),
        var(--ink);
    color: var(--text);
}

/* ---------- Hero header ---------- */
.dossier-hero {
    padding: 2.1rem 2.4rem 1.9rem 2.4rem;
    margin-bottom: 1.6rem;
    border-radius: 18px;
    background: linear-gradient(155deg, #16213E 0%, #0E1729 100%);
    border: 1px solid var(--panel-line);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.04) inset,
        0 24px 48px -18px rgba(0,0,0,0.65),
        0 2px 6px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
}
.dossier-hero::before {
    content: "";
    position: absolute;
    top: -60%; right: -10%;
    width: 320px; height: 320px;
    background: radial-gradient(circle, var(--gold-soft) 0%, transparent 70%);
    pointer-events: none;
}
.dossier-hero .kicker {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--gold);
    margin: 0 0 0.5rem 0;
    letter-spacing: 0.02em;
}
.dossier-hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 0.65rem 0;
    line-height: 1.15;
}
.dossier-hero p {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: var(--muted);
    max-width: 640px;
    margin: 0;
    line-height: 1.55;
}
.dossier-hero .gold-rule {
    width: 54px; height: 3px;
    background: var(--gold);
    border-radius: 2px;
    margin: 1rem 0 1.1rem 0;
}

/* ---------- Section headers (h2/h3 from st.header/st.subheader) ---------- */
h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #F2F4FA !important;
    font-weight: 700 !important;
}
h2 { font-size: 1.55rem !important; }
h3 { font-size: 1.22rem !important; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1526 0%, #0B1120 100%);
    border-right: 1px solid var(--panel-line);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--gold) !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] span {
    color: var(--muted) !important;
}

/* ---------- Buttons: tactile 3D press ---------- */
.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #1A1200;
    background: linear-gradient(180deg, #E4C45C 0%, var(--gold) 100%);
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.6rem;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.5) inset,
        0 6px 0 #8A6C15,
        0 10px 18px -6px rgba(201,162,39,0.5);
    transition: transform 0.08s ease, box-shadow 0.08s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.5) inset,
        0 7px 0 #8A6C15,
        0 14px 22px -6px rgba(201,162,39,0.55);
    color: #1A1200;
}
.stButton > button:active {
    transform: translateY(4px);
    box-shadow: 0 1px 0 rgba(255,255,255,0.3) inset, 0 2px 0 #8A6C15;
}

/* ---------- Metrics: embossed plates ---------- */
[data-testid="stMetric"] {
    background: linear-gradient(155deg, var(--panel-hi) 0%, var(--panel) 100%);
    border: 1px solid var(--panel-line);
    border-radius: 14px;
    padding: 1rem 1.1rem 0.9rem 1.1rem;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.05) inset,
        0 14px 26px -14px rgba(0,0,0,0.7);
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stMetricValue"] {
    color: #F2F4FA !important;
    font-family: 'Playfair Display', serif !important;
}

/* ---------- Alerts, mapped to verdict semantics ---------- */
div[data-testid="stAlertContentSuccess"], .stSuccess {
    background: rgba(47,158,109,0.12) !important;
    border-left: 3px solid var(--verified) !important;
}
div[data-testid="stAlertContentError"], .stError {
    background: rgba(229,72,77,0.12) !important;
    border-left: 3px solid var(--contested) !important;
}
div[data-testid="stAlertContentWarning"], .stWarning {
    background: rgba(201,162,39,0.12) !important;
    border-left: 3px solid var(--gold) !important;
}
div[data-testid="stAlertContentInfo"], .stInfo {
    background: rgba(139,147,172,0.12) !important;
    border-left: 3px solid var(--muted) !important;
}
div[data-testid^="stAlert"] {
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ---------- Expanders, containers with borders ---------- */
div[data-testid="stExpander"] {
    background: var(--panel);
    border: 1px solid var(--panel-line) !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 22px -14px rgba(0,0,0,0.6);
}

/* ---------- Text area / file uploader / radio / text input ---------- */
.stTextArea textarea, .stTextInput input {
    background-color: var(--panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--panel-line) !important;
    border-radius: 10px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px var(--gold-soft) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--panel) !important;
    border: 1.5px dashed var(--panel-line) !important;
    border-radius: 12px !important;
}

div[role="radiogroup"] {
    background: var(--panel);
    padding: 0.35rem;
    border-radius: 12px;
    border: 1px solid var(--panel-line);
    box-shadow: 0 8px 18px -12px rgba(0,0,0,0.6) inset;
}
div[role="radiogroup"] label {
    background: transparent;
    border-radius: 8px;
    padding: 0.3rem 0.7rem;
    color: var(--muted) !important;
}

/* ---------- Dataframe ---------- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--panel-line);
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- Divider ---------- */
hr {
    border-color: var(--panel-line) !important;
    margin: 1.6rem 0 !important;
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--ink); }
::-webkit-scrollbar-thumb { background: var(--panel-hi); border-radius: 6px; }

/* ---------- Film-grain noise overlay ---------- */
.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 9999;
    opacity: 0.035;
    mix-blend-mode: overlay;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

/* ---------- Pop-in entry animation ---------- */
@keyframes popIn {
    0%   { opacity: 0; transform: translateY(14px) scale(0.985); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
}
.dossier-hero {
    animation: popIn 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}
section[data-testid="stSidebar"] > div {
    animation: popIn 0.6s cubic-bezier(0.22, 1, 0.36, 1) 0.1s both;
}
.block-container > div > div > div {
    animation: popIn 0.6s cubic-bezier(0.22, 1, 0.36, 1) 0.18s both;
}

/* ---------- Neo-glassmorphism card (used via st.container(key=...)) ---------- */
.glass-card {
    background: linear-gradient(155deg, rgba(28,42,72,0.55) 0%, rgba(19,29,51,0.55) 100%);
    backdrop-filter: blur(12px) saturate(180%);
    -webkit-backdrop-filter: blur(12px) saturate(180%);
    border: 1px solid rgba(201,162,39,0.18);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    box-shadow:
        0 20px 40px rgba(0,0,0,0.4),
        inset 0 1px 0 rgba(255,255,255,0.08);
    transition: transform 0.25s cubic-bezier(0.22, 1, 0.36, 1),
                box-shadow 0.25s cubic-bezier(0.22, 1, 0.36, 1);
    animation: popIn 0.6s cubic-bezier(0.22, 1, 0.36, 1) 0.12s both;
}
.glass-card:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow:
        0 28px 54px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.12),
        0 0 0 1px rgba(201,162,39,0.3);
}

/* Target Streamlit's native container(key="...") output directly */
div[class*="st-key-"] {
    background: linear-gradient(155deg, rgba(28,42,72,0.55) 0%, rgba(19,29,51,0.55) 100%);
    backdrop-filter: blur(12px) saturate(180%);
    -webkit-backdrop-filter: blur(12px) saturate(180%);
    border: 1px solid rgba(201,162,39,0.18);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    box-shadow:
        0 20px 40px rgba(0,0,0,0.4),
        inset 0 1px 0 rgba(255,255,255,0.08);
    transition: transform 0.25s cubic-bezier(0.22, 1, 0.36, 1),
                box-shadow 0.25s cubic-bezier(0.22, 1, 0.36, 1);
    animation: popIn 0.6s cubic-bezier(0.22, 1, 0.36, 1) 0.12s both;
}
div[class*="st-key-"]:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow:
        0 28px 54px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.12),
        0 0 0 1px rgba(201,162,39,0.3);
}

/* ---------- Gradient glow header text ---------- */
.dossier-hero h1 {
    background: linear-gradient(100deg, #F0D98C 0%, #C9A227 35%, #7FD8D0 75%, #C9A227 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    filter: drop-shadow(0 0 18px rgba(201,162,39,0.35));
    animation: popIn 0.55s cubic-bezier(0.22, 1, 0.36, 1) both,
               shimmer 7s linear infinite 0.6s;
}
@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}

/* ---------- Focus ring / border gradient on inputs ---------- */
.stTextArea textarea:focus, .stTextInput input:focus {
    box-shadow: 0 0 0 2px var(--gold-soft), 0 0 22px rgba(201,162,39,0.28) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--gold) !important;
    box-shadow: 0 0 22px rgba(201,162,39,0.18);
}

/* ---------- General body text ---------- */
p, li, span, label { color: var(--text); }

/* ==================================================
   CINEMATIC CONTINUOUS 3D MOTION LAYER
   ================================================== */

/* ---------- Ambient drifting light orbs (behind everything) ---------- */
.stApp::before {
    content: "";
    position: fixed;
    inset: -10%;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(circle at 20% 30%, rgba(201,162,39,0.10) 0%, transparent 32%),
        radial-gradient(circle at 80% 20%, rgba(127,216,208,0.08) 0%, transparent 30%),
        radial-gradient(circle at 55% 80%, rgba(28,42,72,0.5) 0%, transparent 40%);
    filter: blur(40px);
    animation: driftOrbs 26s ease-in-out infinite alternate;
}
@keyframes driftOrbs {
    0%   { transform: translate3d(0, 0, 0) scale(1); }
    50%  { transform: translate3d(-2%, 3%, 0) scale(1.06); }
    100% { transform: translate3d(2%, -2%, 0) scale(1); }
}

/* Give the app a real 3D viewing context */
.stApp { perspective: 1600px; }

/* ---------- Continuous 3D breathing tilt on elevated cards ---------- */
.dossier-hero,
.glass-card,
div[class*="st-key-"],
[data-testid="stMetric"] {
    transform-style: preserve-3d;
    animation:
        popIn 0.6s cubic-bezier(0.22,1,0.36,1) both,
        breathe3d 9s ease-in-out infinite 0.6s;
    position: relative;
    overflow: hidden;
}
@keyframes breathe3d {
    0%, 100% { transform: rotateX(0deg) rotateY(0deg) translateZ(0px); }
    25%      { transform: rotateX(0.6deg) rotateY(-0.8deg) translateZ(4px); }
    50%      { transform: rotateX(-0.4deg) rotateY(0.6deg) translateZ(2px); }
    75%      { transform: rotateX(0.5deg) rotateY(0.5deg) translateZ(3px); }
}
/* Hover overrides the ambient float with a deliberate, stronger lift */
.glass-card:hover,
div[class*="st-key-"]:hover {
    animation-play-state: paused;
    transform: translateY(-4px) scale(1.01) rotateX(1.2deg);
}

/* ---------- Periodic light-sweep shimmer across cards ---------- */
.dossier-hero::after,
.glass-card::before,
div[class*="st-key-"]::before {
    content: "";
    position: absolute;
    top: 0; left: -60%;
    width: 40%; height: 100%;
    background: linear-gradient(
        100deg,
        transparent 0%,
        rgba(255,255,255,0.06) 45%,
        rgba(201,162,39,0.14) 50%,
        rgba(255,255,255,0.06) 55%,
        transparent 100%
    );
    transform: skewX(-18deg);
    animation: lightSweep 7s ease-in-out infinite;
    pointer-events: none;
    z-index: 1;
}
@keyframes lightSweep {
    0%   { left: -60%; }
    35%  { left: 130%; }
    100% { left: 130%; }
}

/* ---------- Breathing glow on the accent rule + buttons ---------- */
.dossier-hero .gold-rule {
    animation: pulseGlow 3.2s ease-in-out infinite;
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 6px rgba(201,162,39,0.4); }
    50%      { box-shadow: 0 0 18px rgba(201,162,39,0.85); }
}
.stButton > button {
    animation: pulseGlow 3.2s ease-in-out infinite;
}

/* ---------- Slow continuous hue drift on the header glow ---------- */
.dossier-hero h1 {
    animation:
        popIn 0.55s cubic-bezier(0.22,1,0.36,1) both,
        shimmer 7s linear infinite 0.6s,
        glowBreathe 4.5s ease-in-out infinite 0.6s;
}
@keyframes glowBreathe {
    0%, 100% { filter: drop-shadow(0 0 14px rgba(201,162,39,0.30)); }
    50%      { filter: drop-shadow(0 0 26px rgba(127,216,208,0.35)); }
}

/* Respect users who prefer reduced motion */
@media (prefers-reduced-motion: reduce) {
    .stApp::before, .stApp::after,
    .dossier-hero, .glass-card, div[class*="st-key-"], [data-testid="stMetric"],
    .dossier-hero::after, .glass-card::before, div[class*="st-key-"]::before,
    .dossier-hero .gold-rule, .stButton > button, .dossier-hero h1 {
        animation: none !important;
    }
}
/* ==================================================
   OPENING SEQUENCE — MECHANICAL ASSEMBLY + SCAN
   A 4.0s intro: six armor panels fly in and lock into
   a hex frame, the core snaps in, a verification
   scanner sweeps through, the title locks into place,
   then the whole rig dissolves into the dashboard.
   ================================================== */

.intro-splash {
    position: fixed;
    inset: 0;
    z-index: 100000;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: radial-gradient(circle at 50% 42%, #16213E 0%, #0B1120 72%);
    perspective: 1400px;
    overflow: hidden;
    animation: introExit 0.55s ease-in 3.45s forwards;
}

/* ---------- Assembly stage (panels + core) ---------- */
.assembly-stage {
    position: relative;
    width: 220px;
    height: 220px;
    display: flex;
    align-items: center;
    justify-content: center;
    transform-style: preserve-3d;
}

.panel {
    position: absolute;
    width: 14px;
    height: 64px;
    border-radius: 3px;
    background: linear-gradient(160deg, #E4C45C 0%, #8A6C15 55%, #3A2E08 100%);
    box-shadow: 0 0 14px rgba(201,162,39,0.5), inset 0 1px 0 rgba(255,255,255,0.4);
    opacity: 0;
    animation-timing-function: cubic-bezier(0.17, 0.84, 0.44, 1);
    animation-fill-mode: forwards;
    animation-duration: 0.95s;
}
.panel-1 { animation-name: panel1; animation-delay: 0.00s; }
.panel-2 { animation-name: panel2; animation-delay: 0.10s; }
.panel-3 { animation-name: panel3; animation-delay: 0.20s; }
.panel-4 { animation-name: panel4; animation-delay: 0.30s; }
.panel-5 { animation-name: panel5; animation-delay: 0.40s; }
.panel-6 { animation-name: panel6; animation-delay: 0.50s; }

@keyframes panel1 {
    0%   { opacity: 0; transform: translate(-480px, -340px) rotate(-120deg) scale(0.4); }
    75%  { opacity: 1; transform: translate(2px, -78px) rotate(4deg) scale(1.08); }
    100% { opacity: 1; transform: translate(0, -74px) rotate(0deg) scale(1); }
}
@keyframes panel2 {
    0%   { opacity: 0; transform: translate(480px, -300px) rotate(130deg) scale(0.4); }
    75%  { opacity: 1; transform: translate(66px, -42px) rotate(-56deg) scale(1.08); }
    100% { opacity: 1; transform: translate(62px, -38px) rotate(-60deg) scale(1); }
}
@keyframes panel3 {
    0%   { opacity: 0; transform: translate(500px, 260px) rotate(-110deg) scale(0.4); }
    75%  { opacity: 1; transform: translate(66px, 40px) rotate(-124deg) scale(1.08); }
    100% { opacity: 1; transform: translate(62px, 36px) rotate(-120deg) scale(1); }
}
@keyframes panel4 {
    0%   { opacity: 0; transform: translate(0, 420px) rotate(90deg) scale(0.4); }
    75%  { opacity: 1; transform: translate(2px, 80px) rotate(184deg) scale(1.08); }
    100% { opacity: 1; transform: translate(0, 76px) rotate(180deg) scale(1); }
}
@keyframes panel5 {
    0%   { opacity: 0; transform: translate(-500px, 260px) rotate(110deg) scale(0.4); }
    75%  { opacity: 1; transform: translate(-66px, 40px) rotate(124deg) scale(1.08); }
    100% { opacity: 1; transform: translate(-62px, 36px) rotate(120deg) scale(1); }
}
@keyframes panel6 {
    0%   { opacity: 0; transform: translate(-480px, -300px) rotate(-130deg) scale(0.4); }
    75%  { opacity: 1; transform: translate(-66px, -42px) rotate(56deg) scale(1.08); }
    100% { opacity: 1; transform: translate(-62px, -38px) rotate(60deg) scale(1); }
}

/* Mechanical snap pulse once the ring is fully assembled */
.assembly-stage::before {
    content: "";
    position: absolute;
    width: 170px; height: 170px;
    border-radius: 50%;
    border: 1px solid rgba(201,162,39,0.5);
    opacity: 0;
    animation: ringFlash 0.5s ease-out 1.05s forwards;
}
@keyframes ringFlash {
    0%   { opacity: 0.9; transform: scale(0.7); box-shadow: 0 0 0 rgba(201,162,39,0.8); }
    100% { opacity: 0; transform: scale(1.35); box-shadow: 0 0 40px rgba(201,162,39,0); }
}

/* ---------- Core icon locks in after the panels land ---------- */
.core-icon {
    position: relative;
    z-index: 2;
    font-size: 3.1rem;
    line-height: 1;
    opacity: 0;
    transform: scale(0.3) rotateY(180deg);
    animation: coreSnap 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) 1.05s forwards,
               coreGlow 1.1s ease-in-out 1.6s infinite alternate;
}
@keyframes coreSnap {
    0%   { opacity: 0; transform: scale(0.3) rotateY(180deg); }
    60%  { opacity: 1; transform: scale(1.18) rotateY(0deg); }
    100% { opacity: 1; transform: scale(1) rotateY(0deg); }
}
@keyframes coreGlow {
    0%   { filter: drop-shadow(0 0 10px rgba(201,162,39,0.4)); }
    100% { filter: drop-shadow(0 0 30px rgba(127,216,208,0.75)); }
}

/* ---------- Verification scanner sweep ---------- */
.scan-line {
    position: absolute;
    top: -100%;
    left: -10%;
    width: 120%;
    height: 3px;
    background: linear-gradient(90deg, transparent 0%, #7FD8D0 20%, #FFFFFF 50%, #7FD8D0 80%, transparent 100%);
    box-shadow: 0 0 16px 2px rgba(127,216,208,0.75);
    opacity: 0;
    animation: scanSweep 0.85s cubic-bezier(0.45, 0, 0.55, 1) 1.5s forwards;
}
@keyframes scanSweep {
    0%   { top: -5%;  opacity: 1; }
    92%  { top: 100%; opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

/* ---------- Title + readout ---------- */
.intro-title {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 2.1rem;
    color: #FFFFFF;
    margin-top: 1.5rem;
    opacity: 0;
    letter-spacing: 0.01em;
    transform: translateY(14px) scale(0.96);
    animation: titleLock 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) 2.15s forwards;
}
@keyframes titleLock {
    0%   { opacity: 0; transform: translateY(14px) scale(0.96); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
}

.intro-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    color: var(--gold);
    opacity: 0;
    margin-top: 0.55rem;
    animation: readoutIn 0.2s linear 2.5s forwards,
               readoutFlicker 0.18s linear 2.7s 3;
}
@keyframes readoutIn {
    0%   { opacity: 0; }
    100% { opacity: 1; }
}
@keyframes readoutFlicker {
    0%, 100% { opacity: 1; }
    45%      { opacity: 0.25; }
}

@keyframes introExit {
    0%   { opacity: 1; transform: scale(1); filter: blur(0px); }
    100% { opacity: 0; transform: scale(1.06); filter: blur(6px); visibility: hidden; }
}

/* Main dashboard content stays hidden, then reveals right as the splash exits */
[data-testid="stAppViewContainer"], section[data-testid="stSidebar"] {
    opacity: 0;
    animation: mainReveal 0.7s ease-out 3.5s forwards;
}
@keyframes mainReveal {
    0%   { opacity: 0; transform: translateY(8px); }
    100% { opacity: 1; transform: translateY(0); }
}

@media (prefers-reduced-motion: reduce) {
    .intro-splash { display: none !important; }
    [data-testid="stAppViewContainer"], section[data-testid="stSidebar"] {
        opacity: 1 !important; animation: none !important;
    }
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="intro-splash">
    <div class="assembly-stage">
        <div class="panel panel-1"></div>
        <div class="panel panel-2"></div>
        <div class="panel panel-3"></div>
        <div class="panel panel-4"></div>
        <div class="panel panel-5"></div>
        <div class="panel panel-6"></div>
        <div class="scan-line"></div>
        <div class="core-icon">🔎</div>
    </div>
    <div class="intro-title">Fake Content Detection</div>
    <div class="intro-sub">VERIFICATION PROTOCOL · ACTIVE</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dossier-hero">
    <p class="kicker">Machine learning &nbsp;+&nbsp; live web evidence</p>
    <h1>Fake Content Detection &amp; Evidence Verification</h1>
    <div class="gold-rule"></div>
    <p>Paste an article, drop a screenshot, or upload a clip. A trained classifier and a real-time
    web search work together to check what's true — with sources cited, not just a guess.</p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    sidebar_card = st.container(key="sidebar_info_card")

    with sidebar_card:

        st.header("📌 Project Information")

        st.write(
            "**Feature Extraction:** TF-IDF"
        )

        st.write(
            "**Machine Learning Model:** Linear SVM"
        )

        st.write(
            "**Model Accuracy:** 99.58%"
        )

        st.write(
            "**F1 Score:** 99.62%"
        )

        st.markdown("---")

        st.write(
            "🌐 Real-time evidence is retrieved "
            "using web search."
        )

        st.write(
            "🛡️ Sources are analyzed for credibility."
        )

        st.write(
            "📊 Detection results are saved in history."
        )

        st.write(
            "⚖️ ML prediction and evidence are combined "
            "for a final decision."
        )


# ==================================================
# INPUT SECTION
# ==================================================

input_mode_card = st.container(key="input_mode_card")

with input_mode_card:

    input_mode = st.radio(
        "Choose how you want to provide content:",
        options=["📝 Text", "🖼️ Image (screenshot/meme)", "🎬 Audio / Video"],
        horizontal=True
    )


content = ""


if input_mode == "📝 Text":

    content = st.text_area(
        "📰 Enter News / Social Media Content",
        height=250,
        placeholder=(
            "Paste a news article, headline, or "
            "social-media claim here..."
        )
    )


elif input_mode == "🖼️ Image (screenshot/meme)":

    uploaded_image = st.file_uploader(
        "Upload an image (screenshot of a post, meme, etc.)",
        type=["png", "jpg", "jpeg", "webp"]
    )

    if uploaded_image is not None:

        st.image(
            uploaded_image,
            caption="Uploaded image",
            width="stretch"
        )

        with st.spinner("Extracting text from image..."):

            try:

                content = extract_text_from_image(uploaded_image)

                if content:

                    st.success("Text extracted successfully:")
                    st.text_area(
                        "Extracted text (editable before analysis)",
                        value=content,
                        height=150,
                        key="extracted_image_text"
                    )
                    content = st.session_state["extracted_image_text"]

                else:

                    st.warning(
                        "No readable text was found in this image."
                    )

            except Exception as error:

                st.error(f"Could not process image: {error}")


elif input_mode == "🎬 Audio / Video":

    uploaded_media = st.file_uploader(
        "Upload an audio or video file",
        type=["mp3", "wav", "m4a", "mp4", "mov", "avi", "mkv"]
    )

    if uploaded_media is not None:

        st.audio(uploaded_media) if uploaded_media.type.startswith("audio") else st.video(uploaded_media)

        uploaded_media.seek(0)

        with st.spinner("Transcribing audio... this may take a moment"):

            try:

                content = transcribe_audio_video(
                    uploaded_media,
                    uploaded_media.name
                )

                if content:

                    st.success("Transcription completed:")
                    st.text_area(
                        "Transcribed text (editable before analysis)",
                        value=content,
                        height=150,
                        key="transcribed_media_text"
                    )
                    content = st.session_state["transcribed_media_text"]

                else:

                    st.warning(
                        "No speech could be detected in this file."
                    )

            except Exception as error:

                st.error(
                    f"Could not process audio/video: {error}. "
                    "Make sure ffmpeg is installed on your system."
                )


# ==================================================
# ANALYZE BUTTON
# ==================================================

if st.button(
    "🔍 Analyze Content",
    type="primary"
):

    # ==============================================
    # INPUT VALIDATION
    # ==============================================

    if not content.strip():

        st.warning(
            "Please enter some content before analyzing."
        )

        st.stop()


    # ==============================================
    # MACHINE LEARNING PREDICTION
    # ==============================================

    with st.spinner(
        "Running machine-learning analysis..."
    ):

        prediction_result = predict_content(
            content
        )


    # ==============================================
    # GET ML RESULTS
    # ==============================================

    prediction = prediction_result.get(
        "prediction",
        "UNKNOWN"
    )

    decision_score = prediction_result.get(
        "decision_score",
        None
    )

    ml_confidence = prediction_result.get(
        "confidence",
        50.0
    )

    prediction_strength = prediction_result.get(
        "prediction_strength",
        "LOW"
    )


    # ==============================================
    # MACHINE LEARNING RESULT DISPLAY
    # ==============================================

    st.markdown("---")

    st.header(
        "🤖 Machine Learning Result"
    )


    if prediction == "FAKE":

        st.error(
            "🔴 LIKELY FAKE"
        )

    elif prediction == "REAL":

        st.success(
            "🟢 LIKELY REAL"
        )

    else:

        st.info(
            "⚪ PREDICTION UNKNOWN"
        )


    st.write(
        f"**Prediction:** {prediction}"
    )


    # ==============================================
    # ML METRICS
    # ==============================================

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "ML Prediction",
            prediction
        )


    with col2:

        st.metric(
            "ML Confidence",
            f"{ml_confidence}%"
        )


    # ==============================================
    # MODEL PREDICTION STRENGTH
    # ==============================================

    if prediction_strength == "HIGH":

        st.success(
            "🟢 Model Prediction Strength: HIGH"
        )

    elif prediction_strength == "MODERATE":

        st.warning(
            "🟡 Model Prediction Strength: MODERATE"
        )

    else:

        st.info(
            "⚪ Model Prediction Strength: LOW"
        )


    # ==============================================
    # TECHNICAL MODEL DETAILS
    # ==============================================

    if decision_score is not None:

        with st.expander(
            "🔧 View Technical Model Details"
        ):

            st.write(
                f"**SVM Decision Score:** "
                f"`{decision_score:.4f}`"
            )

            st.caption(
                "The decision score represents the "
                "distance from the SVM decision boundary. "
                "Higher absolute values generally indicate "
                "a stronger model prediction."
            )


    # ==============================================
    # REAL-TIME EVIDENCE SEARCH
    # ==============================================

    st.markdown("---")

    st.header(
        "🌐 Real-Time Evidence Search"
    )


    with st.spinner(
        "Searching the web for relevant evidence..."
    ):

        try:

            evidence = search_evidence(
                content,
                max_results=5
            )

        except Exception as error:

            evidence = []

            st.error(
                f"Evidence search failed: {error}"
            )


    # ==============================================
    # DISPLAY SEARCH STATUS
    # ==============================================

    if evidence:

        st.success(
            f"✅ Successfully retrieved "
            f"{len(evidence)} relevant sources "
            f"from the web."
        )

    else:

        st.warning(
            "⚠️ No relevant evidence sources were found."
        )


    # ==============================================
    # SOURCE CREDIBILITY RANKING
    # ==============================================

    if evidence:

        evidence = rank_sources(
            evidence
        )


    # ==============================================
    # ADVANCED EVIDENCE VERIFICATION
    # ==============================================

    st.markdown("---")

    st.header(
        "🧠 Advanced Evidence Verification"
    )


    claim = extract_claim(
        content
    )


    analysis = analyze_evidence(
        claim,
        evidence
    )


    status = analysis.get(
        "status",
        "UNCLEAR"
    )

    score = analysis.get(
        "score",
        0
    )

    reason = analysis.get(
        "reason",
        "No analysis available."
    )


    # ==============================================
    # DISPLAY EVIDENCE STATUS
    # ==============================================

    if status == "SUPPORTING":

        st.success(
            "🟢 Evidence appears SUPPORTING"
        )

    elif status == "CONTRADICTING":

        st.error(
            "🔴 Evidence appears CONTRADICTING"
        )

    elif status == "RELEVANT":

        st.warning(
            "🟡 Relevant evidence found, but the "
            "claim cannot be conclusively verified."
        )

    else:

        st.info(
            "⚪ Evidence status: UNCLEAR"
        )


    st.write(
        f"**Evidence Relevance Score:** {score}%"
    )

    st.write(
        f"**Analysis:** {reason}"
    )


    # ==============================================
    # SOURCE CREDIBILITY ANALYSIS
    # ==============================================

    st.markdown("---")

    st.header(
        "🏆 Source Credibility Analysis"
    )


    if evidence:

        credibility_scores = []


        for source in evidence:

            credibility_scores.append(
                source.get(
                    "credibility_score",
                    0
                )
            )


        average_credibility = round(
            sum(credibility_scores)
            / len(credibility_scores),
            2
        )


        if average_credibility >= 80:

            st.success(
                "🟢 High Credibility Sources"
            )

        elif average_credibility >= 50:

            st.warning(
                "🟡 Moderate Credibility Sources"
            )

        else:

            st.error(
                "🔴 Low Credibility Sources"
            )


        st.write(
            f"**Average Source Credibility:** "
            f"{average_credibility}%"
        )


    else:

        average_credibility = 0

        st.info(
            "⚪ No sources available for "
            "credibility analysis."
        )


    # ==============================================
    # SMART FINAL DECISION
    # ==============================================

    st.markdown("---")

    st.header(
        "⚖️ Smart Final Decision"
    )


    final_result = make_final_decision(
        prediction=prediction,
        evidence_status=status,
        evidence=evidence
    )


    verdict = final_result.get(
        "verdict",
        "UNCLEAR"
    )

    confidence = final_result.get(
        "confidence",
        0
    )

    final_reason = final_result.get(
        "reason",
        "No final decision explanation available."
    )

    final_average_credibility = final_result.get(
        "average_credibility",
        average_credibility
    )


    if (
        final_average_credibility == 0
        and average_credibility > 0
    ):

        final_average_credibility = (
            average_credibility
        )


    # ==============================================
    # DISPLAY FINAL VERDICT
    # ==============================================

    if verdict == "LIKELY TRUE":

        st.success(
            f"🟢 {verdict}"
        )

    elif verdict == "LIKELY FALSE":

        st.error(
            f"🔴 {verdict}"
        )

    elif verdict == "CONFLICTING SIGNALS":

        st.warning(
            f"🟡 {verdict}"
        )

    else:

        st.info(
            f"⚪ {verdict}"
        )


    # ==============================================
    # FINAL METRICS
    # ==============================================

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Final Confidence",
            f"{confidence}%"
        )


    with col2:

        st.metric(
            "Average Source Credibility",
            f"{final_average_credibility}%"
        )


    st.write(
        f"**Decision Explanation:** "
        f"{final_reason}"
    )


    # ==============================================
    # SAVE DETECTION HISTORY
    # ==============================================

    try:

        save_detection_history(

            content=content,

            prediction=prediction,

            evidence_status=status,

            evidence_score=score,

            source_credibility=final_average_credibility,

            final_verdict=verdict,

            final_confidence=confidence

        )


    except Exception as error:

        st.warning(
            f"History could not be saved: {error}"
        )


    # ==============================================
    # RETRIEVED EVIDENCE SOURCES
    # ==============================================

    st.markdown("---")

    st.header(
        "📚 Retrieved Evidence Sources"
    )


    if evidence:

        st.success(
            f"Found {len(evidence)} relevant sources."
        )


        for number, source in enumerate(
            evidence,
            start=1
        ):


            title = source.get(
                "title",
                "Unknown Source"
            )


            st.subheader(
                f"{number}. {title}"
            )


            credibility_level = source.get(
                "credibility_level",
                "UNKNOWN"
            )

            credibility_score = source.get(
                "credibility_score",
                0
            )

            credibility_reason = source.get(
                "credibility_reason",
                ""
            )


            if credibility_level == "HIGH":

                st.success(
                    f"🛡️ HIGH CREDIBILITY "
                    f"({credibility_score}%)"
                )

            elif credibility_level == "MEDIUM":

                st.warning(
                    f"🟡 MEDIUM CREDIBILITY "
                    f"({credibility_score}%)"
                )

            elif credibility_level == "LOW":

                st.error(
                    f"🔴 LOW CREDIBILITY "
                    f"({credibility_score}%)"
                )

            else:

                st.info(
                    f"⚪ UNKNOWN CREDIBILITY "
                    f"({credibility_score}%)"
                )


            if credibility_reason:

                st.caption(
                    credibility_reason
                )


            if source.get("url"):

                st.write(
                    f"🔗 {source['url']}"
                )


            if source.get("content"):

                source_content = source[
                    "content"
                ]


                if len(source_content) > 600:

                    source_content = (
                        source_content[:600]
                        + "..."
                    )


                st.write(
                    source_content
                )


            if "relevance_score" in source:

                st.write(
                    f"**Search Relevance:** "
                    f"{source['relevance_score']}%"
                )


            st.markdown("---")


    else:

        st.warning(
            "No relevant evidence sources were found."
        )


    # ==============================================
    # DISCLAIMER
    # ==============================================

    st.info(
        "⚠️ The machine-learning prediction, "
        "real-time web evidence, source credibility "
        "analysis, and final verdict are automated "
        "analytical signals. Important claims should "
        "still be manually verified using trusted "
        "fact-checking sources."
    )


# ==================================================
# ADVANCED DETECTION ANALYTICS DASHBOARD
# STEP 29
# ==================================================

st.markdown("---")

st.header(
    "📊 Advanced Detection Analytics Dashboard"
)


try:

    history = load_detection_history()


    if history:

        history_df = pd.DataFrame(history)


        # ==============================================
        # TOTAL DETECTIONS
        # ==============================================

        st.success(
            f"📁 Total Saved Detections: {len(history_df)}"
        )


        # ==============================================
        # DETECTION SUMMARY
        # ==============================================

        st.subheader(
            "📈 Detection Summary"
        )


        total_detections = len(history_df)


        fake_count = 0
        real_count = 0


        if "ml_prediction" in history_df.columns:

            fake_count = len(
                history_df[
                    history_df["ml_prediction"] == "FAKE"
                ]
            )

            real_count = len(
                history_df[
                    history_df["ml_prediction"] == "REAL"
                ]
            )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Total Detections",
                total_detections
            )


        with col2:

            st.metric(
                "🔴 ML Fake Predictions",
                fake_count
            )


        with col3:

            st.metric(
                "🟢 ML Real Predictions",
                real_count
            )


        # ==============================================
        # PREDICTION DISTRIBUTION CHART
        # ==============================================

        st.markdown("---")

        st.subheader(
            "📊 Machine Learning Prediction Distribution"
        )


        prediction_data = pd.DataFrame({

            "Prediction": [
                "FAKE",
                "REAL"
            ],

            "Count": [
                fake_count,
                real_count
            ]

        })


        if (
            fake_count > 0
            or real_count > 0
        ):

            prediction_chart = px.pie(

                prediction_data,

                names="Prediction",

                values="Count",

                title=(
                    "Fake vs Real Content Predictions"
                ),

                hole=0.4,

                template="plotly_dark",

                color="Prediction",

                color_discrete_map={
                    "FAKE": "#E5484D",
                    "REAL": "#2F9E6D"
                }

            )

            prediction_chart.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E7EAF3"
            )


            st.plotly_chart(
                prediction_chart,
                width="stretch"
            )


        else:

            st.info(
                "Not enough prediction data "
                "available for visualization."
            )


        # ==============================================
        # FINAL VERDICT SUMMARY
        # ==============================================

        st.markdown("---")

        st.subheader(
            "⚖️ Final Decision Summary"
        )


        likely_true_count = 0
        likely_false_count = 0
        conflicting_count = 0
        unclear_count = 0


        if "final_verdict" in history_df.columns:

            likely_true_count = len(
                history_df[
                    history_df["final_verdict"]
                    == "LIKELY TRUE"
                ]
            )

            likely_false_count = len(
                history_df[
                    history_df["final_verdict"]
                    == "LIKELY FALSE"
                ]
            )

            conflicting_count = len(
                history_df[
                    history_df["final_verdict"]
                    == "CONFLICTING SIGNALS"
                ]
            )

            unclear_count = len(
                history_df[
                    history_df["final_verdict"]
                    == "UNCLEAR"
                ]
            )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "🟢 Likely True",
                likely_true_count
            )


        with col2:

            st.metric(
                "🔴 Likely False",
                likely_false_count
            )


        with col3:

            st.metric(
                "🟡 Conflicting",
                conflicting_count
            )


        with col4:

            st.metric(
                "⚪ Unclear",
                unclear_count
            )


        # ==============================================
        # FINAL VERDICT BAR CHART
        # ==============================================

        verdict_data = pd.DataFrame({

            "Verdict": [

                "LIKELY TRUE",

                "LIKELY FALSE",

                "CONFLICTING SIGNALS",

                "UNCLEAR"

            ],

            "Count": [

                likely_true_count,

                likely_false_count,

                conflicting_count,

                unclear_count

            ]

        })


        if verdict_data["Count"].sum() > 0:

            verdict_chart = px.bar(

                verdict_data,

                x="Verdict",

                y="Count",

                title=(
                    "Final Decision Distribution"
                ),

                text="Count",

                template="plotly_dark",

                color_discrete_sequence=["#C9A227"]

            )

            verdict_chart.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E7EAF3"
            )


            st.plotly_chart(
                verdict_chart,
                width="stretch"
            )


        else:

            st.info(
                "No final verdict data available yet."
            )


        # ==============================================
        # AVERAGE ANALYTICS
        # ==============================================

        st.markdown("---")

        st.subheader(
            "📊 Average Detection Metrics"
        )


        average_confidence = 0
        average_credibility = 0


        if "final_confidence" in history_df.columns:

            confidence_values = pd.to_numeric(

                history_df["final_confidence"],

                errors="coerce"

            )


            average_confidence = confidence_values.mean()


            if pd.isna(average_confidence):

                average_confidence = 0

            else:

                average_confidence = round(
                    average_confidence,
                    2
                )


        if "source_credibility" in history_df.columns:

            credibility_values = pd.to_numeric(

                history_df["source_credibility"],

                errors="coerce"

            )


            average_credibility = credibility_values.mean()


            if pd.isna(average_credibility):

                average_credibility = 0

            else:

                average_credibility = round(
                    average_credibility,
                    2
                )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Average Final Confidence",
                f"{average_confidence}%"
            )


        with col2:

            st.metric(
                "Average Source Credibility",
                f"{average_credibility}%"
            )


        # ==============================================
        # CONFIDENCE VS CREDIBILITY CHART
        # ==============================================

        if (
            "final_confidence" in history_df.columns
            and "source_credibility" in history_df.columns
        ):

            chart_df = history_df.copy()


            chart_df["final_confidence"] = pd.to_numeric(

                chart_df["final_confidence"],

                errors="coerce"

            )


            chart_df["source_credibility"] = pd.to_numeric(

                chart_df["source_credibility"],

                errors="coerce"

            )


            chart_df = chart_df.dropna(

                subset=[
                    "final_confidence",
                    "source_credibility"
                ]

            )


            if not chart_df.empty:

                st.markdown("---")

                st.subheader(
                    "📈 Confidence vs Source Credibility"
                )


                confidence_chart = px.bar(

                    chart_df,

                    y=[
                        "final_confidence",
                        "source_credibility"
                    ],

                    title=(
                        "Detection Confidence and "
                        "Source Credibility Comparison"
                    ),

                    barmode="group",

                    template="plotly_dark",

                    color_discrete_sequence=["#C9A227", "#2F9E6D"]

                )

                confidence_chart.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#E7EAF3"
                )


                st.plotly_chart(
                    confidence_chart,
                    width="stretch"
                )


        # ==============================================
        # DETECTION HISTORY TABLE
        # ==============================================

        st.markdown("---")

        st.subheader(
            "📋 Detection History"
        )


        history_df = history_df.iloc[
            ::-1
        ].reset_index(
            drop=True
        )


        st.dataframe(

            history_df,

            width="stretch",

            hide_index=True

        )


    else:

        st.info(
            "No detection history available yet."
        )


except Exception as error:

    st.warning(
        f"Unable to load analytics dashboard: {error}"
    )
