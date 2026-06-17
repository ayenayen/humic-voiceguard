"""
HUMIC VoiceGuard — Audio Bullying Detection System
Aplikasi Streamlit untuk deteksi perundungan dari rekaman audio.
Desain mengikuti mockup Figma: dark navy theme dengan aksen teal.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os

# Atur Hugging Face cache ke drive D: karena C: penuh
os.environ["HF_HOME"] = r"d:\Intern_HUMIC\hf_cache"

# -- FIX FFMPEG DEPENDENCY ON STREAMLIT CLOUD --
try:
    import imageio_ffmpeg
    ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
except Exception:
    pass
# ---------------------------------------------

import plotly.graph_objects as go
import plotly.express as px
import json
import os
import tempfile
import soundfile as sf
import torch
from transformers import pipeline
from datetime import datetime, timedelta
from pathlib import Path
import random
import time

# ════════════════════════════════════════════════════════════════════════════════
#  KONFIGURASI HALAMAN
# ════════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HUMIC VoiceGuard — Audio Bullying Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Menyesuaikan tampilan Figma
# ════════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ──────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background-color: #0F172A;
}

/* ── Hide Streamlit Defaults ─────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }

/* ── Top Nav Bar ─────────────────────────────────────────────────────────── */
.top-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 0px;
    margin-bottom: 28px;
    border-bottom: 1px solid rgba(45, 212, 191, 0.15);
}
.top-navbar .brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.top-navbar .brand-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #2DD4BF, #14B8A6);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.top-navbar .brand-text {
    font-size: 18px;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.3px;
}
.top-navbar .brand-sub {
    font-size: 13px;
    font-weight: 400;
    color: #94A3B8;
    margin-left: 4px;
}
.nav-tabs {
    display: flex;
    gap: 4px;
    background: #1E293B;
    border-radius: 12px;
    padding: 4px;
}
.nav-tab {
    padding: 8px 20px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    color: #94A3B8;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
    border: none;
    background: transparent;
}
.nav-tab:hover {
    color: #F8FAFC;
    background: rgba(45, 212, 191, 0.08);
}
.nav-tab.active {
    background: #2DD4BF;
    color: #0F172A;
    font-weight: 600;
}

/* ── Metric Cards ────────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, #1E293B 0%, #162032 100%);
    border: 1px solid rgba(45, 212, 191, 0.12);
    border-radius: 16px;
    padding: 24px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(45, 212, 191, 0.08);
}
.metric-label {
    font-size: 12px;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(135deg, #2DD4BF, #22D3EE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.metric-trend {
    font-size: 12px;
    color: #10B981;
    margin-top: 6px;
    display: flex;
    align-items: center;
    gap: 4px;
}
.metric-trend.negative {
    color: #EF4444;
}

/* ── Upload Area ─────────────────────────────────────────────────────────── */
.upload-zone {
    border: 2px dashed rgba(45, 212, 191, 0.35);
    border-radius: 16px;
    padding: 40px 24px;
    text-align: center;
    background: rgba(45, 212, 191, 0.04);
    transition: all 0.3s ease;
    margin: 8px 0;
}
.upload-zone:hover {
    border-color: #2DD4BF;
    background: rgba(45, 212, 191, 0.08);
}
.upload-icon {
    font-size: 40px;
    margin-bottom: 12px;
    color: #2DD4BF;
}
.upload-text {
    font-size: 15px;
    font-weight: 600;
    color: #F8FAFC;
    margin-bottom: 4px;
}
.upload-sub {
    font-size: 12px;
    color: #64748B;
}

/* ── Risk Badge ──────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-high {
    background: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.badge-medium {
    background: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.badge-low {
    background: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

/* ── Top Risk Panel ──────────────────────────────────────────────────────── */
.risk-panel {
    background: linear-gradient(145deg, #1E293B 0%, #162032 100%);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 16px;
    padding: 24px;
    border-left: 4px solid #EF4444;
}
.risk-panel-header {
    font-size: 11px;
    font-weight: 700;
    color: #EF4444;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 12px;
}
.risk-panel-title {
    font-size: 17px;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 6px;
}
.risk-panel-sub {
    font-size: 12px;
    color: #94A3B8;
}

/* ── Analysis Table ──────────────────────────────────────────────────────── */
.analysis-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    background: #1E293B;
    border-radius: 12px;
    margin-bottom: 8px;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}
.analysis-row:hover {
    border-color: rgba(45, 212, 191, 0.2);
    background: #1a2a40;
}
.analysis-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.analysis-name {
    font-size: 14px;
    font-weight: 600;
    color: #F8FAFC;
}
.analysis-meta {
    font-size: 11px;
    color: #64748B;
}
.analysis-actions {
    display: flex;
    align-items: center;
    gap: 12px;
}

/* ── Action Buttons ──────────────────────────────────────────────────────── */
.btn-primary {
    background: linear-gradient(135deg, #2DD4BF, #14B8A6);
    color: #0F172A;
    border: none;
    padding: 8px 20px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
}
.btn-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(45, 212, 191, 0.3);
}
.btn-outline {
    background: transparent;
    color: #2DD4BF;
    border: 1px solid rgba(45, 212, 191, 0.3);
    padding: 8px 20px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
}
.btn-outline:hover {
    background: rgba(45, 212, 191, 0.1);
    border-color: #2DD4BF;
}
.btn-export {
    background: rgba(239, 68, 68, 0.12);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.25);
    padding: 8px 20px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* ── Transcript ──────────────────────────────────────────────────────────── */
.transcript-segment {
    padding: 14px 18px;
    border-radius: 10px;
    margin-bottom: 6px;
    border-left: 3px solid;
    background: rgba(30, 41, 59, 0.6);
}
.transcript-speaker {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}
.transcript-text {
    font-size: 14px;
    color: #CBD5E1;
    line-height: 1.6;
}
.bullying-word {
    color: #EF4444;
    font-weight: 700;
    text-decoration: underline;
    text-decoration-color: rgba(239, 68, 68, 0.4);
    text-underline-offset: 2px;
}
.indicator-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 4px;
}

/* ── Section Headers ─────────────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    margin-top: 28px;
}
.section-title {
    font-size: 17px;
    font-weight: 700;
    color: #F8FAFC;
}
.section-link {
    font-size: 13px;
    color: #2DD4BF;
    font-weight: 500;
    cursor: pointer;
    text-decoration: none;
}
.section-link:hover {
    text-decoration: underline;
}

/* ── History Table ────────────────────────────────────────────────────────── */
.history-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 6px;
}
.history-table th {
    font-size: 11px;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid rgba(100, 116, 139, 0.2);
}
.history-table td {
    padding: 14px 16px;
    font-size: 13px;
    color: #CBD5E1;
    background: #1E293B;
}
.history-table tr td:first-child {
    border-radius: 10px 0 0 10px;
}
.history-table tr td:last-child {
    border-radius: 0 10px 10px 0;
}

/* ── Filter Buttons ──────────────────────────────────────────────────────── */
.filter-group {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
}
.filter-btn {
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid;
    cursor: pointer;
    transition: all 0.2s ease;
}
.filter-all {
    background: #2DD4BF;
    color: #0F172A;
    border-color: #2DD4BF;
}
.filter-low {
    background: transparent;
    color: #10B981;
    border-color: rgba(16, 185, 129, 0.3);
}
.filter-medium {
    background: transparent;
    color: #F59E0B;
    border-color: rgba(245, 158, 11, 0.3);
}
.filter-high {
    background: transparent;
    color: #EF4444;
    border-color: rgba(239, 68, 68, 0.3);
}

/* ── Pagination ──────────────────────────────────────────────────────────── */
.pagination {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-top: 24px;
}
.page-btn {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 600;
    background: #1E293B;
    color: #94A3B8;
    border: 1px solid transparent;
    cursor: pointer;
}
.page-btn.active {
    background: #2DD4BF;
    color: #0F172A;
}

/* ── Streamlit Overrides ─────────────────────────────────────────────────── */
.stFileUploader > div {
    background: rgba(45, 212, 191, 0.04) !important;
    border: 2px dashed rgba(45, 212, 191, 0.35) !important;
    border-radius: 16px !important;
}
.stFileUploader > div:hover {
    border-color: #2DD4BF !important;
    background: rgba(45, 212, 191, 0.08) !important;
}
div[data-testid="stFileUploader"] label {
    color: #94A3B8 !important;
    font-size: 13px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #2DD4BF, #14B8A6) !important;
    color: #0F172A !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 24px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(45, 212, 191, 0.3) !important;
}
.stTextInput > div > div {
    background: #1E293B !important;
    border: 1px solid rgba(100, 116, 139, 0.2) !important;
    border-radius: 10px !important;
    color: #F8FAFC !important;
}
.stTextInput input {
    color: #F8FAFC !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #1E293B;
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
    padding: 8px 20px !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: #2DD4BF !important;
    color: #0F172A !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
div[data-testid="stExpander"] {
    background: #1E293B !important;
    border: 1px solid rgba(100, 116, 139, 0.15) !important;
    border-radius: 12px !important;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #2DD4BF, #22D3EE) !important;
    border-radius: 8px !important;
}
.stSpinner > div {
    border-top-color: #2DD4BF !important;
}
/* Info/Success/Warning/Error boxes */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
}
/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0F172A;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

/* ── Deep Analysis specific ──────────────────────────────────────────────── */
.analysis-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 0;
    border-bottom: 1px solid rgba(100, 116, 139, 0.15);
    margin-bottom: 24px;
}
.analysis-title {
    font-size: 22px;
    font-weight: 700;
    color: #F8FAFC;
}
.analysis-subtitle {
    font-size: 13px;
    color: #64748B;
    margin-top: 2px;
}
.stat-row {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
}
.stat-item {
    background: #1E293B;
    border-radius: 10px;
    padding: 14px 18px;
    flex: 1;
    text-align: center;
    border: 1px solid rgba(100, 116, 139, 0.1);
}
.stat-item-value {
    font-size: 20px;
    font-weight: 700;
    color: #2DD4BF;
}
.stat-item-label {
    font-size: 10px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 2px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#  KONSTANTA & DATA DUMMY (untuk demo — nanti diganti data asli)
# ════════════════════════════════════════════════════════════════════════════════

BULLYING_KEYWORDS = [
    "bodoh", "idiot", "tolol", "goblok", "jelek", "bego", "bangsat",
    "anjing", "kampret", "nyusahin", "ganggu", "nyebelin", "brengsek",
    "sialan", "sampah", "gapunya temen", "jelek banget", "aneh",
    "mati aja", "bunuh", "benci", "jijik", "bajingan", "dungu"
]

SPEAKER_COLORS = {
    "Pembicara A": "#2DD4BF",
    "Pembicara B": "#F59E0B",
    "Pembicara C": "#A78BFA",
    "Korban":      "#60A5FA",
    "Pembicara":   "#2DD4BF", # Default for Whisper without diarization
}

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    """Load Whisper model once and cache it."""
    try:
        # Menggunakan whisper-tiny (150MB) agar loading cepat dan meminimalisir timeout.
        # Setelah model terunduh, internet hanya digunakan sedikit.
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            chunk_length_s=30,
            return_timestamps=True,
        )
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

import re

def detect_and_highlight(text):
    count = 0
    # Sort keywords by length descending so "jelek banget" matches before "jelek"
    for kw in sorted(BULLYING_KEYWORDS, key=len, reverse=True):
        pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            count += len(matches)
            text = pattern.sub(f"<bw>{matches[0]}</bw>", text)
    return text, count


def generate_demo_history():
    """Generate data riwayat analisis demo."""
    filenames = [
        "Sesi_Kelas_9B_recess.mp3", "Wawancara_OSIS_Q3.wav",
        "Diskusi_Lt2_rekaman3.mp3", "Sesi_Kelas_7A_pagi.mp3",
        "Lab_Computer_sore.wav", "Sesi_Kelas_8C_istirahat.mp3",
        "Wawancara_BK_siswa12.wav", "Rekaman_Kantin_siang.mp3",
        "Kelas_9A_olahraga.wav", "Perpustakaan_sore.mp3",
        "Lab_Bahasa_pagi.wav", "Sesi_Kelas_7B_seni.mp3",
        "Koridor_Lt1_siang.wav", "Mushola_sholat_dzuhur.mp3",
        "Kelas_8A_matematika.wav",
    ]
    risks = ["High Risk", "Low Risk", "Medium Risk", "Low Risk",
             "High Risk", "Medium Risk", "Low Risk", "High Risk",
             "Low Risk", "Medium Risk", "High Risk", "Low Risk",
             "Medium Risk", "Low Risk", "High Risk"]
    durations = ["12:40", "08:15", "15:02", "09:48", "20:11",
                 "11:05", "06:30", "18:22", "14:55", "07:33",
                 "22:10", "09:15", "05:40", "13:28", "16:45"]
    scores = [92, 15, 55, 22, 88, 60, 18, 85, 25, 48,
              91, 20, 52, 12, 78]
    base_date = datetime(2026, 6, 17)
    records = []
    for i in range(len(filenames)):
        date = base_date - timedelta(days=i // 2, hours=random.randint(0, 12))
        records.append({
            "filename": filenames[i],
            "tanggal": date.strftime("%d %b %Y"),
            "durasi": durations[i],
            "level_risiko": risks[i],
            "skor_risiko": scores[i],
            "bullying_keywords": random.randint(0, 8),
        })
    return records


def generate_demo_transcript():
    """Transkrip demo untuk Deep Analysis — menampilkan deteksi bullying."""
    return [
        {"speaker": "Pembicara A", "time": "00:12", "text": "Eh, dia bawa tas <bw>jelek banget</bw> ya hahaha"},
        {"speaker": "Pembicara B", "time": "00:18", "text": "Iya mukanya juga aneh,"},
        {"speaker": "Pembicara A", "time": "00:25", "text": "Udah deketin kita lagi, kamu <bw>bodoh</bw> sih"},
        {"speaker": "Pembicara C", "time": "00:33", "text": "Bener, mendingan sendirian aja terus"},
        {"speaker": "Korban",      "time": "00:40", "text": "Aku cuma mau ikut main bareng kalian"},
        {"speaker": "Pembicara B", "time": "00:45", "text": "Ikut-ikut, <bw>ganggu</bw> aja kamu"},
        {"speaker": "Pembicara A", "time": "00:52", "text": "Pergi sana, <bw>nyusahin</bw> doang"},
        {"speaker": "Pembicara B", "time": "01:00", "text": "Emang <bw>gapunya temen</bw>, aneh banget"},
        {"speaker": "Korban",      "time": "01:08", "text": "Ya udah aku pergi deh"},
        {"speaker": "Pembicara C", "time": "01:15", "text": "Ya pergi, <bw>nyebelin</bw> emang"},
        {"speaker": "Pembicara A", "time": "01:22", "text": "Lu itu <bw>idiot</bw> banget sih, heran gue"},
        {"speaker": "Pembicara B", "time": "01:30", "text": "Udah biarin aja, emang dia <bw>sampah</bw>"},
    ]


def render_badge(level):
    """Render badge HTML sesuai level risiko."""
    css_class = {
        "High Risk": "badge-high",
        "Medium Risk": "badge-medium",
        "Low Risk": "badge-low",
    }.get(level, "badge-low")
    return f'<span class="badge {css_class}">{level}</span>'


def render_transcript_html(segments):
    """Render transkrip sebagai HTML dengan highlighting kata bullying."""
    html_parts = []
    for seg in segments:
        speaker = seg["speaker"]
        color = SPEAKER_COLORS.get(speaker, "#94A3B8")

        # Parse <bw>...</bw> tags menjadi span bullying-word
        text = seg["text"]
        text = text.replace("<bw>", '<span class="bullying-word">')
        text = text.replace("</bw>", "</span>")

        html_parts.append(f"""
        <div class="transcript-segment" style="border-left-color: {color};">
            <div class="transcript-speaker" style="color: {color};">
                <span class="indicator-dot" style="background: {color};"></span>
                {speaker} — {seg["time"]}
            </div>
            <div class="transcript-text">{text}</div>
        </div>
        """)
    return "\n".join(html_parts)


# ════════════════════════════════════════════════════════════════════════════════
#  STATE MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "history" not in st.session_state:
    st.session_state.history = generate_demo_history()
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "filter_risk" not in st.session_state:
    st.session_state.filter_risk = "Semua"
if "processing" not in st.session_state:
    st.session_state.processing = False


# ════════════════════════════════════════════════════════════════════════════════
#  NAVIGASI — Top bar tabs
# ════════════════════════════════════════════════════════════════════════════════

def render_navbar():
    """Render top navigation bar."""
    nav_col1, nav_col2 = st.columns([3, 2])

    with nav_col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
            <div style="width: 36px; height: 36px; background: linear-gradient(135deg, #2DD4BF, #14B8A6);
                        border-radius: 10px; display: flex; align-items: center; justify-content: center;
                        font-size: 18px;"></div>
            <div>
                <span style="font-size: 18px; font-weight: 700; color: #F8FAFC;">HUMIC VoiceGuard</span>
                <span style="font-size: 13px; color: #94A3B8; margin-left: 8px;">— Audio Bullying Detection</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with nav_col2:
        tabs = st.columns(3)
        pages = ["Dashboard", "Deep Analysis", "History"]
        icons = ["", "", ""]
        for i, (page, icon) in enumerate(zip(pages, icons)):
            with tabs[i]:
                if st.button(f"{icon} {page}", key=f"nav_{page}",
                             use_container_width=True):
                    st.session_state.page = page
                    st.rerun()

    st.markdown('<div style="border-bottom: 1px solid rgba(45, 212, 191, 0.12); margin-bottom: 24px;"></div>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  HALAMAN 1: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    """Render halaman Dashboard."""

    # ── Baris Metrik + Upload + Top Risk ──
    col_metric, col_upload, col_risk = st.columns([1, 1.6, 1])

    with col_metric:
        total_processed = len(st.session_state.history)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">AUDIO DIPROSES</div>
            <div class="metric-value">{total_processed}</div>
            <div class="metric-trend">▲ +3 dibanding minggu lalu</div>
        </div>
        """, unsafe_allow_html=True)

    with col_upload:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon"></div>
            <div class="upload-text">Tarik & Lepas File Audio</div>
            <div class="upload-sub">atau klik untuk memilih file (.mp3, .wav)</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Pilih file audio",
            type=["wav", "mp3"],
            label_visibility="collapsed",
            key="audio_upload"
        )
        if uploaded_file is not None:
            st.session_state.selected_file = uploaded_file.name
            st.success(f"File **{uploaded_file.name}** berhasil diunggah!")
            if st.button("Mulai Analisis", key="start_analysis"):
                st.session_state.uploaded_bytes = uploaded_file.getvalue()
                st.session_state.page = "Deep Analysis"
                st.session_state.processing = True
                st.rerun()
                
        # [DEBUG] Tombol Auto-Test
        if st.button("[DEBUG] Test Auto-Upload (K1_chunk)", help="Klik ini untuk mensimulasikan upload jika file dialog bermasalah"):
            test_file_path = r"d:\Intern_HUMIC\Internship_Humic\RIN_DATAVERSE_SAMPLE\RIN_DATAVERSE_SAMPLE\CHUNKS\K1\K1_chunk_004.wav"
            if os.path.exists(test_file_path):
                with open(test_file_path, "rb") as f:
                    st.session_state.uploaded_bytes = f.read()
                st.session_state.selected_file = "K1_chunk_004.wav"
                st.session_state.page = "Deep Analysis"
                st.session_state.processing = True
                st.rerun()
            else:
                st.error("File test tidak ditemukan di disk.")

    with col_risk:
        # Cari sesi risiko tertinggi
        top_risk = None
        for rec in st.session_state.history:
            if rec["level_risiko"] == "High Risk":
                top_risk = rec
                break
        if top_risk:
            st.markdown(f"""
            <div class="risk-panel">
                <div class="risk-panel-header">TOP RISK SESSION</div>
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                    <div class="risk-panel-title">{top_risk["filename"]}</div>
                    {render_badge("High Risk")}
                </div>
                <div class="risk-panel-sub">Skor risiko {top_risk["skor_risiko"]}/100 · {top_risk["bullying_keywords"]} kata kunci terdeteksi</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Bagian Analisis Terbaru ──
    st.markdown("""
    <div class="section-header">
        <div class="section-title">Analisis Terbaru</div>
        <div class="section-link" onclick="document.querySelector('[data-testid=\\'nav_History\\']')?.click()">Lihat Semua →</div>
    </div>
    """, unsafe_allow_html=True)

    for rec in st.session_state.history[:5]:
        col_info, col_badge, col_btn = st.columns([4, 1, 0.7])
        with col_info:
            st.markdown(f"""
            <div style="padding: 4px 0;">
                <div class="analysis-name">{rec["filename"]}</div>
                <div class="analysis-meta">{rec["tanggal"]} · Durasi {rec["durasi"]}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_badge:
            st.markdown(f'<div style="padding-top:8px;">{render_badge(rec["level_risiko"])}</div>',
                        unsafe_allow_html=True)
        with col_btn:
            if st.button("Buka", key=f"open_{rec['filename']}"):
                st.session_state.selected_file = rec["filename"]
                st.session_state.page = "Deep Analysis"
                st.session_state.processing = False
                st.rerun()

        st.markdown('<div style="border-bottom: 1px solid rgba(100, 116, 139, 0.1); margin: 2px 0 6px 0;"></div>',
                    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  HALAMAN 2: DEEP ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

def page_deep_analysis():
    """Render halaman Deep Analysis."""

    filename = st.session_state.selected_file or "Sesi_Kelas_9B_recess.mp3"

    # ── Processing animation ──
    if st.session_state.processing:
        with st.spinner("Menganalisis audio dengan Whisper ASR..."):
            audio_bytes = st.session_state.get('uploaded_bytes')
            if audio_bytes:
                pipe = load_whisper_model()
                
                # Gunakan file temporary agar ffmpeg bisa membaca file audio (mp3/wav)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                    
                res = pipe(tmp_path, return_timestamps=True)
                os.remove(tmp_path)
                
                segments = []
                total_bullying = 0
                for chunk in res['chunks']:
                    text = chunk['text'].strip()
                    highlighted_text, b_count = detect_and_highlight(text)
                    total_bullying += b_count
                    
                    # Safe timestamp formatting
                    start_ts = chunk['timestamp'][0]
                    if start_ts is None: start_ts = 0
                    start_time_str = time.strftime('%M:%S', time.gmtime(start_ts))
                    
                    segments.append({
                        "speaker": "Pembicara", 
                        "time": start_time_str, 
                        "text": highlighted_text
                    })
                
                st.session_state.real_segments = segments
                st.session_state.real_bullying_count = total_bullying
                
            else:
                time.sleep(1) # fallback dummy sleep
        st.session_state.processing = False
        st.rerun()

    # ── Header ──
    hdr_left, hdr_right = st.columns([4, 1])
    with hdr_left:
        st.markdown(f"""
        <div>
            <div class="analysis-title">Deep Analysis</div>
            <div class="analysis-subtitle">{filename} · Durasi 20:14</div>
        </div>
        """, unsafe_allow_html=True)
    with hdr_right:
        st.markdown("""
        <div style="text-align: right; padding-top: 8px;">
            <span class="btn-export">Export PDF</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="border-bottom:1px solid rgba(100,116,139,0.15); margin: 8px 0 20px 0;"></div>',
                unsafe_allow_html=True)

    # ── Stat Cards ──
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("""
        <div class="stat-item">
            <div class="stat-item-value">20:14</div>
            <div class="stat-item-label">Durasi</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="stat-item">
            <div class="stat-item-value">1,284</div>
            <div class="stat-item-label">Total Kata</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="stat-item" style="border-color: rgba(239,68,68,0.2);">
            <div class="stat-item-value" style="color:#EF4444;">High</div>
            <div class="stat-item-label">Level Risiko</div>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown("""
        <div class="stat-item">
            <div class="stat-item-value">92<span style="font-size:14px;color:#64748B;">/100</span></div>
            <div class="stat-item-label">Skor Risiko</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column Layout: Transcript | Charts ──
    col_transcript, col_charts = st.columns([1.2, 1])

    with col_transcript:
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
            <div class="section-title">Transkrip Percakapan</div>
            <div style="font-size:12px; color:#94A3B8;">
                <span class="indicator-dot" style="background:#EF4444;"></span> Kata perundungan terdeteksi
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Use real segments if available, otherwise demo
        segments = st.session_state.get('real_segments')
        if not segments:
            segments = generate_demo_transcript()
        # Render each segment individually to avoid Streamlit HTML rendering issues
        transcript_container = st.container(height=520)
        with transcript_container:
            for seg in segments:
                speaker = seg["speaker"]
                color = SPEAKER_COLORS.get(speaker, "#94A3B8")
                text = seg["text"]
                text = text.replace("<bw>", '<span class="bullying-word">')
                text = text.replace("</bw>", "</span>")
                st.markdown(f"""
                <div class="transcript-segment" style="border-left-color: {color};">
                    <div class="transcript-speaker" style="color: {color};">
                        <span class="indicator-dot" style="background: {color};"></span>
                        {speaker} — {seg["time"]}
                    </div>
                    <div class="transcript-text">{text}</div>
                </div>
                """, unsafe_allow_html=True)

    with col_charts:
        # ── Emotional Arc Chart ──
        st.markdown("""
        <div style="margin-bottom: 8px;">
            <div class="section-title">Emotional Arc</div>
            <div style="font-size:12px; color:#64748B;">Intensitas emosi negatif sepanjang 20 menit audio</div>
        </div>
        """, unsafe_allow_html=True)

        # Generate time series data for emotional arc
        time_points = np.linspace(0, 20, 50)
        base_curve = 0.3 + 0.2 * np.sin(time_points * 0.5)
        spikes = np.zeros_like(time_points)
        spikes[8:12] += 0.25
        spikes[18:22] += 0.35
        spikes[30:35] += 0.45
        spikes[40:44] += 0.3
        emotion_values = base_curve + spikes + np.random.normal(0, 0.04, len(time_points))
        emotion_values = np.clip(emotion_values, 0, 1)

        fig_arc = go.Figure()
        fig_arc.add_trace(go.Scatter(
            x=time_points, y=emotion_values,
            mode='lines',
            line=dict(color='#2DD4BF', width=2.5, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(45, 212, 191, 0.08)',
            name='Intensitas Emosi'
        ))
        # Add peak annotation
        peak_idx = np.argmax(emotion_values)
        fig_arc.add_annotation(
            x=time_points[peak_idx], y=emotion_values[peak_idx],
            text="PUNCAK",
            showarrow=True, arrowhead=2, arrowcolor="#EF4444",
            font=dict(color="#EF4444", size=10, family="Inter"),
            bgcolor="rgba(239,68,68,0.12)",
            bordercolor="rgba(239,68,68,0.3)",
            borderwidth=1, borderpad=4,
        )
        fig_arc.update_layout(
            height=240,
            margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30, 41, 59, 0.5)',
            xaxis=dict(
                title=dict(text="Menit", font=dict(size=10, color="#64748B")),
                tickfont=dict(size=10, color="#64748B"),
                gridcolor="rgba(100,116,139,0.1)",
                range=[0, 20],
            ),
            yaxis=dict(
                title=dict(text="Intensitas", font=dict(size=10, color="#64748B")),
                tickfont=dict(size=10, color="#64748B"),
                gridcolor="rgba(100,116,139,0.1)",
                range=[0, 1],
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_arc, use_container_width=True, config={"displayModeBar": False})

        # ── Distribusi Dialek Chart ──
        st.markdown("""
        <div style="margin-bottom: 8px; margin-top: 8px;">
            <div class="section-title">Distribusi Dialek Terdeteksi</div>
            <div style="font-size:12px; color:#64748B;">Analisis aksen regional dari pembicara</div>
        </div>
        """, unsafe_allow_html=True)

        dialek_data = pd.DataFrame({
            "Dialek": ["Sunda", "Jawa", "Betawi", "Lainnya"],
            "Persentase": [45, 25, 20, 10],
        })
        colors_dialek = ["#10B981", "#F59E0B", "#2DD4BF", "#64748B"]

        fig_dialek = go.Figure(data=[go.Pie(
            labels=dialek_data["Dialek"],
            values=dialek_data["Persentase"],
            hole=0.55,
            marker=dict(colors=colors_dialek, line=dict(color='#0F172A', width=2)),
            textinfo='none',
            hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
        )])
        # Add center text
        fig_dialek.add_annotation(
            text="4<br><span style='font-size:10px;color:#64748B;'>Dialek</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=24, color="#F8FAFC", family="Inter"),
        )
        # Add legend items manually
        fig_dialek.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=8, b=8),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle", y=0.5,
                xanchor="left", x=1.05,
                font=dict(color="#CBD5E1", size=12),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig_dialek, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════════════
#  HALAMAN 3: HISTORY
# ════════════════════════════════════════════════════════════════════════════════

def page_history():
    """Render halaman History."""

    st.markdown("""
    <div>
        <div class="analysis-title">Riwayat Analisis</div>
        <div class="analysis-subtitle">Semua rekaman audio yang telah dianalisis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Search & Filter ──
    search_col, filter_col = st.columns([2, 3])

    with search_col:
        search_query = st.text_input("", placeholder="Cari nama file...",
                                     label_visibility="collapsed", key="search_history")

    with filter_col:
        filter_options = ["Semua", "Low Risk", "Medium Risk", "High Risk"]
        filter_cols = st.columns(len(filter_options))
        for i, opt in enumerate(filter_options):
            with filter_cols[i]:
                btn_type = "primary" if st.session_state.filter_risk == opt else "secondary"
                if st.button(opt, key=f"filter_{opt}", use_container_width=True,
                             type=btn_type):
                    st.session_state.filter_risk = opt
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filter & Search Logic ──
    filtered = st.session_state.history.copy()
    if st.session_state.filter_risk != "Semua":
        filtered = [r for r in filtered if r["level_risiko"] == st.session_state.filter_risk]
    if search_query:
        filtered = [r for r in filtered if search_query.lower() in r["filename"].lower()]

    # ── Table Header ──
    h1, h2, h3, h4, h5 = st.columns([3, 1.5, 1.2, 1.2, 0.8])
    with h1:
        st.markdown('<div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:1px; padding:8px 0;">NAMA FILE</div>', unsafe_allow_html=True)
    with h2:
        st.markdown('<div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:1px; padding:8px 0;">TANGGAL</div>', unsafe_allow_html=True)
    with h3:
        st.markdown('<div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:1px; padding:8px 0;">DURASI</div>', unsafe_allow_html=True)
    with h4:
        st.markdown('<div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:1px; padding:8px 0;">LEVEL RISIKO</div>', unsafe_allow_html=True)
    with h5:
        st.markdown('<div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:1px; padding:8px 0;">AKSI</div>', unsafe_allow_html=True)

    st.markdown('<div style="border-bottom:1px solid rgba(100,116,139,0.15); margin-bottom:4px;"></div>',
                unsafe_allow_html=True)

    # ── Rows ──
    if not filtered:
        st.markdown("""
        <div style="text-align:center; padding:40px; color:#64748B;">
            <div style="font-size:32px; margin-bottom:8px;"></div>
            <div>Tidak ada hasil yang cocok dengan pencarian Anda.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for rec in filtered:
            r1, r2, r3, r4, r5 = st.columns([3, 1.5, 1.2, 1.2, 0.8])
            with r1:
                st.markdown(f'<div style="font-size:13px; color:#F8FAFC; font-weight:500; padding:10px 0;">{rec["filename"]}</div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div style="font-size:13px; color:#CBD5E1; padding:10px 0;">{rec["tanggal"]}</div>', unsafe_allow_html=True)
            with r3:
                st.markdown(f'<div style="font-size:13px; color:#CBD5E1; padding:10px 0;">{rec["durasi"]}</div>', unsafe_allow_html=True)
            with r4:
                st.markdown(f'<div style="padding:8px 0;">{render_badge(rec["level_risiko"])}</div>', unsafe_allow_html=True)
            with r5:
                if st.button("Buka", key=f"hist_{rec['filename']}"):
                    st.session_state.selected_file = rec["filename"]
                    st.session_state.page = "Deep Analysis"
                    st.session_state.processing = False
                    st.rerun()

            st.markdown('<div style="border-bottom:1px solid rgba(100,116,139,0.06); margin:0;"></div>',
                        unsafe_allow_html=True)

    # ── Pagination ──
    st.markdown("<br>", unsafe_allow_html=True)
    pg_cols = st.columns([4, 2, 4])
    with pg_cols[1]:
        st.markdown("""
        <div class="pagination">
            <div class="page-btn">‹</div>
            <div class="page-btn active">1</div>
            <div class="page-btn">2</div>
            <div class="page-btn">3</div>
            <div class="page-btn">›</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  ROUTING & RENDER
# ════════════════════════════════════════════════════════════════════════════════

render_navbar()

if st.session_state.page == "Dashboard":
    page_dashboard()
elif st.session_state.page == "Deep Analysis":
    page_deep_analysis()
elif st.session_state.page == "History":
    page_history()


# ── Footer ──
st.markdown("""
<div style="text-align:center; padding:32px 0 16px 0; border-top:1px solid rgba(100,116,139,0.1); margin-top:48px;">
    <div style="font-size:12px; color:#475569;">
        HUMIC VoiceGuard v1.0 — Audio Bullying Detection System<br>
    </div>
</div>
""", unsafe_allow_html=True)
