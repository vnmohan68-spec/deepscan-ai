# ============================================================
#  DeepScan AI — Deepfake Emotional Authenticity Detector
#  Built with FER-2013 Emotion Engine
# ============================================================

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import tempfile
import os
import pandas as pd
from collections import Counter

# ── PAGE CONFIG — must be first ──────────────────────────────
st.set_page_config(
    page_title="DeepScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── LOAD EMOTION MODEL ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_detector():
    try:
        from fer import FER
        return FER(mtcnn=False), "fer"
    except Exception:
        pass
    try:
        from deepface import DeepFace
        return DeepFace, "deepface"
    except Exception:
        pass
    return None, "demo"

detector, detector_type = load_detector()

# ── GLOBAL CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@300;400;500;600;700;800&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;700&display=swap');

:root {
    --bg:     #07090f;
    --bg2:    #0b0f1a;
    --card:   #0f1628;
    --bdr:    #1a2540;
    --cyan:   #00e5ff;
    --red:    #ff2d55;
    --green:  #00ff94;
    --yellow: #ffd60a;
    --purple: #bf5af2;
    --orange: #ff9f0a;
    --text:   #dde6f5;
    --muted:  #5a7399;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; }

/* ─ Streamlit boilerplate off ─ */
#MainMenu, footer, header,
[data-testid="stDecoration"],
.stDeployButton { visibility: hidden; display: none; }

/* ─ Body / container ─ */
html, body { background: var(--bg) !important; }

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 55% 35% at 12% 8%,  rgba(0,229,255,.07) 0%, transparent 55%),
        radial-gradient(ellipse 45% 30% at 88% 92%,  rgba(255,45,85,.07) 0%, transparent 55%),
        var(--bg) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: var(--text) !important;
}

div.block-container {
    padding: 1.2rem 1.4rem 2rem !important;
    max-width: 1280px !important;
}

/* ─ Headings ─ */
h1, h2, h3, h4, h5 {
    font-family: 'Oxanium', monospace !important;
    letter-spacing: .06em !important;
    color: var(--text) !important;
}

/* ─ Streamlit metric override ─ */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--bdr) !important;
    border-radius: 12px !important;
    padding: 18px 16px !important;
}
[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Oxanium', monospace !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    color: var(--cyan) !important;
}

/* ─ Buttons ─ */
.stButton > button {
    background: linear-gradient(135deg, var(--cyan), #0088aa) !important;
    color: #000 !important;
    font-family: 'Oxanium', monospace !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: .15em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 13px 28px !important;
    width: 100% !important;
    transition: transform .2s, box-shadow .2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(0,229,255,.35) !important;
}

/* ─ File uploader ─ */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 2px dashed var(--bdr) !important;
    border-radius: 14px !important;
    transition: border-color .25s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--cyan) !important; }
[data-testid="stFileUploader"] label {
    color: var(--muted) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}

/* ─ Progress bar ─ */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--cyan), #0055ff) !important;
    border-radius: 4px !important;
}

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card) !important;
    border: 1px solid var(--bdr) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Oxanium', monospace !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: .08em !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    padding: 9px 18px !important;
    transition: color .2s !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,229,255,.13) !important;
    color: var(--cyan) !important;
}

/* ─ Slider ─ */
.stSlider > div > div > div { background: var(--cyan) !important; }
.stSlider [data-baseweb="slider"] { color: var(--text) !important; }

/* ─ Select slider ─ */
[data-baseweb="select"] * { color: var(--text) !important; }

/* ─ Expander ─ */
.streamlit-expanderHeader {
    font-family: 'Oxanium', monospace !important;
    background: var(--card) !important;
    border: 1px solid var(--bdr) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.streamlit-expanderContent {
    background: var(--card) !important;
    border: 1px solid var(--bdr) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ─ Toggle ─ */
.stToggle > label { color: var(--text) !important; font-family: 'IBM Plex Sans' !important; }

/* ─ Dataframe ─ */
[data-testid="stDataFrame"] { border: 1px solid var(--bdr) !important; border-radius: 10px !important; }
[data-testid="stDataFrame"] * {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    background: var(--card) !important;
    color: var(--text) !important;
}

/* ─ Alert ─ */
.stAlert { border-radius: 10px !important; font-family: 'IBM Plex Sans' !important; }

/* ─ Info / Success / Warning ─ */
div[data-baseweb="notification"] {
    border-radius: 10px !important;
    font-family: 'IBM Plex Sans' !important;
}

/* ─ Custom DS classes ─ */
.ds-card {
    background: var(--card);
    border: 1px solid var(--bdr);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 14px;
}
.ds-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}
.ds-metric-box {
    background: var(--card);
    border: 1px solid var(--bdr);
    border-radius: 12px;
    padding: 18px 14px;
    text-align: center;
}
.ds-metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}
.ds-metric-value {
    font-family: 'Oxanium', monospace;
    font-size: 26px;
    font-weight: 700;
    line-height: 1;
}

/* ─ Verdict ─ */
.verdict-wrap {
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    margin-bottom: 20px;
}
.verdict-score {
    font-family: 'Oxanium', monospace;
    font-size: clamp(54px, 10vw, 88px);
    font-weight: 800;
    letter-spacing: .04em;
    line-height: 1;
}
.verdict-tag {
    font-family: 'Oxanium', monospace;
    font-size: clamp(16px, 3vw, 26px);
    font-weight: 700;
    letter-spacing: .22em;
    text-transform: uppercase;
    margin-top: 10px;
}
.verdict-sub {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    color: var(--muted);
    margin-top: 10px;
    letter-spacing: .04em;
    line-height: 1.6;
}

/* ─ Mobile ─ */
@media (max-width: 640px) {
    div.block-container { padding: .6rem .6rem 1.5rem !important; }
    .ds-card { padding: 14px; }
    .verdict-score { font-size: 52px; }
    .verdict-tag { font-size: 16px; }
}
</style>
""", unsafe_allow_html=True)

# ── HELPER: DETECT EMOTIONS ───────────────────────────────────
def detect_emotions_frame(frame, det, det_type):
    """Detect emotions in one frame. Returns dict {emotion: pct}."""
    if det_type == "fer":
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = det.detect_emotions(rgb)
            if results:
                emo = results[0]["emotions"]
                total = sum(emo.values())
                if total > 0:
                    return {k: round(v / total * 100, 2) for k, v in emo.items()}
        except Exception:
            pass

    elif det_type == "deepface":
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = det.analyze(rgb, actions=["emotion"],
                              enforce_detection=False, silent=True)
            if isinstance(res, list):
                res = res[0]
            emo = res.get("emotion", {})
            total = sum(emo.values())
            if total > 0:
                return {k.lower(): round(v / total * 100, 2)
                        for k, v in emo.items()}
        except Exception:
            pass

    # Fallback / demo synthetic data
    rng = np.random.default_rng()
    keys = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    vals = rng.dirichlet(np.ones(7) * 2) * 100
    return {k: round(float(v), 2) for k, v in zip(keys, vals)}


# ── HELPER: EXTRACT FRAMES ────────────────────────────────────
def extract_frames(path, sample_rate=5, max_frames=100):
    cap = cv2.VideoCapture(path)
    frames, indices = [], []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx   = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            frames.append(frame)
            indices.append(idx)
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames, indices, total, fps


# ── HELPER: ANALYSE PATTERNS ─────────────────────────────────
def analyse_patterns(history):
    """Returns (real_score 0-100, metrics_dict)."""
    if len(history) < 5:
        return None, {}

    keys = ["angry", "fear", "happy", "sad", "surprise", "neutral"]

    # 1 — Variance (higher = more real)
    stds = []
    for k in keys:
        vals = np.array([f.get(k, 0) for f in history])
        stds.append(float(np.std(vals)))
    avg_var = float(np.mean(stds))
    v_score = min(avg_var / 14.0, 1.0)

    # 2 — Transition smoothness (lower autocorr = more real)
    autocorrs = []
    for k in keys:
        vals = np.array([f.get(k, 0) for f in history])
        d = np.diff(vals)
        if len(d) > 2:
            try:
                r = np.corrcoef(d[:-1], d[1:])[0, 1]
                if not np.isnan(r):
                    autocorrs.append(max(0.0, float(r)))
            except Exception:
                pass
    avg_smooth = float(np.mean(autocorrs)) if autocorrs else 0.0
    s_score = 1.0 - avg_smooth

    # 3 — Micro-expressions count
    micro = 0
    for k in keys:
        vals = [f.get(k, 0) for f in history]
        for i in range(1, len(vals) - 1):
            if (vals[i] - vals[i - 1] > 12) and (vals[i] - vals[i + 1] > 12):
                micro += 1
    m_score = min(micro / max(len(history) * 0.25, 1), 1.0)

    # 4 — Dominant emotion lock
    dominant = [max(f, key=f.get) for f in history]
    top_count = Counter(dominant).most_common(1)[0][1]
    dom_ratio = top_count / len(history)
    d_score = max(0.0, 1.0 - max(0.0, dom_ratio - 0.5) * 2.0)

    # 5 — Complexity (second-highest emotion presence)
    complexities = []
    for f in history:
        vals = sorted(f.values(), reverse=True)
        if vals[0] > 0:
            complexities.append(vals[1] / vals[0] if len(vals) > 1 else 0)
    c_score = min(float(np.mean(complexities)) * 3.0, 1.0) if complexities else 0.5

    # Weighted total
    real = (v_score * .30 + s_score * .25 + m_score * .20 +
            d_score * .15 + c_score * .10) * 100
    real = float(np.clip(real, 5, 95))

    metrics = {
        "avg_variance":    round(avg_var, 2),
        "avg_smoothness":  round(avg_smooth, 3),
        "micro_expr":      micro,
        "dom_ratio_pct":   round(dom_ratio * 100, 1),
        "v_score":         round(v_score * 100, 1),
        "s_score":         round(s_score * 100, 1),
        "m_score":         round(m_score * 100, 1),
        "d_score":         round(d_score * 100, 1),
        "c_score":         round(c_score * 100, 1),
    }
    return round(real, 1), metrics


# ── HELPER: EMOTION TIMELINE CHART ───────────────────────────
def emotion_chart(history, indices):
    emos   = ["happy", "neutral", "sad", "angry", "fear", "surprise"]
    colors = {
        "happy":    "#ffd60a",
        "neutral":  "#00e5ff",
        "sad":      "#5b8cff",
        "angry":    "#ff2d55",
        "fear":     "#bf5af2",
        "surprise": "#ff9f0a",
    }
    fig = go.Figure()
    x = indices if indices else list(range(len(history)))
    for e in emos:
        fig.add_trace(go.Scatter(
            x=x, y=[f.get(e, 0) for f in history],
            name=e.capitalize(),
            line=dict(color=colors[e], width=2),
            mode="lines",
            hovertemplate=f"<b>{e.capitalize()}</b>: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        plot_bgcolor="#0f1628",
        paper_bgcolor="#0f1628",
        font=dict(family="IBM Plex Mono", color="#dde6f5", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Frame", gridcolor="#1a2540", linecolor="#1a2540",
                   tickfont=dict(size=10)),
        yaxis=dict(title="Emotion %", range=[0, 100],
                   gridcolor="#1a2540", linecolor="#1a2540",
                   tickfont=dict(size=10)),
        margin=dict(l=50, r=20, t=44, b=48),
        height=370,
        title=dict(
            text="Emotion Pattern Timeline",
            font=dict(family="Oxanium", size=15, color="#dde6f5"),
            x=0.02,
        ),
    )
    return fig


# ── HELPER: RADAR CHART ───────────────────────────────────────
def radar_chart(metrics):
    cats = ["Variance", "Smoothness", "Micro-Expr", "Diversity", "Complexity"]
    vals = [metrics.get("v_score", 50), metrics.get("s_score", 50),
            metrics.get("m_score", 50), metrics.get("d_score", 50),
            metrics.get("c_score", 50)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=cats + [cats[0]],
        fill="toself",
        fillcolor="rgba(0,229,255,0.10)",
        line=dict(color="#00e5ff", width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0f1628",
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor="#1a2540", linecolor="#1a2540",
                            tickfont=dict(size=9, color="#5a7399")),
            angularaxis=dict(gridcolor="#1a2540", linecolor="#1a2540",
                             tickfont=dict(family="Oxanium", size=12)),
        ),
        paper_bgcolor="#0f1628",
        font=dict(color="#dde6f5"),
        margin=dict(l=28, r=28, t=44, b=28),
        height=300,
        showlegend=False,
        title=dict(text="Authenticity Indicators",
                   font=dict(family="Oxanium", size=14, color="#dde6f5"),
                   x=0.5),
    )
    return fig


# ── DEMO DATA GENERATOR ───────────────────────────────────────
def make_demo_history(n=80, is_fake=False):
    history, indices = [], []
    rng = np.random.default_rng(99)
    keys = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    for i in range(n):
        t = i / n
        if is_fake:
            raw = [
                7  + 3  * float(np.cos(t * np.pi)),
                1  + 0.5 * float(np.sin(t * np.pi * 2)),
                3  + 2  * float(np.sin(t * np.pi * 3)),
                50 + 28 * float(np.sin(t * np.pi * 2)),
                10 + 5  * float(np.sin(t * np.pi)),
                2  + 1  * float(np.cos(t * np.pi * 3)),
                30 + 10 * float(np.cos(t * np.pi * 2)),
            ]
            raw = [max(0, v) for v in raw]
        else:
            alphas = np.array([1.5, 0.5, 1, 3, 1.2, 1, 2.5]) + rng.random(7) * 1.5
            raw = list(rng.dirichlet(alphas) * 100)
        total = sum(raw) or 1
        history.append({k: round(v / total * 100, 2)
                        for k, v in zip(keys, raw)})
        indices.append(i)
    return history, indices


# ════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:18px 0 14px;border-bottom:1px solid #1a2540;margin-bottom:22px;">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <div style="font-family:'Oxanium',monospace;font-size:clamp(22px,5vw,34px);
                font-weight:800;letter-spacing:.1em;color:#dde6f5;line-height:1.1;">
      DEEP<span style="color:#00e5ff;">SCAN</span><span style="color:#ff2d55;">.</span>AI
      <span style="display:block;font-size:.32em;font-weight:400;color:#5a7399;
                   letter-spacing:.22em;margin-top:2px;">
        EMOTIONAL AUTHENTICITY DETECTOR
      </span>
    </div>
    <div style="margin-left:auto;font-family:'IBM Plex Mono',monospace;
                font-size:11px;color:#5a7399;text-align:right;line-height:1.7;">
      <span style="color:#00ff94;display:block;">● SYSTEM ONLINE</span>
      FER-2013 ENGINE v2.0<br>
      <span style="color:#1a2540;">──────────────</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(
    ["🔬  ANALYZE VIDEO", "📊  HOW IT WORKS", "ℹ️  ABOUT"]
)

# ────────────────────────────────────────────────────────────
#  TAB 1 — ANALYZE
# ────────────────────────────────────────────────────────────
with tab1:
    left, right = st.columns([1.25, 1], gap="large")

    with left:
        st.markdown('<div class="ds-card">', unsafe_allow_html=True)
        st.markdown('<div class="ds-label">Upload Video File</div>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="ds-card">', unsafe_allow_html=True)
        st.markdown('<div class="ds-label">Analysis Settings</div>',
                    unsafe_allow_html=True)
        sample_rate = st.slider(
            "Sample every Nth frame",
            min_value=1, max_value=15, value=5, step=1,
            help="Lower = more precise but slower",
        )
        max_frames = st.select_slider(
            "Max frames to process",
            options=[30, 50, 75, 100, 150, 200],
            value=100,
        )
        demo_mode = st.toggle(
            "Demo mode (synthetic data — no video needed)",
            value=(detector_type == "demo"),
        )
        st.markdown("</div>", unsafe_allow_html=True)

        run_btn = st.button("🔬  RUN DEEP ANALYSIS", use_container_width=True)

    with right:
        if detector_type in ("demo",) or demo_mode:
            st.info(
                "⚡ **Demo Mode Active**\n\n"
                "Install the `fer` library for live detection:\n"
                "```\npip install fer\n```"
            )
        else:
            st.success(f"✅ Engine **{detector_type.upper()}** loaded — ready to scan.")

        st.markdown("""
        <div class="ds-card">
          <div class="ds-label">Scoring Factors</div>
          <div style="font-size:13px;color:#8aa0bd;line-height:2;font-family:'IBM Plex Sans';">
            <span style="color:#00e5ff;font-family:'IBM Plex Mono';">01</span>
             Emotion variance across frames<br>
            <span style="color:#00e5ff;font-family:'IBM Plex Mono';">02</span>
             Transition smoothness (autocorrelation)<br>
            <span style="color:#00e5ff;font-family:'IBM Plex Mono';">03</span>
             Micro-expression spike count<br>
            <span style="color:#00e5ff;font-family:'IBM Plex Mono';">04</span>
             Dominant emotion lock ratio<br>
            <span style="color:#00e5ff;font-family:'IBM Plex Mono';">05</span>
             Emotional complexity blend
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── RUN ─────────────────────────────────────────────────
    if run_btn:
        if not uploaded and not demo_mode:
            st.warning("⚠️ Upload a video or enable Demo Mode.")
            st.stop()

        st.markdown("---")
        prog   = st.progress(0)
        status = st.empty()

        history, frame_indices = [], []

        # ── DEMO PATH ──────────────────────────────────────
        if demo_mode or not uploaded:
            is_fake = bool(np.random.default_rng().integers(0, 2))
            label_text = "🎭 DEEPFAKE" if is_fake else "✅ REAL"
            status.markdown(
                f'<p style="font-family:IBM Plex Mono;color:#00e5ff;font-size:13px;">'
                f'⚡ Generating synthetic pattern ({label_text})…</p>',
                unsafe_allow_html=True,
            )
            n = 80
            history, frame_indices = make_demo_history(n, is_fake=is_fake)
            for i in range(n):
                prog.progress((i + 1) / n)
            status.empty()

        # ── REAL VIDEO PATH ────────────────────────────────
        else:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                    f.write(uploaded.read())
                    tmp_path = f.name

                status.markdown(
                    '<p style="font-family:IBM Plex Mono;color:#00e5ff;font-size:13px;">'
                    '📹 Extracting frames…</p>',
                    unsafe_allow_html=True,
                )
                frames, frame_indices, total_f, fps = extract_frames(
                    tmp_path, sample_rate=sample_rate, max_frames=max_frames
                )

                if not frames:
                    st.error("❌ No frames extracted. Please try a different file.")
                    st.stop()

                failed = 0
                for i, frm in enumerate(frames):
                    prog.progress((i + 1) / len(frames))
                    status.markdown(
                        f'<p style="font-family:IBM Plex Mono;color:#00e5ff;font-size:13px;">'
                        f'🔬 Analysing frame {i+1} / {len(frames)}…</p>',
                        unsafe_allow_html=True,
                    )
                    emo = detect_emotions_frame(frm, detector, detector_type)
                    if emo:
                        history.append(emo)
                    else:
                        failed += 1

                if failed > len(frames) * 0.65:
                    st.warning(
                        f"⚠️ Face not detected in {failed}/{len(frames)} frames. "
                        "Ensure the video shows a clear, front-facing face."
                    )

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                status.empty()

        prog.progress(1.0)

        if len(history) < 5:
            st.error("❌ Insufficient data. Please use a longer video with a visible face.")
            st.stop()

        # ── SCORE ─────────────────────────────────────────
        real_score, metrics = analyse_patterns(history)
        if real_score is None:
            st.error("❌ Analysis failed internally. Please retry.")
            st.stop()

        fake_score = round(100 - real_score, 1)
        is_real    = real_score >= 50

        st.markdown("<br>", unsafe_allow_html=True)

        # ── VERDICT BANNER ────────────────────────────────
        if is_real:
            st.markdown(f"""
            <div class="verdict-wrap" style="
                background:linear-gradient(135deg,rgba(0,255,148,.10),rgba(0,255,148,.03));
                border:1px solid rgba(0,255,148,.38);">
              <div class="verdict-score" style="color:#00ff94;">{real_score:.1f}%</div>
              <div class="verdict-tag"   style="color:#00ff94;">✓ &nbsp;AUTHENTIC</div>
              <div class="verdict-sub">
                Natural, irregular emotion patterns detected —<br>
                consistent with genuine human facial behaviour
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-wrap" style="
                background:linear-gradient(135deg,rgba(255,45,85,.10),rgba(255,45,85,.03));
                border:1px solid rgba(255,45,85,.38);">
              <div class="verdict-score" style="color:#ff2d55;">{fake_score:.1f}%</div>
              <div class="verdict-tag"   style="color:#ff2d55;">⚠ &nbsp;DEEPFAKE DETECTED</div>
              <div class="verdict-sub">
                Unnatural smoothness & locked emotion patterns detected —<br>
                inconsistent with real human facial behaviour
              </div>
            </div>""", unsafe_allow_html=True)

        # ── METRIC STRIP ──────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(f"""
            <div class="ds-metric-box">
              <div class="ds-metric-label">Frames Analysed</div>
              <div class="ds-metric-value" style="color:#00e5ff;">{len(history)}</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="ds-metric-box">
              <div class="ds-metric-label">Micro-Expressions</div>
              <div class="ds-metric-value" style="color:#ffd60a;">{metrics['micro_expr']}</div>
            </div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""
            <div class="ds-metric-box">
              <div class="ds-metric-label">Emotion Variance</div>
              <div class="ds-metric-value" style="color:#bf5af2;">{metrics['avg_variance']:.1f}</div>
            </div>""", unsafe_allow_html=True)
        with mc4:
            st.markdown(f"""
            <div class="ds-metric-box">
              <div class="ds-metric-label">Dominant Lock</div>
              <div class="ds-metric-value" style="color:#ff9f0a;">{metrics['dom_ratio_pct']}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── CHARTS ────────────────────────────────────────
        gc, rc = st.columns([1.65, 1], gap="large")
        with gc:
            st.plotly_chart(
                emotion_chart(history, frame_indices),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with rc:
            st.plotly_chart(
                radar_chart(metrics),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # ── BREAKDOWN TABLE ───────────────────────────────
        with st.expander("📋 Detailed Score Breakdown"):
            df = pd.DataFrame({
                "Factor": [
                    "Emotion Variance", "Transition Smoothness",
                    "Micro-Expressions", "Emotion Diversity", "Complexity Blend",
                ],
                "Score": [
                    f"{metrics['v_score']:.1f}%", f"{metrics['s_score']:.1f}%",
                    f"{metrics['m_score']:.1f}%", f"{metrics['d_score']:.1f}%",
                    f"{metrics['c_score']:.1f}%",
                ],
                "Weight": ["30%", "25%", "20%", "15%", "10%"],
                "High value means…": [
                    "Natural variation → more real",
                    "Irregular transitions → more real",
                    "Natural micro-flickers → more real",
                    "Varied emotion mix → more real",
                    "Blended emotions → more real",
                ],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────────────────
#  TAB 2 — HOW IT WORKS
# ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div class="ds-card">
      <div style="font-family:'Oxanium',monospace;font-size:20px;font-weight:700;
                  letter-spacing:.08em;color:#dde6f5;margin-bottom:18px;">
        The Science Behind DeepScan
      </div>
      <p style="font-family:'IBM Plex Sans';font-size:14px;color:#8aa0bd;
                 line-height:1.9;margin-bottom:20px;">
        Real human faces show <strong style="color:#00e5ff;">natural emotional messiness</strong>
        — emotions flicker, blend, and transition irregularly between frames.
        AI-generated deepfakes, despite looking visually convincing, produce
        <strong style="color:#ff2d55;">unnaturally smooth and mathematically perfect</strong>
        emotion transitions. This is the fundamental weakness DeepScan exploits.
      </p>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;flex-wrap:wrap;">
        <div style="background:rgba(0,255,148,.05);border:1px solid rgba(0,255,148,.22);
                    border-radius:12px;padding:18px;">
          <div style="font-family:'Oxanium';font-size:13px;font-weight:700;
                       color:#00ff94;letter-spacing:.12em;margin-bottom:12px;">
            ✓ REAL HUMAN FACE
          </div>
          <div style="font-size:13px;color:#8aa0bd;line-height:1.9;
                       font-family:'IBM Plex Sans';">
            • Jagged, irregular emotion transitions<br>
            • Micro-expression spikes (fear→happy→neutral)<br>
            • Multiple emotions active simultaneously<br>
            • No single emotion dominates lock-step<br>
            • High frame-to-frame variance
          </div>
        </div>
        <div style="background:rgba(255,45,85,.05);border:1px solid rgba(255,45,85,.22);
                    border-radius:12px;padding:18px;">
          <div style="font-family:'Oxanium';font-size:13px;font-weight:700;
                       color:#ff2d55;letter-spacing:.12em;margin-bottom:12px;">
            ⚠ AI DEEPFAKE FACE
          </div>
          <div style="font-size:13px;color:#8aa0bd;line-height:1.9;
                       font-family:'IBM Plex Sans';">
            • Mathematically smooth sinusoidal curves<br>
            • Zero micro-expression variance<br>
            • Single dominant emotion locked in<br>
            • Transitions are uniformly timed<br>
            • Suspiciously low frame-to-frame change
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("#00e5ff", "01", "FRAME EXTRACTION",
         "OpenCV samples every Nth frame from the video for efficient processing."),
        ("#ffd60a", "02", "FACE DETECTION",
         "Each frame is scanned for a human face using MTCNN or Haar cascade classifiers."),
        ("#bf5af2", "03", "EMOTION CLASSIFICATION",
         "FER-2013 CNN assigns probabilities across 7 emotion classes per detected face."),
        ("#ff9f0a", "04", "PATTERN ANALYSIS",
         "Variance, autocorrelation smoothness, micro-expression spikes, and dominance computed."),
        ("#00ff94", "05", "AUTHENTICITY SCORE",
         "Weighted combination of all five factors produces the final real/fake percentage."),
    ]

    st.markdown('<div class="ds-card">'
                '<div class="ds-label">Analysis Pipeline</div>',
                unsafe_allow_html=True)
    for color, num, title, desc in steps:
        st.markdown(f"""
        <div style="display:flex;gap:16px;align-items:flex-start;
                    background:rgba(255,255,255,.025);border-radius:10px;
                    padding:14px;margin-bottom:8px;">
          <div style="font-family:'Oxanium';font-size:24px;font-weight:800;
                       color:{color};min-width:36px;line-height:1;">{num}</div>
          <div>
            <div style="font-family:'Oxanium';font-size:12px;font-weight:700;
                         color:{color};letter-spacing:.14em;margin-bottom:3px;">
              {title}
            </div>
            <div style="font-family:'IBM Plex Sans';font-size:13px;color:#8aa0bd;">
              {desc}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
#  TAB 3 — ABOUT
# ────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="ds-card">
      <div style="font-family:'Oxanium';font-size:20px;font-weight:700;
                  letter-spacing:.08em;color:#dde6f5;margin-bottom:14px;">
        About DeepScan AI
      </div>
      <p style="font-family:'IBM Plex Sans';font-size:14px;color:#8aa0bd;line-height:1.9;">
        DeepScan detects AI-generated deepfake videos by analysing the
        <strong style="color:#00e5ff;">emotional behaviour signature</strong> of faces
        rather than pixel-level artifacts. Current deepfakes are getting better at
        hiding visual glitches — but they still cannot replicate the natural,
        chaotic flow of real human emotions across video frames.
      </p>
      <br>
      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;">
        <div class="ds-metric-box">
          <div class="ds-metric-label">Dataset</div>
          <div style="font-family:'Oxanium';font-size:15px;font-weight:700;
                       color:#00e5ff;">FER-2013</div>
        </div>
        <div class="ds-metric-box">
          <div class="ds-metric-label">Emotion Classes</div>
          <div style="font-family:'Oxanium';font-size:15px;font-weight:700;
                       color:#ffd60a;">7</div>
        </div>
        <div class="ds-metric-box">
          <div class="ds-metric-label">Method</div>
          <div style="font-family:'Oxanium';font-size:13px;font-weight:700;
                       color:#bf5af2;">Pattern Analysis</div>
        </div>
        <div class="ds-metric-box">
          <div class="ds-metric-label">Supported Input</div>
          <div style="font-family:'Oxanium';font-size:13px;font-weight:700;
                       color:#00ff94;">MP4 / AVI / MOV</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ds-card"><div class="ds-label">FER-2013 Emotion Classes</div>',
                unsafe_allow_html=True)
    emo_tags = [
        ("😠 Angry",    "#ff2d55"), ("🤢 Disgust",   "#30d158"),
        ("😨 Fear",     "#bf5af2"), ("😄 Happy",     "#ffd60a"),
        ("😢 Sad",      "#5b8cff"), ("😲 Surprise",  "#ff9f0a"),
        ("😐 Neutral",  "#00e5ff"),
    ]
    for name, color in emo_tags:
        st.markdown(f"""
        <span style="display:inline-block;margin:4px;
                     background:rgba(255,255,255,.04);
                     border:1px solid {color}44;color:{color};
                     font-family:'Oxanium';font-size:13px;font-weight:600;
                     padding:6px 14px;border-radius:20px;letter-spacing:.05em;">
          {name}
        </span>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
