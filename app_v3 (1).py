# ============================================================
#  DeepScan AI v3.0 — Deepfake Emotional Authenticity Detector
#  Backend : Custom CNN (FER-2013) + OpenCV Haarcascade
#  All Plotly chart bugs fixed. No FER. No DeepFace.
# ============================================================

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import tempfile, os, time
import pandas as pd
from collections import Counter

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="DeepScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════
#  CNN MODEL LOADER
# ═══════════════════════════════════════════════════════════
MODEL_PATH   = "emotion_model.h5"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

@st.cache_resource(show_spinner=False)
def load_cnn_model():
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if not os.path.exists(MODEL_PATH):
        return None, face_cascade, "demo"
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_USE_LEGACY_KERAS"]   = "1"
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        # Try standard load first
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            return model, face_cascade, "cnn"
        except Exception:
            pass

        # Fallback: load with safe_mode disabled (TF 2.16+)
        try:
            model = tf.keras.models.load_model(
                MODEL_PATH, compile=False, safe_mode=False
            )
            return model, face_cascade, "cnn"
        except Exception:
            pass

        # Fallback: load weights only into fresh architecture
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (
                Conv2D, MaxPooling2D, BatchNormalization,
                Activation, Flatten, Dense, Dropout
            )
            m = Sequential([
                Conv2D(32,(3,3),padding="same",input_shape=(48,48,1)),
                BatchNormalization(), Activation("relu"),
                Conv2D(32,(3,3),padding="same"),
                BatchNormalization(), Activation("relu"),
                MaxPooling2D(2,2), Dropout(0.25),

                Conv2D(64,(3,3),padding="same"),
                BatchNormalization(), Activation("relu"),
                Conv2D(64,(3,3),padding="same"),
                BatchNormalization(), Activation("relu"),
                MaxPooling2D(2,2), Dropout(0.25),

                Conv2D(128,(3,3),padding="same"),
                BatchNormalization(), Activation("relu"),
                Conv2D(128,(3,3),padding="same"),
                BatchNormalization(), Activation("relu"),
                MaxPooling2D(2,2), Dropout(0.25),

                Flatten(),
                Dense(256), BatchNormalization(), Activation("relu"), Dropout(0.5),
                Dense(128), BatchNormalization(), Activation("relu"), Dropout(0.3),
                Dense(7, activation="softmax"),
            ])
            m.load_weights(MODEL_PATH)
            return m, face_cascade, "cnn"
        except Exception:
            pass

        return None, face_cascade, "demo"

    except Exception:
        return None, face_cascade, "demo"

cnn_model, face_cascade, engine_mode = load_cnn_model()

# ── CONSTANTS ─────────────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMG_SIZE = 48

EMO_COLORS = {
    "happy":    "#ffd60a",
    "neutral":  "#00e5ff",
    "sad":      "#5b8cff",
    "angry":    "#ff2d55",
    "fear":     "#bf5af2",
    "surprise": "#ff9f0a",
    "disgust":  "#30d158",
}

RISK_MAP = {
    "CRITICAL": ("#ff2d55", "rgba(255,45,85,.15)"),
    "HIGH":     ("#ff9f0a", "rgba(255,159,10,.12)"),
    "MEDIUM":   ("#ffd60a", "rgba(255,214,10,.12)"),
    "LOW":      ("#00ff94", "rgba(0,255,148,.12)"),
}

# ── GLOBAL CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@300;400;600;700;800&family=IBM+Plex+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root{
  --bg:#07090f;--bg2:#0b0f1a;--card:#0f1628;--card2:#131c2e;
  --bdr:#1a2540;--bdr2:#233050;
  --cyan:#00e5ff;--red:#ff2d55;--green:#00ff94;
  --yellow:#ffd60a;--purple:#bf5af2;--orange:#ff9f0a;
  --blue:#5b8cff;--text:#dde6f5;--muted:#5a7399;
}
*,*::before,*::after{box-sizing:border-box}
#MainMenu,footer,header,[data-testid="stDecoration"],.stDeployButton,
[data-testid="collapsedControl"]{visibility:hidden!important;display:none!important}
html,body{background:var(--bg)!important}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(ellipse 55% 35% at 12% 8%,rgba(0,229,255,.06) 0%,transparent 55%),
    radial-gradient(ellipse 45% 28% at 88% 92%,rgba(255,45,85,.06) 0%,transparent 55%),
    var(--bg)!important;
  font-family:'IBM Plex Sans',sans-serif!important;
  color:var(--text)!important;
}
div.block-container{padding:1.2rem 1.4rem 2rem!important;max-width:1320px!important}
h1,h2,h3,h4,h5{font-family:'Oxanium',monospace!important;letter-spacing:.06em!important;color:var(--text)!important}
[data-testid="metric-container"]{background:var(--card)!important;border:1px solid var(--bdr)!important;border-radius:12px!important;padding:18px 16px!important}
[data-testid="metric-container"] label{font-family:'IBM Plex Mono',monospace!important;font-size:10px!important;letter-spacing:.14em!important;text-transform:uppercase!important;color:var(--muted)!important}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'Oxanium',monospace!important;font-size:28px!important;font-weight:700!important;color:var(--cyan)!important}
.stButton>button{background:linear-gradient(135deg,var(--cyan),#0088aa)!important;color:#000!important;font-family:'Oxanium',monospace!important;font-weight:700!important;font-size:13px!important;letter-spacing:.14em!important;text-transform:uppercase!important;border:none!important;border-radius:10px!important;padding:13px 28px!important;width:100%!important;transition:transform .18s,box-shadow .18s!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 28px rgba(0,229,255,.3)!important}
[data-testid="stFileUploader"]{background:var(--card)!important;border:2px dashed var(--bdr)!important;border-radius:14px!important;transition:border-color .25s!important}
[data-testid="stFileUploader"]:hover{border-color:var(--cyan)!important}
[data-testid="stFileUploader"] label{color:var(--muted)!important;font-family:'IBM Plex Mono',monospace!important;font-size:11px!important}
.stProgress>div>div{background:linear-gradient(90deg,var(--cyan),#0055ff)!important;border-radius:4px!important}
.stTabs [data-baseweb="tab-list"]{background:var(--card)!important;border:1px solid var(--bdr)!important;border-radius:12px!important;padding:4px!important;gap:3px!important;overflow-x:auto!important}
.stTabs [data-baseweb="tab"]{font-family:'Oxanium',monospace!important;font-size:12px!important;font-weight:600!important;letter-spacing:.07em!important;color:var(--muted)!important;border-radius:8px!important;padding:9px 16px!important;transition:all .2s!important;white-space:nowrap!important}
.stTabs [aria-selected="true"]{background:rgba(0,229,255,.13)!important;color:var(--cyan)!important}
.stSlider>div>div>div{background:var(--cyan)!important}
[data-baseweb="select"] *{color:var(--text)!important;background:var(--card2)!important}
[data-baseweb="popover"] *{background:var(--card2)!important;color:var(--text)!important}
.streamlit-expanderHeader{font-family:'Oxanium',monospace!important;background:var(--card)!important;border:1px solid var(--bdr)!important;border-radius:10px!important;color:var(--text)!important;font-size:13px!important;letter-spacing:.06em!important}
.streamlit-expanderContent{background:var(--card)!important;border:1px solid var(--bdr)!important;border-top:none!important;border-radius:0 0 10px 10px!important}
[data-testid="stDataFrame"]{border:1px solid var(--bdr)!important;border-radius:10px!important}
[data-testid="stDataFrame"] *{font-family:'IBM Plex Mono',monospace!important;font-size:11px!important;background:var(--card)!important;color:var(--text)!important}
.stToggle>label{color:var(--text)!important;font-family:'IBM Plex Sans'!important;font-size:13px!important}
.stAlert{border-radius:10px!important;font-family:'IBM Plex Sans'!important}
hr{border:none!important;border-top:1px solid var(--bdr)!important;margin:14px 0!important}
[data-testid="stCameraInput"]{background:var(--card)!important;border:1px solid var(--bdr)!important;border-radius:14px!important}
.ds-card{background:var(--card);border:1px solid var(--bdr);border-radius:14px;padding:20px 22px;margin-bottom:14px;transition:border-color .22s}
.ds-card:hover{border-color:#233050}
.ds-label{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin-bottom:10px;display:block}
.ds-metric{background:var(--card);border:1px solid var(--bdr);border-radius:12px;padding:16px 12px;text-align:center}
.ds-metric-label{font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
.ds-metric-value{font-family:'Oxanium',monospace;font-size:26px;font-weight:700;line-height:1}
.verdict-real{background:linear-gradient(135deg,rgba(0,255,148,.10),rgba(0,255,148,.03));border:1px solid rgba(0,255,148,.38);border-radius:16px;padding:28px 20px;text-align:center;margin-bottom:14px}
.verdict-fake{background:linear-gradient(135deg,rgba(255,45,85,.10),rgba(255,45,85,.03));border:1px solid rgba(255,45,85,.38);border-radius:16px;padding:28px 20px;text-align:center;margin-bottom:14px}
.verdict-score{font-family:'Oxanium',monospace;font-size:clamp(52px,10vw,82px);font-weight:800;letter-spacing:.04em;line-height:1}
.verdict-tag{font-family:'Oxanium',monospace;font-size:clamp(14px,3vw,22px);font-weight:700;letter-spacing:.22em;text-transform:uppercase;margin-top:8px}
.verdict-sub{font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:var(--muted);margin-top:10px;line-height:1.6}
.risk-badge{display:inline-block;font-family:'Oxanium',monospace;font-weight:700;font-size:11px;letter-spacing:.18em;padding:5px 14px;border-radius:20px;text-transform:uppercase}
.xai-bar-track{background:var(--bdr);border-radius:4px;height:7px;overflow:hidden;margin-bottom:4px}
.xai-bar-fill{height:100%;border-radius:4px;transition:width .4s ease}
@media(max-width:640px){div.block-container{padding:.6rem .6rem 1.5rem!important}.ds-card{padding:14px}.verdict-score{font-size:52px}}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  CHART HELPER — prevents duplicate keyword errors
# ═══════════════════════════════════════════════════════════
_BASE_LAYOUT = dict(
    plot_bgcolor  = "#0f1628",
    paper_bgcolor = "#0f1628",
    font          = dict(family="IBM Plex Mono", color="#dde6f5", size=10),
    margin        = dict(l=48, r=16, t=44, b=44),
    legend        = dict(
        orientation="h", yanchor="bottom", y=1.01,
        xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
        font=dict(size=10),
    ),
)

_AXIS_STYLE = dict(gridcolor="#1a2540", linecolor="#1a2540", tickfont=dict(size=9))

def make_layout(**overrides) -> dict:
    """
    Merge base layout with overrides safely — no duplicate keys.
    xaxis / yaxis / title / height etc. are always passed as overrides.
    """
    layout = dict(_BASE_LAYOUT)
    layout.update(overrides)
    return layout

def hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ═══════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════
def preprocess_face(face_gray: np.ndarray) -> np.ndarray:
    face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    return face.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def predict_emotion_cnn(frame: np.ndarray) -> dict:
    if engine_mode == "demo" or cnn_model is None:
        rng  = np.random.default_rng()
        vals = rng.dirichlet(np.ones(7) * 2) * 100
        return {k: round(float(v), 2) for k, v in zip(EMOTIONS, vals)}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return {}

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi    = gray[y: y + h, x: x + w]
    if face_roi.size == 0:
        return {}

    try:
        blob  = preprocess_face(face_roi)
        preds = cnn_model.predict(blob, verbose=0)
        probs = preds[0].tolist()
        total = sum(probs) or 1.0
        return {k: round(float(v / total * 100), 2) for k, v in zip(EMOTIONS, probs)}
    except Exception:
        return {}


def extract_frames(path: str, sample_rate: int = 5, max_frames: int = 100):
    cap     = cv2.VideoCapture(path)
    frames, indices = [], []
    idx = 0
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
    return frames, indices


def analyse_patterns(history: list):
    if len(history) < 5:
        return None, {}

    keys = ["angry", "fear", "happy", "sad", "surprise", "neutral"]

    stds    = [float(np.std([f.get(k, 0) for f in history])) for k in keys]
    avg_var = float(np.mean(stds))
    v_score = min(avg_var / 14.0, 1.0)

    autocorrs = []
    for k in keys:
        vals = np.array([f.get(k, 0) for f in history])
        d    = np.diff(vals)
        if len(d) > 2:
            try:
                r = np.corrcoef(d[:-1], d[1:])[0, 1]
                if not np.isnan(r):
                    autocorrs.append(max(0.0, float(r)))
            except Exception:
                pass
    avg_smooth = float(np.mean(autocorrs)) if autocorrs else 0.0
    s_score    = 1.0 - avg_smooth

    micro = 0
    for k in keys:
        vals = [f.get(k, 0) for f in history]
        for i in range(1, len(vals) - 1):
            if (vals[i] - vals[i - 1] > 12) and (vals[i] - vals[i + 1] > 12):
                micro += 1
    m_score = min(micro / max(len(history) * 0.25, 1), 1.0)

    dominant  = [max(f, key=f.get) for f in history]
    top_count = Counter(dominant).most_common(1)[0][1]
    dom_ratio = top_count / len(history)
    d_score   = max(0.0, 1.0 - max(0.0, dom_ratio - 0.5) * 2.0)

    complexities = []
    for f in history:
        vals = sorted(f.values(), reverse=True)
        if vals[0] > 0:
            complexities.append(vals[1] / vals[0] if len(vals) > 1 else 0)
    c_score = min(float(np.mean(complexities)) * 3.0, 1.0) if complexities else 0.5

    real = (v_score * .30 + s_score * .25 + m_score * .20 +
            d_score * .15 + c_score * .10) * 100
    real = float(np.clip(real, 5, 95))

    return round(real, 1), {
        "avg_variance":   round(avg_var, 2),
        "avg_smoothness": round(avg_smooth, 3),
        "micro_expr":     micro,
        "dom_ratio_pct":  round(dom_ratio * 100, 1),
        "v_score": round(v_score * 100, 1),
        "s_score": round(s_score * 100, 1),
        "m_score": round(m_score * 100, 1),
        "d_score": round(d_score * 100, 1),
        "c_score": round(c_score * 100, 1),
    }


def make_demo_history(n: int = 80, is_fake: bool = False):
    history, indices = [], []
    rng = np.random.default_rng(42)
    for i in range(n):
        t = i / n
        if is_fake:
            raw = [
                7  + 3  * np.cos(t * np.pi),
                1  + .5 * np.sin(t * np.pi * 2),
                3  + 2  * np.sin(t * np.pi * 3),
                50 + 28 * np.sin(t * np.pi * 2),
                10 + 5  * np.sin(t * np.pi),
                2  + 1  * np.cos(t * np.pi * 3),
                30 + 10 * np.cos(t * np.pi * 2),
            ]
            raw = [max(0, v) for v in raw]
        else:
            alphas = np.array([1.5, .5, 1, 3, 1.2, 1, 2.5]) + rng.random(7) * 1.5
            raw    = list(rng.dirichlet(alphas) * 100)
        total = sum(raw) or 1
        history.append({k: round(v / total * 100, 2) for k, v in zip(EMOTIONS, raw)})
        indices.append(i)
    return history, indices


def risk_level(real_score: float) -> str:
    if real_score >= 65: return "LOW"
    if real_score >= 45: return "MEDIUM"
    if real_score >= 28: return "HIGH"
    return "CRITICAL"


def risk_badge_html(level: str) -> str:
    c, bg = RISK_MAP.get(level, ("#ffd60a", "rgba(255,214,10,.12)"))
    return (f'<span class="risk-badge" '
            f'style="background:{bg};border:1px solid {c}44;color:{c};">'
            f'⬡ {level} RISK</span>')


# ═══════════════════════════════════════════════════════════
#  CHART BUILDERS  — all use make_layout(), zero duplicate keys
# ═══════════════════════════════════════════════════════════
def timeline_chart(history, indices):
    fig = go.Figure()
    x   = indices or list(range(len(history)))
    for e, c in EMO_COLORS.items():
        fig.add_trace(go.Scatter(
            x=x, y=[f.get(e, 0) for f in history],
            name=e.capitalize(),
            line=dict(color=c, width=2),
            mode="lines",
            fill="tozeroy",
            fillcolor=hex_to_rgba(c, 0.08),
            hovertemplate=f"<b>{e.capitalize()}</b>: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(**make_layout(
        height=340,
        xaxis=dict(title="Frame", **_AXIS_STYLE),
        yaxis=dict(title="Emotion %", range=[0, 100], **_AXIS_STYLE),
        title=dict(
            text="Emotion Pattern Timeline",
            font=dict(family="Oxanium", size=14, color="#dde6f5"), x=0.02,
        ),
    ))
    return fig


def radar_chart(metrics, is_real):
    cats  = ["Variance", "Smoothness", "Micro-Expr", "Diversity", "Complexity"]
    vals  = [metrics.get("v_score", 50), metrics.get("s_score", 50),
             metrics.get("m_score", 50), metrics.get("d_score", 50),
             metrics.get("c_score", 50)]
    color = "#00ff94" if is_real else "#ff2d55"
    fig   = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself",
        fillcolor=hex_to_rgba(color, 0.13),
        line=dict(color=color, width=2),
    ))
    fig.update_layout(
        plot_bgcolor  = "#0f1628",
        paper_bgcolor = "#0f1628",
        font          = dict(color="#dde6f5"),
        margin        = dict(l=28, r=28, t=44, b=28),
        height        = 300,
        showlegend    = False,
        polar=dict(
            bgcolor="#0f1628",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="#1a2540", linecolor="#1a2540",
                tickfont=dict(size=9, color="#5a7399"),
            ),
            angularaxis=dict(
                gridcolor="#1a2540", linecolor="#1a2540",
                tickfont=dict(family="Oxanium", size=11),
            ),
        ),
        title=dict(
            text="Authenticity Indicators",
            font=dict(family="Oxanium", size=13, color="#dde6f5"), x=0.5,
        ),
    )
    return fig


def xai_chart(metrics, is_real):
    factors = ["Variance", "Smoothness", "Micro-Expr", "Diversity", "Complexity"]
    vals    = [metrics.get("v_score", 50), metrics.get("s_score", 50),
               metrics.get("m_score", 50), metrics.get("d_score", 50),
               metrics.get("c_score", 50)]
    colors  = ["#00e5ff", "#bf5af2", "#ffd60a", "#00ff94", "#ff9f0a"]
    fig = go.Figure(go.Bar(
        x=vals, y=factors, orientation="h",
        marker=dict(color=colors),
        hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(**make_layout(
        height=240,
        margin=dict(l=48, r=16, t=44, b=28),
        showlegend=False,
        xaxis=dict(range=[0, 100], **_AXIS_STYLE),
        yaxis=dict(tickfont=dict(size=11, family="IBM Plex Mono"),
                   **{k: v for k, v in _AXIS_STYLE.items() if k != "tickfont"}),
        title=dict(
            text="Factor Contribution",
            font=dict(family="Oxanium", size=13, color="#dde6f5"), x=0.02,
        ),
    ))
    return fig


def live_stream_chart(sim_history):
    fig = go.Figure()
    for e, c in EMO_COLORS.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(sim_history))),
            y=[f.get(e, 0) for f in sim_history],
            name=e.capitalize(),
            line=dict(color=c, width=1.6),
            mode="lines",
            hoverinfo="skip",
        ))
    fig.update_layout(**make_layout(
        height=260,
        xaxis=dict(title="Frame", **_AXIS_STYLE),
        yaxis=dict(range=[0, 100], **_AXIS_STYLE),
        title=dict(
            text="Live Emotion Stream",
            font=dict(family="Oxanium", size=13, color="#dde6f5"), x=0.02,
        ),
    ))
    return fig


def game_radar_chart(scores):
    cats = ["Variance", "Smoothness", "Micro-Expr", "Diversity", "Complexity"]
    vals = [scores["v"], scores["s"], scores["m"], scores["d"], scores["c"]]
    fig  = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself",
        fillcolor="rgba(0,229,255,0.12)",
        line=dict(color="#00e5ff", width=2),
    ))
    fig.update_layout(
        plot_bgcolor  = "#0f1628",
        paper_bgcolor = "#0f1628",
        font          = dict(color="#dde6f5"),
        margin        = dict(l=24, r=24, t=30, b=24),
        height        = 260,
        showlegend    = False,
        polar=dict(
            bgcolor="#0f1628",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="#1a2540", linecolor="#1a2540",
                tickfont=dict(size=9, color="#5a7399"),
            ),
            angularaxis=dict(
                gridcolor="#1a2540", linecolor="#1a2540",
                tickfont=dict(family="Oxanium", size=11),
            ),
        ),
    )
    return fig


# ═══════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════
engine_label = "CNN v3.0 — FER-2013" if engine_mode == "cnn" else "DEMO MODE"
engine_color = "#00ff94"              if engine_mode == "cnn" else "#ff9f0a"

st.markdown(f"""
<div style="padding:18px 0 14px;border-bottom:1px solid #1a2540;margin-bottom:22px;">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <div style="font-family:'Oxanium',monospace;font-size:clamp(22px,5vw,34px);
                font-weight:800;letter-spacing:.1em;color:#dde6f5;line-height:1.1;">
      DEEP<span style="color:#00e5ff;">SCAN</span><span style="color:#ff2d55;">.</span>AI
      <span style="display:block;font-size:.32em;font-weight:400;color:#5a7399;
                   letter-spacing:.22em;margin-top:2px;">
        EMOTIONAL AUTHENTICITY DETECTOR v3.0
      </span>
    </div>
    <div style="margin-left:auto;font-family:'IBM Plex Mono',monospace;
                font-size:11px;color:#5a7399;text-align:right;line-height:1.8;">
      <span style="color:#00ff94;display:block;">● SYSTEM ONLINE</span>
      <span style="color:{engine_color};">{engine_label}</span><br>
      OpenCV Haarcascade
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if engine_mode == "demo":
    st.warning(
        "⚡ **Demo Mode** — `emotion_model.h5` not found. "
        "Open `train_colab.ipynb` in Google Colab, train the CNN, "
        "download `emotion_model.h5`, upload to your GitHub repo root, then redeploy."
    )
else:
    st.success("✅ **CNN Model Loaded** — FER-2013 trained model active. OpenCV Haarcascade ready.")

# ═══════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════
tab_analyze, tab_live, tab_game, tab_how, tab_about = st.tabs([
    "🔬  ANALYZE VIDEO",
    "📡  LIVE CAMERA",
    "🎮  GAME MODE",
    "📊  HOW IT WORKS",
    "ℹ️   ABOUT",
])

# ═══════════════════════════════════════════════════════════
#  TAB 1 — ANALYZE VIDEO
# ═══════════════════════════════════════════════════════════
with tab_analyze:
    left_col, right_col = st.columns([1.3, 1], gap="large")

    with left_col:
        st.markdown('<div class="ds-card"><span class="ds-label">Upload Video File</span>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "video", type=["mp4", "avi", "mov", "mkv", "webm"],
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="ds-card"><span class="ds-label">Analysis Settings</span>',
                    unsafe_allow_html=True)
        sample_rate = st.slider("Sample every Nth frame", 1, 15, 5)
        max_frames  = st.select_slider(
            "Max frames to analyse",
            options=[30, 50, 75, 100, 150, 200], value=100,
        )
        demo_toggle = st.toggle(
            "Demo Mode (no video / no model needed)",
            value=(engine_mode == "demo"),
        )
        st.markdown("</div>", unsafe_allow_html=True)
        run_btn = st.button("🔬  RUN DEEP ANALYSIS", use_container_width=True)

    with right_col:
        st.markdown("""
        <div class="ds-card">
          <span class="ds-label">CNN Architecture</span>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8aa0bd;line-height:2.2;">
            <span style="color:#00e5ff;">INPUT</span>    48×48 grayscale<br>
            <span style="color:#bf5af2;">BLOCK 1</span>  Conv2D(32)×2 → BN → ReLU → MaxPool<br>
            <span style="color:#bf5af2;">BLOCK 2</span>  Conv2D(64)×2 → BN → ReLU → MaxPool<br>
            <span style="color:#bf5af2;">BLOCK 3</span>  Conv2D(128)×2→ BN → ReLU → MaxPool<br>
            <span style="color:#ffd60a;">FC</span>       Dense(256) → Dense(128) → Dropout<br>
            <span style="color:#00ff94;">OUTPUT</span>   Dense(7) → Softmax
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="ds-card">
          <span class="ds-label">Scoring Factors</span>
          <div style="font-size:13px;color:#8aa0bd;line-height:2.1;font-family:'IBM Plex Sans';">
            <span style="color:#00e5ff;font-family:'IBM Plex Mono';">01</span>  Emotion variance<br>
            <span style="color:#00e5ff;font-family:'IBM Plex Mono';">02</span>  Transition smoothness<br>
            <span style="color:#00e5ff;font-family:'IBM Plex Mono';">03</span>  Micro-expression count<br>
            <span style="color:#00e5ff;font-family:'IBM Plex Mono';">04</span>  Dominant emotion lock<br>
            <span style="color:#00e5ff;font-family:'IBM Plex Mono';">05</span>  Complexity blend
          </div>
        </div>""", unsafe_allow_html=True)

    if run_btn:
        if not uploaded and not demo_toggle:
            st.warning("⚠️ Upload a video or enable Demo Mode.")
            st.stop()

        st.markdown("---")
        prog   = st.progress(0)
        status = st.empty()
        history, frame_indices = [], []

        if demo_toggle or not uploaded:
            is_fake = bool(np.random.default_rng().integers(0, 2))
            status.markdown(
                f'<p style="font-family:IBM Plex Mono;color:#00e5ff;font-size:13px;">'
                f'⚡ Generating synthetic {"DEEPFAKE" if is_fake else "REAL"} pattern…</p>',
                unsafe_allow_html=True,
            )
            history, frame_indices = make_demo_history(80, is_fake=is_fake)
            for i in range(80):
                prog.progress((i + 1) / 80)
        else:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                    f.write(uploaded.read())
                    tmp_path = f.name

                status.markdown(
                    '<p style="font-family:IBM Plex Mono;color:#00e5ff;font-size:13px;">'
                    '📹 Extracting frames with OpenCV…</p>',
                    unsafe_allow_html=True,
                )
                frames, frame_indices = extract_frames(
                    tmp_path, sample_rate=sample_rate, max_frames=max_frames
                )
                if not frames:
                    st.error("❌ No frames extracted. Try a different video.")
                    st.stop()

                no_face = 0
                for i, frm in enumerate(frames):
                    prog.progress((i + 1) / len(frames))
                    status.markdown(
                        f'<p style="font-family:IBM Plex Mono;color:#00e5ff;font-size:12px;">'
                        f'🔬 CNN inference — frame {i+1}/{len(frames)}</p>',
                        unsafe_allow_html=True,
                    )
                    emo = predict_emotion_cnn(frm)
                    if emo:
                        history.append(emo)
                    else:
                        no_face += 1

                if no_face > len(frames) * 0.65:
                    st.warning(
                        f"⚠️ No face detected in {no_face}/{len(frames)} frames. "
                        "Ensure the video shows a clear front-facing face."
                    )
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        prog.progress(1.0)
        status.empty()

        if len(history) < 5:
            st.error("❌ Insufficient data. Use a longer video with a visible face.")
            st.stop()

        real_score, metrics = analyse_patterns(history)
        if real_score is None:
            st.error("❌ Analysis failed. Please retry.")
            st.stop()

        fake_score = round(100 - real_score, 1)
        is_real    = real_score >= 50
        rl         = risk_level(real_score)

        st.markdown("<br>", unsafe_allow_html=True)

        if is_real:
            st.markdown(f"""
            <div class="verdict-real">
              <div style="margin-bottom:10px;">{risk_badge_html(rl)}</div>
              <div class="verdict-score" style="color:#00ff94;">{real_score:.1f}%</div>
              <div class="verdict-tag" style="color:#00ff94;">✓ &nbsp;AUTHENTIC</div>
              <div class="verdict-sub">Natural irregular emotion patterns —
              consistent with genuine human facial behaviour</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-fake">
              <div style="margin-bottom:10px;">{risk_badge_html(rl)}</div>
              <div class="verdict-score" style="color:#ff2d55;">{fake_score:.1f}%</div>
              <div class="verdict-tag" style="color:#ff2d55;">⚠ &nbsp;DEEPFAKE DETECTED</div>
              <div class="verdict-sub">Unnatural smoothness &amp; locked emotion patterns —
              inconsistent with real human facial behaviour</div>
            </div>""", unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, lbl, val, color in [
            (mc1, "Frames",        str(len(history)),               "#00e5ff"),
            (mc2, "Micro-Expr",    str(metrics["micro_expr"]),      "#ffd60a"),
            (mc3, "Variance",      f"{metrics['avg_variance']:.1f}","#bf5af2"),
            (mc4, "Dom Lock",      f"{metrics['dom_ratio_pct']}%",  "#ff9f0a"),
        ]:
            with col:
                st.markdown(f"""
                <div class="ds-metric">
                  <div class="ds-metric-label">{lbl}</div>
                  <div class="ds-metric-value" style="color:{color};">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        gc, rc = st.columns([1.6, 1], gap="large")
        with gc:
            st.plotly_chart(timeline_chart(history, frame_indices),
                            use_container_width=True, config={"displayModeBar": False})
        with rc:
            st.plotly_chart(radar_chart(metrics, is_real),
                            use_container_width=True, config={"displayModeBar": False})

        with st.expander("🧠 Explainable AI — Why this verdict?", expanded=True):
            rc1, rc2 = st.columns([1, 1], gap="medium")
            with rc1:
                vc = "#00ff94" if is_real else "#ff2d55"
                st.markdown(f"""
                <div style="background:rgba({"0,255,148" if is_real else "255,45,85"},.06);
                     border:1px solid {vc}33;border-radius:10px;padding:14px;margin-bottom:14px;">
                  <div style="font-family:'Oxanium',monospace;font-weight:700;color:{vc};
                       font-size:13px;margin-bottom:8px;">
                    {"✓ AUTHENTIC — Key Reasons" if is_real else "⚠ DEEPFAKE — Key Reasons"}
                  </div>
                  <div style="font-size:12px;color:#8aa0bd;line-height:1.9;">
                    {"Jagged, unpredictable emotion variance detected frame-to-frame. Multiple emotions active simultaneously. Micro-expression spikes present — hallmarks of genuine facial behaviour." if is_real else "Mathematically smooth emotion curves detected. Single dominant emotion lock-in. Near-zero micro-expression activity — consistent with AI-generated face synthesis."}
                  </div>
                </div>""", unsafe_allow_html=True)

                for lbl, val, color, wt, desc in [
                    ("Emotion Variance",   metrics["v_score"], "#00e5ff", "30%",
                     "High natural variance → real" if is_real else "Too low → AI"),
                    ("Transition Smooth.", metrics["s_score"], "#bf5af2", "25%",
                     "Irregular → real"             if is_real else "Too smooth → fake"),
                    ("Micro-Expressions", metrics["m_score"], "#ffd60a", "20%",
                     f"{metrics['micro_expr']} spikes → authentic" if is_real else "Zero spikes → AI"),
                    ("Emotion Diversity",  metrics["d_score"], "#00ff94", "15%",
                     "Varied mix → real"            if is_real else "Locked → fake"),
                    ("Complexity Blend",   metrics["c_score"], "#ff9f0a", "10%",
                     "Blended → real"               if is_real else "No blend → fake"),
                ]:
                    st.markdown(f"""
                    <div style="margin-bottom:12px;">
                      <div style="display:flex;justify-content:space-between;
                                  align-items:center;margin-bottom:4px;">
                        <span style="font-family:'IBM Plex Mono';font-size:10px;
                                     color:#dde6f5;">{lbl}</span>
                        <div style="display:flex;gap:8px;align-items:center;">
                          <span style="font-family:'IBM Plex Mono';font-size:9px;
                                       color:#5a7399;">w:{wt}</span>
                          <span style="font-family:'Oxanium';font-weight:700;
                                       font-size:13px;color:{color};">{val:.1f}%</span>
                        </div>
                      </div>
                      <div class="xai-bar-track">
                        <div class="xai-bar-fill" style="width:{val}%;
                             background:linear-gradient(90deg,{color}88,{color});"></div>
                      </div>
                      <div style="font-size:10px;color:#5a7399;">{desc}</div>
                    </div>""", unsafe_allow_html=True)

            with rc2:
                st.plotly_chart(xai_chart(metrics, is_real),
                                use_container_width=True, config={"displayModeBar": False})

        with st.expander("📋 Full Score Breakdown"):
            st.dataframe(
                pd.DataFrame({
                    "Factor":  ["Emotion Variance", "Transition Smoothness",
                                "Micro-Expressions", "Emotion Diversity", "Complexity Blend"],
                    "Score":   [f"{metrics['v_score']:.1f}%", f"{metrics['s_score']:.1f}%",
                                f"{metrics['m_score']:.1f}%", f"{metrics['d_score']:.1f}%",
                                f"{metrics['c_score']:.1f}%"],
                    "Weight":  ["30%", "25%", "20%", "15%", "10%"],
                    "High =":  ["More real"] * 5,
                }),
                use_container_width=True, hide_index=True,
            )

# ═══════════════════════════════════════════════════════════
#  TAB 2 — LIVE CAMERA
# ═══════════════════════════════════════════════════════════
with tab_live:
    st.markdown('<div class="ds-card"><span class="ds-label">Live Camera — CNN Inference</span>',
                unsafe_allow_html=True)

    cam_col, bar_col = st.columns([1, 1], gap="large")

    with cam_col:
        use_webcam = st.toggle("Enable Webcam", value=False)
        live_emo   = None

        if use_webcam:
            img = st.camera_input("camera", label_visibility="collapsed")
            if img:
                file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
                frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame is not None:
                    with st.spinner("Running CNN inference…"):
                        live_emo = predict_emotion_cnn(frame)
                    if not live_emo:
                        st.warning(
                            "⚠️ No face detected. Move closer, improve lighting, "
                            "or face the camera directly."
                        )
                    else:
                        dom       = max(live_emo, key=live_emo.get)
                        dom_color = EMO_COLORS.get(dom, "#00e5ff")
                        st.markdown(f"""
                        <div style="text-align:center;padding:18px 0;">
                          <div style="font-family:'Oxanium',monospace;font-weight:800;
                               font-size:42px;color:{dom_color};">{dom.upper()}</div>
                          <div style="font-family:'IBM Plex Mono';font-size:13px;
                               color:#5a7399;margin-top:4px;">
                            DOMINANT — {live_emo[dom]:.1f}%
                          </div>
                        </div>""", unsafe_allow_html=True)
        else:
            st.info("💡 Toggle **Enable Webcam** above for live CNN inference, or use Simulation below.")

        st.markdown("---")
        st.markdown('<span class="ds-label">Simulation Mode</span>', unsafe_allow_html=True)

        for k, v in [("sim_running", False), ("sim_history", []), ("sim_frame", 0)]:
            if k not in st.session_state:
                st.session_state[k] = v

        s1, s2 = st.columns(2)
        with s1:
            if st.button("▶ START SIMULATION", use_container_width=True):
                st.session_state.sim_running = True
                st.session_state.sim_history = []
                st.session_state.sim_frame   = 0
        with s2:
            if st.button("■ STOP", use_container_width=True):
                st.session_state.sim_running = False

        if st.session_state.sim_running and st.session_state.sim_frame < 60:
            rng = np.random.default_rng()
            for _ in range(4):
                vals = rng.dirichlet(np.ones(7) * 2) * 100
                st.session_state.sim_history.append(
                    {k: round(float(v), 2) for k, v in zip(EMOTIONS, vals)}
                )
                st.session_state.sim_frame += 1
            time.sleep(0.05)
            st.rerun()

    with bar_col:
        display_emo = live_emo or (
            st.session_state.sim_history[-1]
            if st.session_state.sim_history else None
        )
        if display_emo:
            st.markdown('<span class="ds-label">Emotion Distribution</span>',
                        unsafe_allow_html=True)
            for k in EMOTIONS:
                v = display_emo.get(k, 0)
                c = EMO_COLORS.get(k, "#00e5ff")
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                    <span style="font-family:'IBM Plex Mono';font-size:11px;color:{c};">
                      {k.capitalize()}</span>
                    <span style="font-family:'IBM Plex Mono';font-size:11px;color:#5a7399;">
                      {v:.1f}%</span>
                  </div>
                  <div style="background:#1a2540;border-radius:3px;height:6px;overflow:hidden;">
                    <div style="width:{v}%;height:100%;background:{c};border-radius:3px;
                                transition:width .2s;"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            if len(st.session_state.sim_history) > 3:
                st.plotly_chart(
                    live_stream_chart(st.session_state.sim_history),
                    use_container_width=True, config={"displayModeBar": False},
                )

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  TAB 3 — GAME MODE
# ═══════════════════════════════════════════════════════════
with tab_game:
    GAME_DATA = [
        {"label": "News Anchor — 6PM Broadcast",
         "hint":  "Steady delivery, minimal expression variance.",
         "is_real": False,
         "scores": {"v": 14, "s": 12, "m": 7, "d": 18, "c": 11}},
        {"label": "Child reacting to surprise gift",
         "hint":  "Rapid spikes — joy, surprise, back to neutral instantly.",
         "is_real": True,
         "scores": {"v": 88, "s": 72, "m": 91, "d": 84, "c": 79}},
        {"label": "CEO quarterly results interview",
         "hint":  "Very controlled, consistent expression throughout.",
         "is_real": False,
         "scores": {"v": 16, "s": 11, "m": 8, "d": 21, "c": 13}},
        {"label": "Street argument caught on camera",
         "hint":  "Anger spikes with fear flickers — messy mix.",
         "is_real": True,
         "scores": {"v": 83, "s": 68, "m": 76, "d": 81, "c": 74}},
        {"label": "AI-generated political speech",
         "hint":  "Perfect emotion timing, zero micro-expression activity.",
         "is_real": False,
         "scores": {"v": 12, "s": 9, "m": 5, "d": 16, "c": 10}},
    ]

    for key, default in [
        ("game_round", 0), ("game_score", 0),
        ("game_guess", None), ("game_history", []), ("game_done", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if st.session_state.game_done:
        total = len(GAME_DATA)
        pct   = int(st.session_state.game_score / total * 100)
        _, gc2, _ = st.columns([1, 1.2, 1])
        with gc2:
            st.markdown(f"""
            <div class="ds-card" style="text-align:center;padding:36px 24px;">
              <div style="font-family:'IBM Plex Mono';font-size:11px;color:#5a7399;
                   letter-spacing:.2em;margin-bottom:14px;">GAME COMPLETE</div>
              <div style="font-family:'Oxanium',monospace;font-weight:800;font-size:72px;
                   color:{"#00ff94" if pct>=60 else "#ff2d55"};line-height:1;">
                {st.session_state.game_score}
                <span style="font-size:28px;color:#5a7399;">/{total}</span>
              </div>
              <div style="font-family:'IBM Plex Sans';font-size:14px;color:#5a7399;margin:12px 0 20px;">
                {"🏆 Expert! You understand the CNN analysis." if pct>=80
                 else "🎯 Good instincts! Study the radar patterns." if pct>=60
                 else "📚 Deepfakes are convincing. Review the scoring factors."}
              </div>
            </div>""", unsafe_allow_html=True)
            if st.button("🔄 PLAY AGAIN", use_container_width=True):
                for k, v in [("game_round", 0), ("game_score", 0),
                              ("game_guess", None), ("game_history", []), ("game_done", False)]:
                    st.session_state[k] = v
                st.rerun()

        st.markdown('<span class="ds-label" style="margin-top:18px;display:block;">History</span>',
                    unsafe_allow_html=True)
        for h in st.session_state.game_history:
            c  = "#00ff94" if h["correct"] else "#ff2d55"
            bg = "rgba(0,255,148,.07)" if h["correct"] else "rgba(255,45,85,.07)"
            st.markdown(f"""
            <div style="display:flex;gap:12px;align-items:center;padding:10px 14px;
                 margin-bottom:6px;border-radius:10px;background:{bg};border:1px solid {c}33;">
              <span style="color:{c};font-size:16px;">{"✓" if h["correct"] else "✗"}</span>
              <span style="font-family:'IBM Plex Sans';font-size:13px;
                   color:#dde6f5;flex:1;">{h["label"]}</span>
              <span style="font-family:'IBM Plex Mono';font-size:11px;
                   color:{"#00ff94" if h["is_real"] else "#ff2d55"};">
                {"REAL" if h["is_real"] else "FAKE"}
              </span>
            </div>""", unsafe_allow_html=True)
    else:
        cur   = GAME_DATA[st.session_state.game_round]
        total = len(GAME_DATA)

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
          <div style="flex:1;background:#1a2540;border-radius:4px;height:5px;overflow:hidden;">
            <div style="width:{st.session_state.game_round/total*100:.0f}%;height:100%;
                 background:linear-gradient(90deg,#00e5ff,#0055ff);border-radius:4px;"></div>
          </div>
          <span style="font-family:'IBM Plex Mono';font-size:11px;color:#5a7399;">
            {st.session_state.game_round+1}/{total}
          </span>
        </div>""", unsafe_allow_html=True)

        g_left, g_right = st.columns([1.3, 1], gap="large")

        with g_left:
            st.markdown(f"""
            <div class="ds-card">
              <span class="ds-label">Round {st.session_state.game_round+1} — Read The Pattern</span>
              <div style="font-family:'Oxanium',monospace;font-weight:700;font-size:17px;
                   color:#dde6f5;line-height:1.4;margin-bottom:10px;">{cur["label"]}</div>
              <div style="font-size:13px;color:#5a7399;line-height:1.7;">
                💡 {cur["hint"]}
              </div>
            </div>""", unsafe_allow_html=True)

            st.plotly_chart(
                game_radar_chart(cur["scores"]),
                use_container_width=True, config={"displayModeBar": False},
            )

            if st.session_state.game_guess is None:
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("✓  REAL", use_container_width=True):
                        correct = cur["is_real"]
                        if correct: st.session_state.game_score += 1
                        st.session_state.game_guess = "real"
                        st.session_state.game_history.append(
                            {"label": cur["label"], "guess": "real",
                             "correct": correct, "is_real": cur["is_real"]})
                        st.rerun()
                with b2:
                    st.markdown("""<style>
                      div[data-testid="column"]:nth-of-type(2) .stButton>button{
                        background:linear-gradient(135deg,#ff2d55,#aa1133)!important;
                        color:#fff!important;}
                    </style>""", unsafe_allow_html=True)
                    if st.button("⚠  FAKE", use_container_width=True):
                        correct = not cur["is_real"]
                        if correct: st.session_state.game_score += 1
                        st.session_state.game_guess = "fake"
                        st.session_state.game_history.append(
                            {"label": cur["label"], "guess": "fake",
                             "correct": correct, "is_real": cur["is_real"]})
                        st.rerun()
            else:
                correct = st.session_state.game_history[-1]["correct"]
                vc      = "#00ff94" if correct else "#ff2d55"
                bg_v    = "rgba(0,255,148,.08)" if correct else "rgba(255,45,85,.08)"
                is_last = st.session_state.game_round == total - 1
                st.markdown(f"""
                <div style="background:{bg_v};border:1px solid {vc}44;border-radius:12px;
                     padding:16px;text-align:center;margin-bottom:12px;">
                  <div style="font-family:'Oxanium',monospace;font-weight:700;
                       font-size:18px;color:{vc};">
                    {"✓ CORRECT!" if correct else "✗ WRONG"}
                  </div>
                  <div style="font-size:12px;color:#5a7399;margin-top:6px;">
                    This was a <strong style="color:{"#00ff94" if cur["is_real"] else "#ff2d55"};">
                    {"REAL" if cur["is_real"] else "DEEPFAKE"}</strong> video.
                  </div>
                </div>""", unsafe_allow_html=True)
                if st.button(
                    "SEE RESULTS →" if is_last else "NEXT ROUND →",
                    use_container_width=True,
                ):
                    if is_last:
                        st.session_state.game_done = True
                    else:
                        st.session_state.game_round += 1
                        st.session_state.game_guess  = None
                    st.rerun()

        with g_right:
            st.markdown(f"""
            <div class="ds-card" style="text-align:center;">
              <span class="ds-label">Score</span>
              <div style="font-family:'Oxanium',monospace;font-weight:800;
                   font-size:64px;color:#00e5ff;line-height:1;">
                {st.session_state.game_score}
              </div>
              <div style="font-family:'IBM Plex Mono';font-size:11px;
                   color:#5a7399;margin:4px 0 16px;">correct</div>
            </div>""", unsafe_allow_html=True)
            for h in st.session_state.game_history:
                c  = "#00ff94" if h["correct"] else "#ff2d55"
                bg = "rgba(0,255,148,.06)" if h["correct"] else "rgba(255,45,85,.06)"
                st.markdown(f"""
                <div style="display:flex;gap:8px;align-items:center;padding:7px 10px;
                     border-radius:7px;margin-bottom:5px;background:{bg};
                     border:1px solid {c}33;">
                  <span style="color:{c};">{"✓" if h["correct"] else "✗"}</span>
                  <span style="font-family:'IBM Plex Mono';font-size:9px;flex:1;
                       color:{"#00ff94" if h["is_real"] else "#ff2d55"};">
                    {"REAL" if h["is_real"] else "FAKE"}
                  </span>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  TAB 4 — HOW IT WORKS
# ═══════════════════════════════════════════════════════════
with tab_how:
    hw_l, hw_r = st.columns(2, gap="large")
    with hw_l:
        st.markdown("""
        <div class="ds-card">
          <span class="ds-label">Core Insight</span>
          <p style="font-size:13px;color:#8aa0bd;line-height:1.9;">
            Real faces show <strong style="color:#00e5ff;">natural emotional messiness</strong> —
            irregular, unpredictable frame-to-frame. Deepfakes produce
            <strong style="color:#ff2d55;">mathematically smooth</strong> emotion curves.
            The CNN reads every frame. DeepScan detects the difference.
          </p>
        </div>
        <div class="ds-card">
          <span class="ds-label">✓ Real Human Face</span>
          <div style="font-size:13px;color:#8aa0bd;line-height:2.1;">
            • Jagged unpredictable transitions<br>
            • Micro-expression spikes<br>
            • Multiple emotions simultaneously<br>
            • High frame-to-frame variance<br>
            • No single emotion locked in
          </div>
        </div>""", unsafe_allow_html=True)

    with hw_r:
        st.markdown("""
        <div class="ds-card">
          <span class="ds-label">⚠ AI Deepfake Face</span>
          <div style="font-size:13px;color:#8aa0bd;line-height:2.1;">
            • Smooth sinusoidal curves<br>
            • Near-zero micro-expressions<br>
            • Single dominant emotion locked<br>
            • Uniformly timed transitions<br>
            • Very low frame-to-frame change
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="ds-card"><span class="ds-label">Full Pipeline</span>',
                unsafe_allow_html=True)
    for color, num, title, desc in [
        ("#00e5ff","01","FRAME EXTRACTION",
         "OpenCV reads video and samples every Nth frame for processing."),
        ("#ffd60a","02","HAARCASCADE DETECTION",
         "cv2.CascadeClassifier detects faces per frame. Largest face selected."),
        ("#bf5af2","03","CNN PREPROCESSING",
         "Face → grayscale → 48×48 resize → /255 normalize → shape (1,48,48,1)."),
        ("#ff9f0a","04","FER-2013 CNN INFERENCE",
         "3-block Conv2D network predicts 7-class emotion probability per frame."),
        ("#00ff94","05","PATTERN ANALYSIS",
         "Variance + smoothness + micro-expr + dominance + complexity → authenticity score."),
    ]:
        st.markdown(f"""
        <div style="display:flex;gap:14px;align-items:flex-start;
             background:rgba(255,255,255,.025);border-radius:10px;
             padding:14px;margin-bottom:8px;">
          <div style="font-family:'Oxanium',monospace;font-size:22px;font-weight:800;
               color:{color};min-width:34px;line-height:1;">{num}</div>
          <div>
            <div style="font-family:'Oxanium',monospace;font-size:11px;font-weight:700;
                 color:{color};letter-spacing:.14em;margin-bottom:3px;">{title}</div>
            <div style="font-size:12px;color:#8aa0bd;">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  TAB 5 — ABOUT
# ═══════════════════════════════════════════════════════════
with tab_about:
    a1, a2 = st.columns([1.2, 1], gap="large")
    with a1:
        st.markdown("""
        <div class="ds-card">
          <span class="ds-label">About DeepScan AI v3.0</span>
          <p style="font-size:13px;color:#8aa0bd;line-height:1.9;">
            Custom trained <strong style="color:#00e5ff;">FER-2013 CNN</strong> classifies
            emotions per frame. Pattern analysis across all frames determines authenticity.
            No FER library. No DeepFace. Pure CNN + OpenCV.
          </p>
          <hr>
        </div>""", unsafe_allow_html=True)
        for v, d in [
            ("FER-2013",  "35,887 grayscale 48×48 face images"),
            ("7 classes", "Angry Disgust Fear Happy Sad Surprise Neutral"),
            ("5 factors", "Multi-dimensional authenticity scoring"),
            ("OpenCV",    "Haarcascade — no deep learning for detection"),
        ]:
            st.markdown(f"""
            <div style="display:flex;gap:14px;align-items:center;padding:10px 14px;
                 margin-bottom:6px;border-radius:8px;background:rgba(255,255,255,.025);">
              <div style="font-family:'Oxanium',monospace;font-weight:800;
                   font-size:15px;color:#00e5ff;min-width:100px;">{v}</div>
              <div style="font-size:12px;color:#5a7399;">{d}</div>
            </div>""", unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="ds-card"><span class="ds-label">Emotion Classes</span>',
                    unsafe_allow_html=True)
        for name, color in [
            ("😠 Angry","#ff2d55"),("🤢 Disgust","#30d158"),("😨 Fear","#bf5af2"),
            ("😄 Happy","#ffd60a"),("😢 Sad","#5b8cff"),
            ("😲 Surprise","#ff9f0a"),("😐 Neutral","#00e5ff"),
        ]:
            st.markdown(
                f'<span style="display:inline-block;margin:4px;background:rgba(255,255,255,.04);'
                f'border:1px solid {color}44;color:{color};font-family:Oxanium,monospace;'
                f'font-size:12px;font-weight:600;padding:6px 14px;border-radius:20px;">'
                f'{name}</span>', unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="ds-card" style="margin-top:4px;">
          <span class="ds-label">Tech Stack</span>
          <div style="font-size:13px;color:#8aa0bd;line-height:2.4;">
            <span style="color:#00e5ff;font-family:'IBM Plex Mono';">Model</span>
              → TensorFlow / Keras CNN<br>
            <span style="color:#00e5ff;font-family:'IBM Plex Mono';">Detection</span>
              → OpenCV Haarcascade<br>
            <span style="color:#00e5ff;font-family:'IBM Plex Mono';">Dataset</span>
              → FER-2013 (Kaggle)<br>
            <span style="color:#00e5ff;font-family:'IBM Plex Mono';">UI</span>
              → Streamlit<br>
            <span style="color:#00e5ff;font-family:'IBM Plex Mono';">Charts</span>
              → Plotly
          </div>
        </div>""", unsafe_allow_html=True)
