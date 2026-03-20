"""
Sentinel AI — Streamlit Frontend
Main landing page with navigation cards and sidebar configuration.
"""
import streamlit as st

st.set_page_config(
    page_title="Sentinel AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global Styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base */
  html, body, [data-testid="stAppViewContainer"] {
    background-color: #FFFFFF;
    font-family: 'Inter', 'Segoe UI', sans-serif;
  }
  [data-testid="stSidebar"] {
    background-color: #F8FAFC;
    border-right: 1px solid #E2E8F0;
  }

  /* Hero */
  .hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: #1E293B;
    letter-spacing: -0.03em;
    line-height: 1.1;
  }
  .hero-sub {
    font-size: 1.2rem;
    color: #64748B;
    margin-top: 0.5rem;
    font-weight: 400;
  }
  .hero-badge {
    display: inline-block;
    background: #EFF6FF;
    color: #1A56DB;
    border: 1px solid #BFDBFE;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 6px;
  }

  /* Feature Cards */
  .feat-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 28px 24px;
    transition: box-shadow .2s, transform .2s;
    height: 200px;
    cursor: pointer;
  }
  .feat-card:hover {
    box-shadow: 0 8px 30px rgba(26,86,219,.12);
    transform: translateY(-3px);
  }
  .feat-icon { font-size: 2rem; margin-bottom: 12px; }
  .feat-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 6px;
  }
  .feat-desc { font-size: 0.85rem; color: #64748B; line-height: 1.5; }

  /* Status Pills */
  .pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 8px;
  }
  .pill-green { background: #DCFCE7; color: #166534; }
  .pill-blue  { background: #DBEAFE; color: #1E40AF; }
  .pill-gray  { background: #F1F5F9; color: #475569; }

  /* Section Headers */
  .section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1E293B;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 8px;
    margin-bottom: 20px;
  }

  /* Metric Cards */
  .metric-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
  }
  .metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #1A56DB;
  }
  .metric-label {
    font-size: 0.8rem;
    color: #64748B;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* Divider */
  hr { border-color: #E2E8F0; margin: 2rem 0; }

  /* Hide Streamlit branding */
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
from frontend.components.sidebar import render_sidebar
config = render_sidebar()


# ── Hero Section ───────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("### 🧠", unsafe_allow_html=False)
with col_title:
    st.markdown('<p class="hero-title">Sentinel AI</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="hero-sub">Private · On-Premise · Multimodal AI Pipeline</p>',
    unsafe_allow_html=True,
)
st.markdown("""
<span class="hero-badge">🔒 100% Local</span>
<span class="hero-badge">⚡ Real-Time</span>
<span class="hero-badge">🧠 RAG-Powered</span>
<span class="hero-badge">📊 Observable</span>
<span class="hero-badge">🎛️ Multi-Model</span>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature Cards ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🚀 Capabilities</div>', unsafe_allow_html=True)

cards = [
    ("🖼️", "Image Analysis",
     "Upload any image. Vision models describe it in detail, LLM answers your questions — all locally.",
     "pages/01_🖼️_Image_Analysis"),
    ("🎬", "Video Analysis",
     "Frame-by-frame video understanding. Extract keyframes, get per-frame descriptions, then a complete narration.",
     "pages/02_🎬_Video_Analysis"),
    ("📚", "RAG Explorer",
     "Ingest your internal documents. Query the knowledge base and see exactly which chunks were retrieved.",
     "pages/03_📚_RAG_Explorer"),
    ("📡", "Live Stream",
     "Real-time WebSocket streaming. Upload an image, watch tokens arrive token-by-token.",
     "pages/04_📡_Live_Stream"),
    ("📊", "Monitoring",
     "Live Prometheus metrics. Latency trends, throughput, error rates, and memory usage — all in one view.",
     "pages/05_📊_Monitoring"),
]

cols = st.columns(3)
for i, (icon, title, desc, _page) in enumerate(cards[:3]):
    with cols[i]:
        st.markdown(f"""
        <div class="feat-card">
          <div class="feat-icon">{icon}</div>
          <div class="feat-title">{title}</div>
          <div class="feat-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
cols2 = st.columns([1, 3, 3, 1])
for i, (icon, title, desc, _page) in enumerate(cards[3:]):
    with cols2[i + 1]:
        st.markdown(f"""
        <div class="feat-card">
          <div class="feat-icon">{icon}</div>
          <div class="feat-title">{title}</div>
          <div class="feat-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Architecture Section ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">🏗️ Architecture</div>', unsafe_allow_html=True)

arch_col1, arch_col2 = st.columns([3, 2])
with arch_col1:
    st.code("""
INPUT  Image / Video / Document
  │
  ▼
VISION MODEL (local)
  LLaVA · Florence-2 · InternVL · Phi-3V
  │
  ├── RAG RETRIEVAL (ChromaDB)
  │     Internal docs → ranked context
  │
  ▼
LLM REASONING (Ollama)
  Llama 3.1 · Mistral · Phi-3
  │
  ▼
OUTPUT  Structured response + audit trail
    """, language="text")

with arch_col2:
    api_url = config.get("api_url", "http://localhost:8080")
    try:
        import httpx
        r = httpx.get(f"{api_url}/health", timeout=2.0)
        if r.status_code == 200:
            data = r.json()
            st.success("✅ API Connected")
            m1, m2 = st.columns(2)
            m1.metric("Vision Model", data.get("vision_model", "—"))
            m2.metric("LLM Model", data.get("llm_model", "—"))
            m3, m4 = st.columns(2)
            m3.metric("Device", data.get("device", "cpu"))
            m4.metric("RAG Ready", "Yes" if data.get("llm_available") else "No")
        else:
            st.warning("⚠️ API returned non-200")
    except Exception:
        st.warning("⚠️ API not reachable — start with `uvicorn api.main:app --port 8080`")
        st.caption("Configure the API URL in the sidebar →")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Quick Stats ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Pipeline Specs</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
stats = [
    ("5", "Vision Models"),
    ("4", "LLM Backends"),
    ("6", "Doc Formats"),
    ("9", "Prometheus Metrics"),
    ("110", "Tests Passing"),
]
for col, (val, label) in zip([c1, c2, c3, c4, c5], stats):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-value">{val}</div>
      <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Sentinel AI v0.3.0 · Apache 2.0 · github.com/Phani3108/Sentinel-AI")
