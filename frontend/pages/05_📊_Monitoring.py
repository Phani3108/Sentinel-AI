"""
Sentinel AI — Page 5: Monitoring Dashboard
Live Prometheus metrics, latency charts, component health, and system info.
"""
import os
import sys
import streamlit as st
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

st.set_page_config(page_title="Monitoring · Sentinel AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background:#FFFFFF; }
[data-testid="stSidebar"] { background:#F8FAFC; border-right:1px solid #E2E8F0; }
.page-header { font-size:2rem; font-weight:800; color:#1E293B; margin-bottom:4px; }
.page-sub { font-size:1rem; color:#64748B; margin-bottom:24px; }
.status-card {
  background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px;
  padding:16px 20px; text-align:center;
}
.status-green { border-top:4px solid #22C55E; }
.status-red   { border-top:4px solid #EF4444; }
.status-gray  { border-top:4px solid #CBD5E1; }
.status-val   { font-size:1.4rem; font-weight:800; color:#1E293B; margin:4px 0; }
.status-label { font-size:.75rem; color:#64748B; text-transform:uppercase; letter-spacing:.05em; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

from frontend.components.sidebar import render_sidebar

config = render_sidebar(show_model_selector=False, show_rag=False)
api_url = config.get("api_url", "http://localhost:8080")

st.markdown('<div class="page-header">📊 Monitoring Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Live metrics from the Sentinel AI pipeline — latency, throughput, memory, and health.</div>',
    unsafe_allow_html=True,
)

# ── Auto-refresh ───────────────────────────────────────────────────────────────
col_r1, col_r2 = st.columns([3, 1])
with col_r1:
    auto_refresh = st.toggle("🔄 Auto-refresh every 10s", value=False)
with col_r2:
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()

if auto_refresh:
    time.sleep(10)
    st.rerun()

st.divider()

# ── Fetch health ───────────────────────────────────────────────────────────────
import httpx

health_data = {}
metrics_text = ""

try:
    with httpx.Client(timeout=5.0) as client:
        h_resp = client.get(f"{api_url}/health")
        if h_resp.status_code == 200:
            health_data = h_resp.json()
except Exception:
    pass

try:
    with httpx.Client(timeout=5.0) as client:
        m_resp = client.get(f"{api_url}/metrics")
        if m_resp.status_code == 200:
            metrics_text = m_resp.text
except Exception:
    pass

# ── Component Health ───────────────────────────────────────────────────────────
st.markdown("### 🔍 Component Health")

statuses = [
    ("API Server",     bool(health_data), health_data.get("status", "unreachable")),
    ("LLM (Ollama)",   health_data.get("llm_available", False), "ready" if health_data.get("llm_available") else "offline"),
    ("Vision Model",   bool(health_data.get("vision_model")), health_data.get("vision_model", "—")),
    ("RAG (ChromaDB)", bool(health_data.get("rag_docs", 0) >= 0), f"{health_data.get('rag_docs', 0)} docs"),
    ("Metrics",        bool(metrics_text), "active" if metrics_text else "no data"),
]

cols = st.columns(len(statuses))
for col, (label, is_ok, val) in zip(cols, statuses):
    css = "status-green" if is_ok else "status-red"
    icon = "✅" if is_ok else "❌"
    col.markdown(f"""
    <div class="status-card {css}">
      <div style="font-size:1.3rem;">{icon}</div>
      <div class="status-val">{val}</div>
      <div class="status-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Parse and display Prometheus metrics ────────────────────────────────────────
def _parse_metric(text: str, metric_name: str, label_filter: str = "") -> float:
    """Extract a metric value from prometheus text format."""
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if metric_name in line and (not label_filter or label_filter in line):
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                try:
                    return float(parts[1])
                except ValueError:
                    pass
    return 0.0


if metrics_text:
    st.markdown("### 📈 Live Metrics")

    m1, m2, m3, m4 = st.columns(4)
    total_requests = _parse_metric(metrics_text, "sentinel_requests_total")
    total_errors = _parse_metric(metrics_text, "sentinel_errors_total")
    memory_mb = _parse_metric(metrics_text, "sentinel_memory_usage_mb")
    rag_docs = _parse_metric(metrics_text, "sentinel_rag_documents_total")

    m1.metric("Total Requests", f"{int(total_requests):,}")
    m2.metric("Errors", f"{int(total_errors):,}", delta=None)
    m3.metric("Memory", f"{memory_mb:.0f} MB")
    m4.metric("RAG Docs", f"{int(rag_docs):,}")

    st.divider()

    # Raw metrics expander
    st.markdown("### 📋 Raw Prometheus Output")
    with st.expander("View all metrics (text/plain)", expanded=False):
        st.code(metrics_text, language="text")

    # External links
    st.divider()
    st.markdown("### 🔗 Monitoring Stack")
    link_cols = st.columns(3)
    link_cols[0].markdown("""
    **📊 Grafana**
    [Open Dashboard →](http://localhost:3000)
    *admin / sentinel123*
    """)
    link_cols[1].markdown("""
    **🔥 Prometheus**
    [Open Explorer →](http://localhost:9090)
    *PromQL query interface*
    """)
    link_cols[2].markdown("""
    **🔍 Jaeger Traces**
    [Open Jaeger →](http://localhost:16686)
    *Distributed tracing UI*
    """)

else:
    st.info(
        f"⚠️ Could not fetch metrics from `{api_url}/metrics`.\n\n"
        "Make sure the API is running: `uvicorn api.main:app --port 8080`"
    )

st.divider()
st.markdown("### 🔧 System Info")
info_cols = st.columns(3)
info_cols[0].markdown(f"""
| Key | Value |
|---|---|
| Vision Model | `{health_data.get('vision_model','—')}` |
| LLM Model | `{health_data.get('llm_model','—')}` |
""")
info_cols[1].markdown(f"""
| Key | Value |
|---|---|
| Device | `{health_data.get('device','—')}` |
| RAG Docs | `{health_data.get('rag_docs','—')}` |
""")
info_cols[2].markdown(f"""
| Key | Value |
|---|---|
| OTEL Enabled | `{health_data.get('otel_enabled', False)}` |
| Metrics | `{health_data.get('metrics_enabled', False)}` |
""")
