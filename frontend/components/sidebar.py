"""
Sentinel AI — Shared Sidebar Component
Renders the configuration sidebar used across all pages.
"""
import streamlit as st
from typing import Optional


# Default vision models to show in the selector
_VISION_MODELS = ["llava:7b", "llava:13b", "florence2", "moondream", "phi3v", "internvl"]
_LLM_MODELS = ["llama3.1:8b", "mistral:7b", "phi3:medium", "qwen2.5:7b"]
_DEVICES = ["cpu", "mps", "cuda"]


def render_sidebar(show_model_selector: bool = True, show_rag: bool = True) -> dict:
    """
    Render the shared sidebar and return a config dict.

    Returns:
        dict with keys: api_url, vision_model, llm_model, device, enable_rag, top_k
    """
    with st.sidebar:
        # Logo + Title
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
          <span style="font-size:1.6rem;">🧠</span>
          <div>
            <div style="font-size:1.1rem;font-weight:800;color:#1E293B;">Sentinel AI</div>
            <div style="font-size:0.72rem;color:#94A3B8;letter-spacing:.05em;">v0.3.0 · On-Premise</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # API Connection
        st.markdown("**🔗 API Connection**")
        api_url = st.text_input(
            "API Base URL",
            value=st.session_state.get("api_url", "http://localhost:8080"),
            placeholder="http://localhost:8080",
            label_visibility="collapsed",
        )
        st.session_state["api_url"] = api_url

        # Quick health check
        if st.button("Test Connection", use_container_width=True):
            try:
                import httpx
                r = httpx.get(f"{api_url}/health", timeout=3.0)
                if r.status_code == 200:
                    st.success("✅ Connected")
                else:
                    st.error(f"❌ HTTP {r.status_code}")
            except Exception as e:
                st.error(f"❌ {str(e)[:60]}")

        st.divider()

        config = {"api_url": api_url}

        if show_model_selector:
            st.markdown("**🎛️ Models**")
            vision_model = st.selectbox(
                "Vision Model",
                options=_VISION_MODELS,
                index=0,
                key="sidebar_vision_model",
            )
            llm_model = st.selectbox(
                "LLM Model",
                options=_LLM_MODELS,
                index=0,
                key="sidebar_llm_model",
            )
            device = st.selectbox(
                "Device",
                options=_DEVICES,
                index=0,
                key="sidebar_device",
            )
            config.update({
                "vision_model": vision_model,
                "llm_model": llm_model,
                "device": device,
            })
            st.divider()

        if show_rag:
            st.markdown("**📚 RAG Settings**")
            enable_rag = st.toggle("Enable RAG", value=False, key="sidebar_rag")
            if enable_rag:
                top_k = st.slider("Top-K Chunks", 1, 10, 3, key="sidebar_topk")
                config["top_k"] = top_k
            config["enable_rag"] = enable_rag
            st.divider()

        # System info
        st.markdown("**ℹ️ Info**")
        _info_rows = [
            ("Phase", "4 / 6"),
            ("Backend", "FastAPI"),
            ("RAG", "ChromaDB"),
            ("Monitoring", "Prometheus"),
        ]
        for label, val in _info_rows:
            cols = st.columns([1, 1])
            cols[0].caption(label)
            cols[1].caption(f"**{val}**")

        st.divider()
        st.caption("🔒 All inference runs locally — your data never leaves this machine.")

    return config
