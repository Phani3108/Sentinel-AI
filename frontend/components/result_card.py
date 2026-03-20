"""
Sentinel AI — Result Card Component
Renders a premium result display for pipeline output.
"""
import streamlit as st
from typing import Optional


def render_result_card(
    vision_description: str,
    final_answer: str,
    vision_model: str = "—",
    llm_model: str = "—",
    vision_latency_ms: float = 0.0,
    llm_latency_ms: float = 0.0,
    total_latency_ms: float = 0.0,
    llm_tokens_used: int = 0,
    rag_context: Optional[str] = None,
    rag_chunks: int = 0,
):
    """Render a complete pipeline result card."""

    st.markdown("""
    <style>
      .result-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 28px;
        margin-top: 16px;
      }
      .result-section-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: #94A3B8;
        margin-bottom: 8px;
      }
      .result-text {
        font-size: 0.95rem;
        color: #1E293B;
        line-height: 1.7;
      }
      .answer-box {
        background: #F0F7FF;
        border-left: 4px solid #1A56DB;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 8px 0;
      }
      .answer-text {
        font-size: 1.02rem;
        color: #1E293B;
        line-height: 1.75;
        font-weight: 400;
      }
      .badge-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px; }
      .badge {
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid;
      }
      .badge-blue { background: #EFF6FF; color: #1A56DB; border-color: #BFDBFE; }
      .badge-green { background: #F0FDF4; color: #15803D; border-color: #BBF7D0; }
      .badge-purple { background: #FAF5FF; color: #7E22CE; border-color: #E9D5FF; }
      .badge-amber { background: #FFFBEB; color: #B45309; border-color: #FDE68A; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    # Latency badges
    st.markdown("**📊 Performance**")
    st.markdown(f"""
    <div class="badge-row">
      <span class="badge badge-blue">👁️ Vision {vision_latency_ms:.0f}ms</span>
      <span class="badge badge-green">🦙 LLM {llm_latency_ms:.0f}ms</span>
      <span class="badge badge-purple">⚡ Total {total_latency_ms:.0f}ms</span>
      <span class="badge badge-amber">🔤 {llm_tokens_used} tokens</span>
      {"<span class='badge badge-blue'>📚 " + str(rag_chunks) + " RAG chunks</span>" if rag_chunks > 0 else ""}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model info
    m_col1, m_col2 = st.columns(2)
    m_col1.caption(f"🎨 **Vision:** {vision_model}")
    m_col2.caption(f"🧠 **LLM:** {llm_model}")

    st.markdown("---")

    # Vision description
    st.markdown('<div class="result-section-title">👁️ Vision Model Output</div>', unsafe_allow_html=True)
    with st.expander("See full vision description", expanded=False):
        st.markdown(f'<div class="result-text">{vision_description}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Final answer
    st.markdown('<div class="result-section-title">🧠 AI Analysis</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="answer-box">
      <div class="answer-text">{final_answer}</div>
    </div>
    """, unsafe_allow_html=True)

    # RAG context
    if rag_context:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="result-section-title">📚 Retrieved Context</div>', unsafe_allow_html=True)
        with st.expander(f"View {rag_chunks} retrieved chunks", expanded=False):
            st.markdown(f'<div class="result-text">{rag_context}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_error_card(error_message: str):
    """Render an error state card."""
    st.markdown(f"""
    <div style="background:#FFF1F2;border:1px solid #FECDD3;border-radius:12px;padding:20px;">
      <div style="font-size:1.5rem;margin-bottom:8px;">❌</div>
      <div style="font-weight:700;color:#BE123C;margin-bottom:4px;">Analysis Failed</div>
      <div style="color:#9F1239;font-size:0.9rem;">{error_message}</div>
    </div>
    """, unsafe_allow_html=True)


def render_loading_card(stage: str = "Analyzing..."):
    """Render a loading state."""
    st.markdown(f"""
    <div style="background:#F0F7FF;border:1px solid #BFDBFE;border-radius:12px;padding:24px;text-align:center;">
      <div style="font-size:2rem;margin-bottom:8px;">⏳</div>
      <div style="font-weight:600;color:#1A56DB;">{stage}</div>
      <div style="color:#64748B;font-size:0.85rem;margin-top:4px;">Local inference — no data leaves this machine</div>
    </div>
    """, unsafe_allow_html=True)
