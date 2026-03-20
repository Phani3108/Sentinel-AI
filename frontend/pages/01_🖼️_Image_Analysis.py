"""
Sentinel AI — Page 1: Image Analysis
Upload an image, run the full vision+LLM pipeline, see results in real-time.
"""
import io
import sys
import os
import streamlit as st

# Allow importing from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

st.set_page_config(page_title="Image Analysis · Sentinel AI", page_icon="🖼️", layout="wide")

# Shared styles
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background:#FFFFFF; }
[data-testid="stSidebar"] { background:#F8FAFC; border-right:1px solid #E2E8F0; }
.page-header { font-size:2rem; font-weight:800; color:#1E293B; margin-bottom:4px; }
.page-sub { font-size:1rem; color:#64748B; margin-bottom:24px; }
.upload-zone {
  border: 2px dashed #CBD5E1;
  border-radius: 16px;
  padding: 40px;
  text-align: center;
  background: #F8FAFC;
  transition: border-color .2s;
}
.step-badge {
  display:inline-block; background:#1A56DB; color:#fff;
  border-radius:999px; width:24px; height:24px;
  text-align:center; font-size:.75rem; font-weight:700;
  line-height:24px; margin-right:8px;
}
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

from frontend.components.sidebar import render_sidebar
from frontend.components.result_card import render_result_card, render_error_card, render_loading_card

config = render_sidebar()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-header">🖼️ Image Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Upload any image — vision model describes it, LLM answers your question.</div>',
    unsafe_allow_html=True,
)

# ── Upload + Prompt ────────────────────────────────────────────────────────────
col_upload, col_settings = st.columns([3, 2])

with col_upload:
    st.markdown('<span class="step-badge">1</span>**Upload Image**', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True, caption=uploaded_file.name)

with col_settings:
    st.markdown('<span class="step-badge">2</span>**Configure**', unsafe_allow_html=True)
    prompt = st.text_area(
        "Your question / instruction",
        value="What do you observe in this image? Describe any objects, people, or scenes in detail.",
        height=120,
    )
    vision_prompt = st.text_input(
        "Vision model prompt (optional override)",
        value="",
        placeholder="Leave blank to use default description prompt",
    )
    stream_mode = st.toggle("⚡ Stream tokens (real-time)", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="step-badge">3</span>**Analyse**', unsafe_allow_html=True)
    run_btn = st.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

# ── Result ─────────────────────────────────────────────────────────────────────
if run_btn and uploaded_file:
    api_url = config.get("api_url", "http://localhost:8080")

    with st.spinner(""):
        render_loading_card("Running vision model + LLM reasoning...")

        try:
            import httpx

            file_bytes = uploaded_file.getvalue()
            files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
            data = {"prompt": prompt}
            if vision_prompt.strip():
                data["vision_prompt"] = vision_prompt.strip()

            if stream_mode:
                # SSE streaming
                st.markdown("---")
                st.markdown("**🧠 Streaming Response**")
                placeholder = st.empty()
                full_text = ""

                with httpx.Client(timeout=120.0) as client:
                    with client.stream(
                        "POST",
                        f"{api_url}/analyze/image/stream",
                        files=files,
                        data={"prompt": prompt},
                    ) as response:
                        for line in response.iter_lines():
                            if line.startswith("data: "):
                                token = line[6:]
                                if token == "[DONE]":
                                    break
                                full_text += token
                                placeholder.markdown(
                                    f'<div style="background:#F0F7FF;border-left:4px solid #1A56DB;'
                                    f'border-radius:0 12px 12px 0;padding:16px 20px;line-height:1.7;">'
                                    f'{full_text}▌</div>',
                                    unsafe_allow_html=True,
                                )
                placeholder.markdown(
                    f'<div style="background:#F0F7FF;border-left:4px solid #1A56DB;'
                    f'border-radius:0 12px 12px 0;padding:16px 20px;line-height:1.7;">'
                    f'{full_text}</div>',
                    unsafe_allow_html=True,
                )
                # Final stats
                st.success("✅ Analysis complete")
                st.caption(f"Streamed {len(full_text.split())} words")

            else:
                # Batch mode
                with httpx.Client(timeout=120.0) as client:
                    resp = client.post(
                        f"{api_url}/analyze/image",
                        files=files,
                        data=data,
                    )
                resp.raise_for_status()
                result = resp.json()

                st.empty()  # clear loading card

                render_result_card(
                    vision_description=result.get("vision_description", ""),
                    final_answer=result.get("final_answer", ""),
                    vision_model=result.get("vision_model", config.get("vision_model", "—")),
                    llm_model=config.get("llm_model", "—"),
                    total_latency_ms=result.get("total_latency_ms", 0),
                    llm_tokens_used=result.get("llm_tokens_used", 0),
                )

            # --- PHASE 9 FEEDBACK UI ---
            st.markdown("---")
            st.markdown("### 🧠 Continuous Learning Feedback")
            st.caption("Help fine-tune Sentinel by correcting hallucinations or missing details.")
            with st.form("feedback_form"):
                rating_choice = st.radio("Was the analysis accurate?", ["👍 Good (+1)", "👎 Poor (-1)"], horizontal=True)
                correction = st.text_area("Supervised Correction (optional)", help="Provide the ground-truth answer here.")
                submit_fb = st.form_submit_button("Submit RLHF Feedback")
                
                if submit_fb:
                    rating = 1 if "👍" in rating_choice else -1
                    ans = full_text if stream_mode else result.get("final_answer", "")
                    payload = {
                        "prompt": prompt,
                        "original_answer": ans,
                        "rating": rating,
                        "image_hash": "frontend_submission",
                        "user_correction": correction if correction.strip() else None
                    }
                    try:
                        import httpx
                        r = httpx.post(f"{api_url}/feedback/", json=payload, headers={"X-API-Key": "sk-admin-secret-key-123"})
                        if r.status_code == 200:
                            st.success("✅ Feedback integrated into the Flywheel dataset!")
                        else:
                            st.error(f"Error: {r.text}")
                    except Exception as fe:
                        st.error(f"Failed: {fe}")

        except httpx.ConnectError:
            st.empty()
            render_error_card(
                f"Cannot reach API at {api_url}.<br>"
                "Start the server: <code>uvicorn api.main:app --port 8080</code>"
            )
        except Exception as e:
            st.empty()
            render_error_card(str(e))

elif not uploaded_file:
    st.markdown("""
    <div class="upload-zone">
      <div style="font-size:3rem;margin-bottom:12px;">📂</div>
      <div style="font-weight:600;color:#475569;font-size:1rem;">Upload an image to get started</div>
      <div style="color:#94A3B8;font-size:0.85rem;margin-top:4px;">JPG · PNG · WebP · BMP</div>
    </div>
    """, unsafe_allow_html=True)

# ── Tips ───────────────────────────────────────────────────────────────────────
with st.expander("💡 Tips for better results"):
    st.markdown("""
    - **High-resolution images** give more detail to the vision model
    - **Specific prompts** like *"List all visible safety hazards"* outperform vague ones
    - Enable **RAG** in the sidebar if you have relevant knowledge base documents ingested
    - Use **stream mode** for visual feedback during long inferences
    """)
