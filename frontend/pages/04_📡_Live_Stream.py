"""
Sentinel AI — Page 4: Live Stream
Real-time WebSocket token streaming for image analysis.
"""
import os
import sys
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

st.set_page_config(page_title="Live Stream · Sentinel AI", page_icon="📡", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background:#FFFFFF; }
[data-testid="stSidebar"] { background:#F8FAFC; border-right:1px solid #E2E8F0; }
.page-header { font-size:2rem; font-weight:800; color:#1E293B; margin-bottom:4px; }
.page-sub { font-size:1rem; color:#64748B; margin-bottom:24px; }
.stream-box {
  background:#0F172A; border-radius:12px; padding:20px 24px;
  font-family:'JetBrains Mono','Fira Code',monospace;
  font-size:.9rem; color:#E2E8F0; min-height:200px;
  line-height:1.7; white-space:pre-wrap; overflow-y:auto;
}
.cursor { display:inline-block; width:2px; height:1em;
          background:#1A56DB; animation:blink 1s step-end infinite; vertical-align:text-bottom; }
@keyframes blink { 50% { opacity:0; } }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

from frontend.components.sidebar import render_sidebar

config = render_sidebar(show_rag=False)
api_url = config.get("api_url", "http://localhost:8080")

st.markdown('<div class="page-header">📡 Live Stream</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Real-time SSE token streaming — watch the AI think, token by token.</div>',
    unsafe_allow_html=True,
)

col_img, col_out = st.columns([1, 2])

with col_img:
    st.markdown("**Upload Image**")
    img_file = st.file_uploader(
        "Image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
    )
    if img_file:
        st.image(img_file, use_container_width=True)

    prompt = st.text_area(
        "Prompt",
        value="Describe this image in rich detail. What do you observe?",
        height=90,
    )
    stream_btn = st.button("▶️ Start Streaming", type="primary",
                           disabled=img_file is None, use_container_width=True)

with col_out:
    st.markdown("**Live Token Stream**")
    stream_placeholder = st.empty()
    stats_placeholder = st.empty()

    if not img_file:
        stream_placeholder.markdown("""
        <div class="stream-box" style="opacity:.5;">
          Upload an image and press Start Streaming...
        </div>
        """, unsafe_allow_html=True)

if stream_btn and img_file:
    import httpx, time

    full_text = ""
    token_count = 0
    t_start = time.perf_counter()

    try:
        with httpx.Client(timeout=180.0) as client:
            with client.stream(
                "POST",
                f"{api_url}/analyze/image/stream",
                files={"file": (img_file.name, img_file.getvalue(), img_file.type)},
                data={"prompt": prompt},
            ) as response:
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    token = line[6:]
                    if token == "[DONE]":
                        break
                    full_text += token
                    token_count += 1
                    elapsed = time.perf_counter() - t_start

                    stream_placeholder.markdown(
                        f'<div class="stream-box">{full_text}'
                        f'<span class="cursor"></span></div>',
                        unsafe_allow_html=True,
                    )
                    stats_placeholder.markdown(
                        f"⚡ `{token_count}` tokens · `{elapsed:.1f}s` · "
                        f"`{token_count / max(elapsed, 0.01):.1f}` tok/s"
                    )

        elapsed = time.perf_counter() - t_start
        stream_placeholder.markdown(
            f'<div class="stream-box">{full_text}</div>',
            unsafe_allow_html=True,
        )
        stats_placeholder.success(
            f"✅ Done — {token_count} tokens in {elapsed:.1f}s "
            f"({token_count / max(elapsed, 0.01):.1f} tok/s)"
        )

    except httpx.ConnectError:
        stream_placeholder.error(
            f"❌ Cannot reach API at {api_url}  \n"
            "Run: `uvicorn api.main:app --port 8080`"
        )
    except Exception as e:
        stream_placeholder.error(f"❌ {str(e)}")

# ── Tips ───────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📖 How it works"):
    st.markdown("""
    **Live Stream** uses Server-Sent Events (SSE) to deliver LLM tokens in real-time:

    1. Your image is sent to `POST /analyze/image/stream`
    2. The vision model describes it (one round-trip)
    3. The LLM streams tokens back via SSE
    4. Each `data: <token>` line is rendered live in the terminal-style box above

    **Latency breakdown:**
    - Vision model: typically 1–5s (depends on model and image size)
    - LLM first token: typically 0.5–2s
    - Streaming rate: limited by Ollama's generation speed (~10–50 tok/s on CPU)
    """)
