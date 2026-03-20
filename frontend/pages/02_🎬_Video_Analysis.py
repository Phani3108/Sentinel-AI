"""
Sentinel AI — Page 2: Video Analysis
Upload a video, extract frames, analyse each frame, get a narrated summary.
"""
import os
import sys
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

st.set_page_config(page_title="Video Analysis · Sentinel AI", page_icon="🎬", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background:#FFFFFF; }
[data-testid="stSidebar"] { background:#F8FAFC; border-right:1px solid #E2E8F0; }
.page-header { font-size:2rem; font-weight:800; color:#1E293B; margin-bottom:4px; }
.page-sub { font-size:1rem; color:#64748B; margin-bottom:24px; }
.frame-card {
  background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px;
  padding:12px; margin-bottom:8px;
}
.timeline-ts { font-size:.75rem; font-weight:700; color:#1A56DB; margin-bottom:4px; }
.timeline-desc { font-size:.85rem; color:#374151; line-height:1.5; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

from frontend.components.sidebar import render_sidebar
from frontend.components.result_card import render_error_card

config = render_sidebar()

st.markdown('<div class="page-header">🎬 Video Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Upload a video and get frame-by-frame descriptions plus a complete narration.</div>',
    unsafe_allow_html=True,
)

col_up, col_cfg = st.columns([3, 2])

with col_up:
    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        label_visibility="collapsed",
    )
    if uploaded_video:
        st.video(uploaded_video)

with col_cfg:
    prompt = st.text_area(
        "Narration question",
        value="Describe what is happening in this video, including any notable events, objects, or people.",
        height=110,
    )
    fps = st.slider("Frame extraction rate (frames/sec)", 0.1, 2.0, 0.5, 0.1)
    max_frames = st.slider("Max frames to analyse", 5, 40, 15)
    use_keyframes = st.toggle("Use keyframe detection", value=False,
                              help="Extract scene-change keyframes instead of uniform frames")
    run_btn = st.button("🚀 Analyse Video", type="primary", use_container_width=True,
                        disabled=uploaded_video is None)

if run_btn and uploaded_video:
    api_url = config.get("api_url", "http://localhost:8080")

    import tempfile, httpx, json

    # Save video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.getvalue())
        tmp_path = tmp.name

    with st.spinner("Extracting frames and analysing..."):
        try:
            with open(tmp_path, "rb") as f:
                files = {"file": (uploaded_video.name, f, uploaded_video.type)}
                data = {
                    "prompt": prompt,
                    "fps": str(fps),
                    "max_frames": str(max_frames),
                    "use_keyframes": str(use_keyframes).lower(),
                }
                with httpx.Client(timeout=300.0) as client:
                    resp = client.post(f"{api_url}/analyze/video", files=files, data=data)
                resp.raise_for_status()
                result = resp.json()

            os.unlink(tmp_path)

            # ── Results ─────────────────────────────────────────────────────────
            st.success(f"✅ Analysed {result.get('total_frames', 0)} frames in "
                       f"{result.get('total_latency_ms', 0):.0f}ms")

            # Narration
            st.markdown("---")
            st.markdown("### 📝 Video Narration")
            st.markdown(
                f'<div style="background:#F0F7FF;border-left:4px solid #1A56DB;'
                f'border-radius:0 12px 12px 0;padding:20px 24px;font-size:1rem;line-height:1.75;">'
                f'{result.get("narration", "No narration generated.")}</div>',
                unsafe_allow_html=True,
            )

            # Frame timeline
            summaries = result.get("frame_summaries", [])
            if summaries:
                st.markdown("---")
                st.markdown(f"### 🎞️ Frame Timeline ({len(summaries)} frames)")
                for frame in summaries:
                    ts = frame.get("timestamp", 0)
                    desc = frame.get("description", "")
                    lat = frame.get("latency_ms", 0)
                    st.markdown(f"""
                    <div class="frame-card">
                      <div class="timeline-ts">⏱ {ts:.1f}s &nbsp;·&nbsp; {lat:.0f}ms</div>
                      <div class="timeline-desc">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

        except httpx.ConnectError:
            render_error_card(f"Cannot reach API at {api_url}.")
        except Exception as e:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            render_error_card(str(e))

elif not uploaded_video:
    st.markdown("""
    <div style="border:2px dashed #CBD5E1;border-radius:16px;padding:48px;text-align:center;background:#F8FAFC;">
      <div style="font-size:3rem;margin-bottom:12px;">🎬</div>
      <div style="font-weight:600;color:#475569;">Upload a video file to get started</div>
      <div style="color:#94A3B8;font-size:.85rem;margin-top:4px;">MP4 · AVI · MOV · MKV · WebM</div>
    </div>
    """, unsafe_allow_html=True)
