"""
Sentinel AI — Page 3: RAG Explorer
Ingest documents into ChromaDB, query the knowledge base, and see retrieved chunks.
"""
import os
import sys
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

st.set_page_config(page_title="RAG Explorer · Sentinel AI", page_icon="📚", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background:#FFFFFF; }
[data-testid="stSidebar"] { background:#F8FAFC; border-right:1px solid #E2E8F0; }
.page-header { font-size:2rem; font-weight:800; color:#1E293B; margin-bottom:4px; }
.page-sub { font-size:1rem; color:#64748B; margin-bottom:24px; }
.chunk-card {
  background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px;
  padding:16px 20px; margin:8px 0; border-left:4px solid #1A56DB;
}
.chunk-score {
  font-size:.72rem; font-weight:700; color:#1A56DB;
  text-transform:uppercase; letter-spacing:.05em;
}
.chunk-source { font-size:.75rem; color:#94A3B8; margin-bottom:8px; }
.chunk-text { font-size:.9rem; color:#374151; line-height:1.6; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

from frontend.components.sidebar import render_sidebar

config = render_sidebar(show_model_selector=False, show_rag=True)

st.markdown('<div class="page-header">📚 RAG Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Ingest documents into your private knowledge base, then query it semantically.</div>',
    unsafe_allow_html=True,
)

tab_ingest, tab_query, tab_manage = st.tabs(["📥 Ingest Documents", "🔍 Query", "⚙️ Manage Collection"])

# ── Tab 1: Ingest ──────────────────────────────────────────────────────────────
with tab_ingest:
    st.markdown("### Upload files to ingest into the knowledge base")
    uploaded_docs = st.file_uploader(
        "Upload documents",
        type=["txt", "md", "pdf", "docx", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        chunk_size = st.number_input("Chunk size (characters)", 100, 2000, 500, 50)
    with col_b:
        overlap = st.number_input("Overlap (characters)", 0, 500, 50, 10)

    if uploaded_docs:
        st.markdown(f"**{len(uploaded_docs)} file(s) ready to ingest:**")
        for f in uploaded_docs:
            st.markdown(f"- 📄 `{f.name}` ({f.size // 1024}KB)")

    if st.button("📥 Ingest Documents", type="primary", disabled=not uploaded_docs):
        try:
            import httpx, tempfile, os

            api_url = config.get("api_url", "http://localhost:8080")
            progress = st.progress(0)
            success_count = 0

            for i, doc in enumerate(uploaded_docs):
                suffix = "." + doc.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(doc.getvalue())
                    tmp_path = tmp.name

                with open(tmp_path, "rb") as f:
                    with httpx.Client(timeout=60.0) as client:
                        resp = client.post(
                            f"{api_url}/rag/ingest",
                            files={"file": (doc.name, f, doc.type)},
                            data={"chunk_size": str(chunk_size), "overlap": str(overlap)},
                        )
                os.unlink(tmp_path)
                if resp.status_code == 200:
                    success_count += 1
                progress.progress((i + 1) / len(uploaded_docs))

            st.success(f"✅ Successfully ingested {success_count}/{len(uploaded_docs)} document(s)!")
        except Exception as e:
            st.error(f"Ingest failed: {e}")

    # ── Or ingest sample docs ──────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Or ingest sample knowledge base")
    st.caption("Loads the 3 sample docs from `data/docs/` (maintenance manual, safety policy, knowledge base)")

    if st.button("📚 Ingest Sample Docs"):
        try:
            import httpx
            api_url = config.get("api_url", "http://localhost:8080")
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(f"{api_url}/rag/ingest/samples")
            if resp.status_code == 200:
                r = resp.json()
                st.success(f"✅ Ingested {r.get('chunks_added', '?')} chunks from sample docs")
            else:
                st.error(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            st.error(str(e))

# ── Tab 2: Query ───────────────────────────────────────────────────────────────
with tab_query:
    st.markdown("### Semantic Search")
    query = st.text_input(
        "Query",
        placeholder="e.g. What maintenance schedule should I follow for the pump?",
    )
    top_k = st.slider("Number of results", 1, 10, config.get("top_k", 3))

    if st.button("🔍 Search", type="primary", disabled=not query.strip()):
        try:
            import httpx
            api_url = config.get("api_url", "http://localhost:8080")
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{api_url}/rag/query",
                    json={"query": query, "top_k": top_k},
                )
            resp.raise_for_status()
            results = resp.json().get("results", [])

            if not results:
                st.info("No results found. Try ingesting documents first.")
            else:
                st.markdown(f"**{len(results)} results found:**")
                for i, r in enumerate(results):
                    score = r.get("score", 0)
                    text = r.get("text", "")
                    metadata = r.get("metadata", {})
                    source = metadata.get("source", "unknown")

                    # Score bar
                    score_pct = int(score * 100)
                    st.markdown(f"""
                    <div class="chunk-card">
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                          <span class="chunk-score">#{i+1} — Score: {score:.3f}</span>
                          <div class="chunk-source">📄 {source}</div>
                        </div>
                        <div style="text-align:right;">
                          <div style="width:80px;height:6px;background:#E2E8F0;border-radius:3px;">
                            <div style="width:{score_pct}%;height:100%;background:#1A56DB;border-radius:3px;"></div>
                          </div>
                          <div style="font-size:.7rem;color:#94A3B8;margin-top:2px;">{score_pct}%</div>
                        </div>
                      </div>
                      <div class="chunk-text">{text}</div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(str(e))

# ── Tab 3: Manage ──────────────────────────────────────────────────────────────
with tab_manage:
    st.markdown("### Collection Management")
    api_url = config.get("api_url", "http://localhost:8080")

    col_info, col_actions = st.columns(2)

    with col_info:
        st.markdown("**Collection Stats**")
        if st.button("🔄 Refresh Stats"):
            try:
                import httpx
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(f"{api_url}/rag/stats")
                if resp.status_code == 200:
                    stats = resp.json()
                    st.metric("Documents", stats.get("document_count", 0))
                    st.metric("Collection", stats.get("collection_name", "—"))
                    st.metric("Embedding Model", stats.get("embedding_model", "—"))
                else:
                    st.warning("Could not fetch stats")
            except Exception as e:
                st.error(str(e))

    with col_actions:
        st.markdown("**Actions**")
        st.warning("⚠️ Resetting the collection will delete all ingested documents.")
        if st.button("🗑️ Reset Collection", type="secondary"):
            try:
                import httpx
                with httpx.Client(timeout=30.0) as client:
                    resp = client.delete(f"{api_url}/rag/reset")
                if resp.status_code == 200:
                    st.success("✅ Collection reset successfully.")
                else:
                    st.error(f"HTTP {resp.status_code}")
            except Exception as e:
                st.error(str(e))
