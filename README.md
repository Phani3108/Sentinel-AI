# Sentinel AI

An ultra-premium, production-grade Artificial Intelligence dashboard for Multimodal Active Monitoring, Video Intelligence, and Semantic RAG operations. 

Built with **Next.js**, **FastAPI**, **Celery**, and **Ollama**.

````carousel
![Active Sentinel Live Tracker](/Users/phani.m/.gemini/antigravity/brain/d3111712-5c2a-47f4-ba03-115bf8233b8b/sentinel_live_1774083070226.png)
<!-- slide -->
![Multimodal Analysis Sandbox](/Users/phani.m/.gemini/antigravity/brain/d3111712-5c2a-47f4-ba03-115bf8233b8b/nextjs_image_final_1774080964697.png)
<!-- slide -->
![Monitoring Telemetry](/Users/phani.m/.gemini/antigravity/brain/d3111712-5c2a-47f4-ba03-115bf8233b8b/final_monitoring_1774007479645.png)
````

## ⚡ Core Capabilities

- **Active WebRTC Sentinel**: Hook directly into physical webcams. Monitor streams live with bi-directional WebSockets and configure natural-language "Tripwires" to automatically map and raise threat events to the Incident Dashboard.
- **Multimodal Sandbox**: Run complex multimodal LLMs locally without cloud API costs using our SSE `Fetch API` chunk sequence decoders for lightning-fast DOM rendering.
- **Video Intel Pipeline**: Extract and chunk massive `.mp4` payloads synchronously utilizing **Celery + Redis** worker detachments.
- **Local RAG Integration**: Semantic search against PDF documents with continuous `all-MiniLM-L6-v2` embeddings in ChromaDB collections.
- **Google Labs UI**: A heavily elevated enterprise interface styled using Framer Motion micro-animations and a custom Terracotta/Desert Sand configuration.

## 🚀 Quickstart

Everything you need to launch the pipeline out-of-the-box.

```bash
# 1. Clone & Install
git clone https://github.com/Phani3108/Sentinel-AI.git
cd Sentinel-AI
pip install -r requirements.txt
npm install --prefix web

# 2. Boot Local Infrastructure (requires Redis/Celery)
docker-compose -f docker/docker-compose.yml up -d redis worker

# 3. Boot Web Sockets & FastAPI
uvicorn api.main:app --port 8080

# 4. Launch the Next.js UI (Google Labs Theme)
cd web && npm run dev
# Go to http://localhost:3000
```
