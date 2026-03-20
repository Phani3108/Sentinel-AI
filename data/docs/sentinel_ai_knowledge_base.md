# Sentinel AI — Internal Knowledge Base

## Project Description

Sentinel AI is a fully local, enterprise-grade multimodal AI pipeline for
processing images and video using open-source vision models and LLMs.
All inference happens on-premise — no data leaves the firewall.

## Core Capabilities

### Vision Models Supported
- **LLaVA 1.6** (7B, 13B, 34B): Best open-source visual question answering (VQA)
- **Florence-2** (0.2B–0.77B): Microsoft model, ultra-fast, fine-tunable, Apache 2.0
- **InternVL2** (2B–26B): Excellent OCR, chart understanding, and document analysis
- **Phi-3.5 Vision** (4.2B): Microsoft model, optimized for edge devices
- **Moondream2** (1.9B): Lightweight captioning and VQA for edge deployment

### Pipeline Flow

1. **Input**: Image file, video file, or webcam stream
2. **Preprocessing**: Resize, normalize, frame extraction (video)
3. **Vision Model**: Generates textual description of visual content
4. **RAG Retrieval** (optional): Retrieve relevant internal documents for context
5. **LLM Reasoning**: Ollama LLM produces final detailed analysis
6. **Output**: Structured JSON response with latency and token metrics

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | System health check |
| GET | /models | List available vision models |
| POST | /analyze/image | Analyze a single image |
| POST | /analyze/image/stream | Stream analysis tokens (SSE) |
| WS | /ws/analyze | WebSocket real-time analysis |
| GET | /metrics | Prometheus metrics endpoint |

## Configuration

All settings are loaded from `.env` or environment variables. Key settings:

- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `DEFAULT_VISION_MODEL`: Default vision model (default: llava)
- `DEVICE`: Compute device — cpu, cuda, or mps (default: cpu)
- `RAG_TOP_K`: Number of documents to retrieve per query (default: 5)
- `CHROMA_COLLECTION`: ChromaDB collection name (default: sentinel_docs)

## Docker Services

The docker-compose stack includes:
- **sentinel-api**: FastAPI application on port 8080
- **ollama**: Local LLM server on port 11434
- **chromadb**: Vector database on port 8000
- **prometheus**: Metrics scraper on port 9090
- **grafana**: Dashboard UI on port 3000
- **jaeger**: Distributed trace viewer on port 16686
