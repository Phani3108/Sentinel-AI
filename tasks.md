# ✅ Sentinel AI — Task Tracker

> **Rule**: Every prompt from the user and every task completed by the AI is recorded here. Tasks are checked off when completed. This is the canonical record of all work done on this project.

---

## 📋 Task Status Legend

- `[ ]` — Not started
- `[/]` — In progress
- `[x]` — Completed
- `[~]` — Deferred / on hold
- `[!]` — Blocked

---

## 🗂️ Session Log

### Session 001 — 2026-03-20 (Project Kickoff)

**User Prompt:**
> Build a Private Multimodal AI Stack (On-Prem Vision + LLM). Fully local multimodal pipeline: image/video → vision model (LLaVA/Kosmos/Florence) → Ollama LLM → RAG over internal docs. Benchmark latency (CPU vs GPU), memory, accuracy. Gaps to fill: Monitoring & Observability, Real-time multimodal app, Fine-tuning. Suggest deeper integrations, expansion directions, planning, and maintain tasks.md + planning.md throughout.

**Tasks:**
- [x] Review project idea and design full architecture
- [x] Research vision model landscape (LLaVA, Florence-2, InternVL, Kosmos-2, Phi-3V, Moondream)
- [x] Research LLM backend options (Ollama, Mistral, Llama3, Phi-3)
- [x] Define 6 project phases (Foundation → RAG → Observability → Real-time → Fine-tune → Hybrid Router)
- [x] Design benchmarking suite (latency, memory, accuracy metrics)
- [x] Design Monitoring & Observability architecture (OpenTelemetry + Prometheus + Grafana)
- [x] Design Real-Time Multimodal Application architecture (FastAPI WebSockets + OpenCV)
- [x] Design Fine-tuning workflow (LoRA/QLoRA with PEFT + MLflow)
- [x] Design Hybrid Routing architecture (sensitivity classifier → local/cloud routing)
- [x] Define project folder structure
- [x] Create `planning.md` — master planning document
- [x] Create `tasks.md` — this task tracker

---

### Session 002 — 2026-03-20 (Phase 1 Build)

**User Prompt:**
> Continue with building based on best recommendations.

**Tasks:**
- [x] Create full project directory structure
- [x] Write `requirements.txt` with all core dependencies
- [x] Write `.env.example` with all config keys
- [x] Write `README.md` with quickstart guide
- [x] Build `core/vision/base.py` — abstract base class + VisionResult + VisionModelConfig
- [x] Build `core/vision/llava.py` — LLaVA via Ollama (sync + streaming)
- [x] Build `core/vision/florence.py` — Florence-2 (all task types: caption, OCR, OD, grounding)
- [x] Build `core/vision/moondream.py` — Moondream2 ultra-light (caption + VQA)
- [x] Build `core/vision/phi3v.py` — Phi-3.5 Vision (chat template, MPS/CUDA)
- [x] Build `core/vision/internvl.py` — InternVL2 (OCR/chart specialist)
- [x] Build `core/vision/__init__.py` — registry + `get_vision_model()` factory
- [x] Build `core/llm/ollama_client.py` — Ollama client (sync, async, streaming, chat)
- [x] Build `core/llm/__init__.py`
- [x] Build `core/utils/image_utils.py` — load, resize, normalize, base64
- [x] Build `core/utils/video_utils.py` — frame extraction + keyframe detection
- [x] Build `core/utils/__init__.py`
- [x] Build `core/config.py` — Pydantic Settings from .env
- [x] Build `core/__init__.py`
- [x] Build `core/pipeline.py` — orchestrator: image + video, streaming, RAG hook
- [x] Build `api/main.py` — FastAPI: REST + SSE streaming + WebSocket
- [x] Build `api/__init__.py`
- [x] Build `benchmark/run_benchmark.py` — CLI benchmarker with Rich output + JSON/MD reports
- [x] Build `docker/docker-compose.yml` — Ollama + ChromaDB + Prometheus + Grafana + Jaeger
- [x] Build `monitoring/prometheus.yml` — Prometheus scrape config
- [x] Build `tests/test_vision.py` — tests for wrappers, utils, registry, pipeline

---

## 📦 Phase 1 — Foundation: Local Multimodal Pipeline

### Setup
- [ ] Install Ollama and pull base LLM (llama3.1:8b or mistral:7b)
- [ ] Set up Python virtual environment (uv or conda)
- [ ] Install core dependencies: transformers, torch, Pillow, requests, fastapi
- [ ] Verify hardware: detect CPU/GPU/MPS availability

### Vision Model Wrappers
- [ ] Build `core/vision/base.py` — abstract base class for all vision models
- [ ] Build `core/vision/llava.py` — LLaVA wrapper (via Ollama or HF)
- [ ] Build `core/vision/florence.py` — Florence-2 wrapper (HuggingFace)
- [ ] Build `core/vision/internvl.py` — InternVL wrapper (HuggingFace)
- [ ] Build `core/vision/phi3v.py` — Phi-3.5 Vision wrapper
- [ ] Build `core/vision/moondream.py` — Moondream2 wrapper (ultra-light)

### LLM Client
- [ ] Build `core/llm/ollama_client.py` — Ollama API client with streaming support

### Pipeline Orchestrator
- [ ] Build `core/pipeline.py` — orchestrate vision → context assembly → LLM → response
- [ ] Add image preprocessing utilities (resize, normalize, format conversion)
- [ ] Add video preprocessing (frame extraction with OpenCV)

### Testing
- [ ] Write `tests/test_vision.py` — test each vision model with sample images
- [ ] Write `tests/test_pipeline.py` — end-to-end pipeline test
- [ ] Run smoke test: image → LLaVA → Ollama → response

---

## 📦 Phase 2 — RAG Integration

- [ ] Choose vector DB (ChromaDB for local dev)
- [ ] Build `core/rag/embedder.py` — document embedding (nomic-embed or sentence-transformers)
- [ ] Build `core/rag/vectorstore.py` — ChromaDB interface
- [ ] Build `core/rag/retriever.py` — similarity search + reranking
- [ ] Populate `data/docs/` with sample internal documents (PDF, TXT, MD)
- [ ] Integrate RAG into pipeline: vision output + retrieved context → LLM
- [ ] Write `tests/test_rag.py`
- [ ] Demo: image of a machine part → retrieve relevant maintenance manual → LLM explains issue

---

## 📦 Phase 3 — Monitoring & Observability _(Gap #1)_

- [ ] Install OpenTelemetry SDK (Python)
- [ ] Install Prometheus + Grafana via Docker Compose
- [ ] Install Jaeger or Tempo for distributed tracing
- [ ] Build `monitoring/otel_setup.py` — OTEL tracer/meter/logger initialization
- [ ] Build `monitoring/prometheus_metrics.py` — custom metrics (inference latency, memory, accuracy)
- [ ] Instrument `core/pipeline.py` with OTEL spans
- [ ] Instrument `api/main.py` with request tracing
- [ ] Build Grafana dashboards:
  - [ ] Model latency heatmap by model name
  - [ ] Memory usage over time
  - [ ] Request routing: local vs cloud
  - [ ] Error rate and retry count
  - [ ] Accuracy vs latency scatter plot
- [ ] Set up alerting rules in Grafana (latency > threshold)
- [ ] Document observability setup in `monitoring/README.md`

---

## 📦 Phase 4 — Real-Time Multimodal Application _(Gap #2)_

### Backend
- [ ] Build `api/main.py` — FastAPI app with REST + WebSocket endpoints
- [ ] Build `api/websocket_handler.py` — stream frames from client, return analysis
- [ ] Integrate OpenCV for webcam capture
- [ ] Build frame buffer / queue for real-time processing
- [ ] Add rate limiting and backpressure handling

### Frontend (Phase 4A — Streamlit)
- [ ] Build `frontend/app.py` — Streamlit UI with webcam feed + live LLM output
- [ ] Add video file upload mode
- [ ] Add document image upload + extraction mode

### Frontend (Phase 4B — Next.js)
- [ ] Scaffold Next.js app in `frontend/`
- [ ] Build real-time WebSocket client
- [ ] Build split-view: video input | LLM response stream
- [ ] Add model selector (LLaVA / Florence / Phi-3V)

### Use Cases to Demo
- [ ] Demo 1: Live webcam → describe scene every 5 seconds
- [ ] Demo 2: Upload video → scene-by-scene narration with timestamps
- [ ] Demo 3: Screen recording → auto-generate incident report
- [ ] Demo 4: Document image → extract structured JSON

---

## 📦 Phase 5 — Fine-Tuning _(Gap #3)_

### Dataset Preparation
- [ ] Choose target domain (e.g., document images, industrial parts, medical)
- [ ] Set up LabelStudio locally for annotation
- [ ] Collect / curate 500–2000 image+QA pairs
- [ ] Build `finetune/dataset_builder.py` — convert annotations to LLaVA/Florence JSONL format

### Training
- [ ] Set up GPU environment (CUDA or MPS)
- [ ] Build `finetune/train_lora.py` — LoRA/QLoRA fine-tuning with PEFT
- [ ] Integrate MLflow for experiment tracking
- [ ] Fine-tune Florence-2 (smallest, fastest to iterate)
- [ ] Fine-tune LLaVA 7B with LoRA
- [ ] Track: loss curves, accuracy on eval set, inference speed pre/post fine-tune

### Evaluation & Export
- [ ] Build `finetune/evaluate.py` — compute accuracy, BLEU, ROUGE on held-out set
- [ ] Export fine-tuned model to GGUF format for Ollama
- [ ] Export to ONNX for edge deployment
- [ ] Compare fine-tuned vs base model on benchmark

---

## 📦 Phase 6 — Hybrid Router

- [ ] Build sensitivity classifier (rule-based + small ML model)
  - [ ] PII detection (face detection, personal data patterns)
  - [ ] Internal branding / watermark detection
  - [ ] User/org policy config
- [ ] Build `router/hybrid_router.py` — route to local or cloud
- [ ] Integrate cloud fallback (GPT-4V or Gemini Vision)
- [ ] Track per-request: route taken, cost, latency, response quality
- [ ] Build admin UI panel: routing stats + cost dashboard

---

## 📦 Benchmarking Suite

- [ ] Build `benchmark/run_benchmark.py` — unified benchmark runner
- [ ] Build `benchmark/metrics.py` — latency, memory, accuracy collectors
- [ ] Run benchmarks on: LLaVA 7B, Florence-2, InternVL2-2B, Moondream2
- [ ] Run on: CPU, GPU (CUDA), Apple Metal (MPS)
- [ ] Run with quantization: FP32, INT8, INT4
- [ ] Generate benchmark report (Markdown + charts)
- [ ] Compare vs GPT-4V API baseline

---

## 📦 Infrastructure & DevOps

- [ ] Write `docker/docker-compose.yml` — Ollama + ChromaDB + Prometheus + Grafana + Jaeger
- [ ] Write Dockerfiles for API and frontend
- [ ] Write `.env.example` with all configuration keys
- [ ] Write `README.md` — project overview + quickstart guide
- [ ] Set up GitHub repo
- [ ] Add CI (GitHub Actions): lint + test on push

---

## 🔮 Stretch Goals / Future Ideas

- [ ] Video temporal reasoning: track objects/events across frames
- [ ] Multi-agent mode: vision agent + reasoning agent + retrieval agent
- [ ] Async batch processing queue (Celery + Redis)
- [ ] Edge deployment: export quantized model to Raspberry Pi / Jetson Nano
- [ ] Audio+Vision multimodal (Whisper + LLaVA joint pipeline)
- [ ] Synthetic data generation for fine-tuning (using GPT-4V as teacher)
- [ ] Model versioning and A/B testing infrastructure
- [ ] Compliance audit logs (for enterprise use)

---

## 📝 Prompt History

| Session | Date | Prompt Summary |
|---|---|---|
| 001 | 2026-03-20 | Project kickoff — architecture, planning, task breakdown |

---

*Last updated: 2026-03-20 | Session: 001*
