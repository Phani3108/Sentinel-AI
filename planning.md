# 🧠 Sentinel AI — Private Multimodal AI Stack

> **Project Vision**: A fully local, enterprise-grade multimodal AI pipeline that processes images and video using open-source vision models + LLMs, with RAG over internal docs — all running on-premise with zero data leaving the firewall.

---

## 📌 Why This Project

| Problem | Solution |
|---|---|
| Enterprises can't send sensitive images/video to cloud APIs | Fully local vision + LLM pipeline |
| No visibility into local model tradeoffs | Benchmarking suite (CPU vs GPU, speed vs accuracy) |
| No precedent for hybrid routing | Sensitive → Local, Non-sensitive → Cloud |
| Gaps in: monitoring, real-time inference, fine-tuning | Built as core pillars of this project |

**Learning Goals (personal):**
- Gain deep expertise in image & video models
- Build production-grade observability for AI systems
- Learn real-time streaming inference pipelines
- Get hands-on with fine-tuning multimodal models

---

## 🗺️ Project Architecture

```
INPUT LAYER
  Images / Video / Documents
       ↓
PREPROCESSING LAYER
  Frame extraction (video)
  Resize / normalize
  OCR (optional, for documents)
       ↓
VISION MODEL LAYER (Local)
  LLaVA 1.6 / Florence-2 / InternVL / Kosmos-2
       ↓
CONTEXT ASSEMBLY
  Vision output + metadata + retrieved docs
       ↓
LLM REASONING LAYER (Local via Ollama)
  Mistral / Llama 3 / Phi-3
       ↓
RAG LAYER
  ChromaDB / Qdrant → internal knowledge base
       ↓
OUTPUT
  Structured JSON / natural language response
  Audit trail / observability events
```

---

## 🏗️ Core Pillars (Phases)

### Phase 1 — Foundation (Local Inference Pipeline)
Build and validate the end-to-end local pipeline from image/video → LLaVA → Ollama LLM → response.

### Phase 2 — RAG Integration
Connect ChromaDB/Qdrant to retrieve internal documents as context before the LLM reasoning step.

### Phase 3 — Monitoring & Observability _(Gap #1)_
Build a full telemetry layer: latency tracing, memory profiling, accuracy tracking, model drift alerts using OpenTelemetry + Prometheus + Grafana.

### Phase 4 — Real-Time Multimodal Application _(Gap #2)_
Build a live inference UI: webcam/video stream → vision model → LLM → real-time response using FastAPI WebSockets + Streamlit or Next.js.

### Phase 5 — Fine-Tuning _(Gap #3)_
Fine-tune Florence-2 or LLaVA on a domain-specific dataset. Track experiments with MLflow or W&B.

### Phase 6 — Hybrid Routing (Cloud/Local)
Build a sensitivity classifier that routes: sensitive content → local pipeline, non-sensitive → OpenAI/Azure GPT-4V.

---

## 🔬 Model Landscape

### Vision Models (Local)

| Model | Size | Strengths | Notes |
|---|---|---|---|
| LLaVA 1.6 (34B) | 34B | Best open-source VQA | Requires GPU |
| LLaVA 1.6 (7B) | 7B | CPU-feasible | Good starting point |
| Florence-2 | 0.2B–0.77B | Tiny, fast, fine-tunable | Microsoft, Apache 2.0 |
| InternVL2 | 2B–26B | Strong on OCR/charts | Chinese research, excellent |
| Kosmos-2 | 1.6B | Grounding + referring | Microsoft |
| Phi-3.5 Vision | 4.2B | Microsoft, small + capable | Great for edge |
| Moondream2 | 1.9B | Ultra-lightweight | Edge devices |

### LLM Backend (via Ollama)

| Model | Notes |
|---|---|
| Llama 3.1 8B | Best open-source 8B |
| Mistral 7B | Fast, efficient |
| Phi-3 Medium | Microsoft, strong reasoning |
| Qwen2.5 | Strong multilingual |

---

## 🧪 Benchmarking Suite

### Metrics to Track
- **Latency**: Time to first token (TTFT), total inference time
- **Memory**: Peak VRAM, peak RAM
- **Accuracy**: vs GPT-4V on standard VQA benchmarks (VQAv2, TextVQA, DocVQA)
- **Throughput**: Frames/second for video processing

### Benchmark Configurations
- CPU (MacBook M1/M2/M3) vs GPU (CUDA / Apple Metal MPS)
- Quantization levels: FP32, INT8, INT4 (GGUF)
- Batch sizes: 1, 4, 8

---

## 📡 Monitoring & Observability Architecture _(Phase 3)_

```
App (FastAPI)
    → OpenTelemetry SDK
        → OTLP Exporter
            → Prometheus (metrics)
            → Jaeger / Tempo (traces)
            → Loki (logs)
                → Grafana Dashboard
```

**Dashboards to build:**
- Model latency heatmaps
- Memory usage over time
- Accuracy vs latency tradeoff scatter
- Error rate and retry counts
- Request routing (local vs cloud) tracking

---

## 🎥 Real-Time Multimodal Application _(Phase 4)_

**Stack:**
- FastAPI + WebSockets (backend streaming)
- OpenCV (camera / video capture)
- LLaVA / Phi-3 Vision (frame analysis)
- Ollama (LLM reasoning)
- Streamlit or Next.js (frontend)

**Use Cases to Demo:**
1. Live webcam → describe what's happening every N seconds
2. Upload video → scene-by-scene narration
3. Screen recording → auto-generate incident report
4. Document image → extract structured data

---

## 🎛️ Fine-Tuning Plan _(Phase 5)_

### Datasets to Use
- **Florence-2**: Fine-tune on custom document/chart images
- **LLaVA**: LoRA fine-tune on domain-specific VQA pairs
- **InternVL**: Fine-tune on medical/industrial image QA

### Tooling
- Hugging Face Transformers + PEFT (LoRA/QLoRA)
- MLflow or Weights & Biases for experiment tracking
- Custom dataset builder (annotation tool → JSONL format)

### Fine-Tuning Workflow
```
Raw images (domain-specific)
    → Annotation (LabelStudio or manual)
        → Convert to LLaVA/Florence format
            → LoRA fine-tune (GPU)
                → Evaluate on held-out set
                    → Export to GGUF / ONNX
                        → Deploy via Ollama / HF Inference
```

---

## 🔀 Hybrid Routing Architecture _(Phase 6)_

```
Request arrives
    → Sensitivity Classifier (local NLP/rules)
        → SENSITIVE: route to local pipeline
        → NON-SENSITIVE: route to cloud (GPT-4V / Gemini)
    → Unified response format
    → Cost + latency tracking per route
```

**Classifier signals:**
- Contains PII (faces, documents, IDs)
- Internal branding detected
- File metadata (internal network origin)
- User/org-level policy settings

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Vision Models | LLaVA, Florence-2, InternVL, Phi-3V |
| LLM Runtime | Ollama |
| Vector DB | ChromaDB (start) → Qdrant (production) |
| API Layer | FastAPI |
| Streaming | WebSockets, Server-Sent Events |
| Monitoring | OpenTelemetry + Prometheus + Grafana |
| Tracing | Jaeger / Tempo |
| Experiment Tracking | MLflow |
| Fine-tuning | HuggingFace + PEFT (LoRA/QLoRA) |
| Frontend | Streamlit (Phase 4A) → Next.js (Phase 4B) |
| Containerization | Docker + Docker Compose |
| Hardware Support | CPU, CUDA (NVIDIA), Apple Metal (MPS) |

---

## 📁 Planned Project Structure

```
sentinel-ai/
├── planning.md                  ← this file
├── tasks.md                     ← task tracker
├── README.md
│
├── core/
│   ├── vision/                  ← vision model wrappers
│   │   ├── llava.py
│   │   ├── florence.py
│   │   ├── internvl.py
│   │   └── base.py
│   ├── llm/                     ← Ollama LLM client
│   │   └── ollama_client.py
│   ├── rag/                     ← RAG pipeline
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   └── vectorstore.py
│   └── pipeline.py              ← orchestrates vision → LLM → RAG
│
├── api/
│   ├── main.py                  ← FastAPI entrypoint
│   ├── routes/
│   └── websocket_handler.py     ← real-time streaming
│
├── benchmark/
│   ├── run_benchmark.py
│   ├── metrics.py
│   └── reports/
│
├── monitoring/
│   ├── otel_setup.py
│   ├── prometheus_metrics.py
│   └── dashboards/              ← Grafana JSON exports
│
├── finetune/
│   ├── dataset_builder.py
│   ├── train_lora.py
│   └── evaluate.py
│
├── router/
│   └── hybrid_router.py         ← local vs cloud routing
│
├── frontend/
│   └── app.py / pages/          ← Streamlit or Next.js
│
├── data/
│   ├── sample_images/
│   ├── sample_videos/
│   └── docs/                    ← RAG knowledge base
│
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfiles/
│
└── tests/
    ├── test_vision.py
    ├── test_pipeline.py
    └── test_api.py
```

---

## 📅 Timeline (Suggested)

| Phase | Duration | Milestone |
|---|---|---|
| Phase 1: Foundation | Week 1–2 | End-to-end image → LLaVA → Ollama → response |
| Phase 2: RAG | Week 3 | RAG over internal docs working |
| Phase 3: Observability | Week 4–5 | Grafana dashboard live with latency/memory metrics |
| Phase 4: Real-Time App | Week 6–7 | Live webcam inference UI demo |
| Phase 5: Fine-Tuning | Week 8–10 | Fine-tuned Florence-2 on custom dataset |
| Phase 6: Hybrid Router | Week 11–12 | Sensitivity-based cloud/local routing |
| Phase 7: Ent. Security | Week 13–14 | RBAC, API Keys, Audit Logs, Data Masking |
| Phase 8: Scaling & HA | Week 15–16 | K8s, Async Queues, Redis Cache, vLLM engine |
| Phase 9: Data Flywheel | Week 17–18 | RLHF/DPO pipeline based on user feedback |
| Phase 10: Agentic | Week 19-20 | Multi-agent orchestration and dynamic tool use |

---

## 🧭 Session Log

| Date | Session | Summary |
|---|---|---|
| 2026-03-20 | Session 001 | Project kickoff — planning, architecture, task breakdown created |

---

*Last updated: 2026-03-20 | Session: 001*
