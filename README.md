# 🛡️ Sentinel AI — Private Multimodal AI Stack

> **Fully local, enterprise-grade multimodal inference pipeline.**  
> Images & video → Vision Model → LLM → RAG — zero data leaves your infrastructure.

---

## 🎯 What This Is

Sentinel AI is a production-ready, on-premise multimodal AI system that:
- Runs **vision models** (LLaVA, Florence-2, InternVL, Phi-3 Vision) **entirely locally**
- Connects to a **local LLM** (via [Ollama](https://ollama.com)) for reasoning
- Grounds responses in **internal documents** via RAG (ChromaDB/Qdrant)
- Benchmarks **latency, memory, and accuracy** across models and hardware
- Includes **full observability** (OpenTelemetry + Prometheus + Grafana)
- Supports **real-time streaming** via WebSockets
- Enables **fine-tuning** on domain-specific datasets (LoRA/QLoRA)
- Routes **sensitive content locally**, non-sensitive to cloud (hybrid mode)

---

## 🏗️ Architecture

```
Input (Image / Video / Document)
        ↓
[Preprocessing] — resize, frame extraction, normalize
        ↓
[Vision Model] — LLaVA / Florence-2 / InternVL / Phi-3V
        ↓
[Context Assembly] — vision output + RAG retrieved docs
        ↓
[LLM Reasoning] — Ollama (Llama3 / Mistral / Phi-3)
        ↓
[Response] — structured JSON or natural language
        ↓
[Observability] — latency, memory, accuracy → Grafana
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running
- (Optional) NVIDIA GPU with CUDA, or Apple Silicon (MPS)

### 1. Install Ollama & Pull Models
```bash
# Install Ollama (macOS)
brew install ollama

# Pull the LLM and vision model
ollama pull llama3.1:8b
ollama pull llava:7b
```

### 2. Set Up Python Environment
```bash
cd sentinel-ai
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env
# Edit .env — set DEVICE=mps (Mac) or DEVICE=cuda (NVIDIA GPU)
```

### 4. Run the Pipeline (CLI)
```bash
python -m core.pipeline --image data/sample_images/sample.jpg --prompt "What do you see?"
```

### 5. Run the API Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
# Visit: http://localhost:8080/docs
```

### 6. Run Benchmarks
```bash
python benchmark/run_benchmark.py --models llava florence2 moondream --device cpu
```

### 7. Launch Full Stack (Docker)
```bash
docker compose -f docker/docker-compose.yml up -d
```

---

## 📦 Project Structure

```
sentinel-ai/
├── core/
│   ├── vision/          # Vision model wrappers
│   ├── llm/             # Ollama LLM client
│   ├── rag/             # RAG pipeline (embedder + retriever)
│   └── utils/           # Image/video preprocessing
├── api/                 # FastAPI server + WebSocket streaming
├── benchmark/           # Latency/memory/accuracy benchmarking
├── monitoring/          # OpenTelemetry + Prometheus config
├── finetune/            # LoRA/QLoRA fine-tuning workflow
├── router/              # Hybrid local/cloud routing
├── frontend/            # Streamlit / Next.js UI
├── data/                # Sample images, videos, RAG docs
├── docker/              # Docker Compose + Dockerfiles
└── tests/               # Unit + integration tests
```

---

## 🔬 Supported Vision Models

| Model | Size | Hardware | Via |
|---|---|---|---|
| LLaVA 1.6 7B | 7B | CPU/GPU | Ollama |
| Florence-2 Base | 0.23B | CPU | HuggingFace |
| Moondream2 | 1.9B | CPU | HuggingFace |
| Phi-3.5 Vision | 4.2B | CPU/GPU/MPS | HuggingFace |
| InternVL2-2B | 2B | CPU/GPU | HuggingFace |

---

## 📊 Benchmarking

Run model comparisons across hardware devices:

```bash
python benchmark/run_benchmark.py \
  --models llava florence2 moondream \
  --device cpu \
  --images data/sample_images/ \
  --output benchmark/reports/
```

---

## 📡 Monitoring

After Docker launch, access:
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686

---

## 🛠️ Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check .

# Format
black .
```

---

## 📋 Project Docs
- [`planning.md`](planning.md) — Architecture, phases, model landscape
- [`tasks.md`](tasks.md) — Task tracker and session log

---

## 📄 License

MIT — see [`LICENSE`](LICENSE)
