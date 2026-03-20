"""
Sentinel AI — Instrumented Pipeline Orchestrator

This is the Phase 2/3 enhanced version of pipeline.py that adds:
  - RAG integration (ChromaDB + sentence-transformers)
  - OpenTelemetry spans on every stage
  - Prometheus metrics recording

The public API is identical to Phase 1 — this is a drop-in replacement.
"""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Union

from .config import get_settings
from .vision import get_vision_model, VisionResult
from .llm import OllamaLLMClient, LLMResponse
from .utils.image_utils import load_image, resize_image, get_image_info
from .utils.video_utils import extract_frames, extract_keyframes, VideoFrame

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result from the Sentinel AI pipeline."""
    # Inputs
    input_path: str
    input_type: str  # 'image' | 'video'
    prompt: str

    # Vision stage
    vision_model: str
    vision_description: str
    vision_latency_ms: float = 0.0
    vision_memory_mb: float = 0.0

    # RAG stage (optional)
    rag_context: Optional[str] = None
    rag_chunks_retrieved: int = 0
    rag_latency_ms: float = 0.0

    # LLM stage
    llm_model: str = ""
    final_answer: str = ""
    llm_latency_ms: float = 0.0
    llm_tokens_used: int = 0

    # Overall
    total_latency_ms: float = 0.0
    timestamp: str = ""

    # Video-specific
    frames_analyzed: int = 0
    frame_descriptions: List[dict] = field(default_factory=list)


@dataclass
class VideoNarration:
    """Result of narrating a full video."""
    video_path: str
    total_frames: int
    narration: str
    frame_summaries: List[dict] = field(default_factory=list)
    total_latency_ms: float = 0.0


class SentinelPipeline:
    """
    Main pipeline orchestrator for Sentinel AI.

    Phases 2 & 3 enhancements:
    - Fully integrated RAG (ChromaDB + sentence-transformers)
    - OpenTelemetry tracing on vision, RAG, and LLM stages
    - Prometheus metrics for latency, throughput, and errors
    """

    def __init__(
        self,
        vision_model: str = "llava",
        llm_model: Optional[str] = None,
        device: Optional[str] = None,
        enable_rag: bool = False,
        ollama_base_url: Optional[str] = None,
    ):
        settings = get_settings()
        self.device = device or settings.device
        self.enable_rag = enable_rag
        self._settings = settings

        # Resolve model names
        vision_name = vision_model or settings.default_vision_model
        llm_name = llm_model or settings.ollama_llm_model
        ollama_url = ollama_base_url or settings.ollama_base_url

        logger.info(
            f"Initializing SentinelPipeline: "
            f"vision={vision_name}, llm={llm_name}, device={self.device}"
        )

        # Vision model
        self.vision = get_vision_model(vision_name, device=self.device)

        # LLM client
        import os
        if os.getenv("VLLM_API_BASE"):
            from core.llm.vllm_client import VLLMClient
            self.llm = VLLMClient(base_url=os.getenv("VLLM_API_BASE"), model=llm_name)
        else:
            self.llm = OllamaLLMClient(
                model=llm_name,
                base_url=ollama_url,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )

        # RAG retriever (optional)
        self._retriever = None
        if enable_rag:
            self._init_rag()

        # Monitoring (lazy init)
        self._tracer = None
        self._metrics = None
        self._init_monitoring()

    # ------------------------------------------------------------------ #
    # Monitoring init
    # ------------------------------------------------------------------ #

    def _init_monitoring(self) -> None:
        """Initialize OTEL tracer and Prometheus metrics (non-fatal)."""
        try:
            from monitoring.otel_setup import get_tracer
            self._tracer = get_tracer("sentinel.pipeline")
            logger.debug("OTEL tracer initialized for pipeline")
        except Exception:
            self._tracer = None

        try:
            from monitoring.prometheus_metrics import METRICS
            self._metrics = METRICS
            logger.debug("Prometheus metrics initialized for pipeline")
        except Exception:
            self._metrics = None

    def _get_span(self, name: str):
        """Return an OTEL span context manager (or no-op if unavailable)."""
        if self._tracer:
            return self._tracer.start_as_current_span(name)

        class _Noop:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def set_attribute(self, *args): pass
            def record_exception(self, *args): pass

        return _Noop()

    # ------------------------------------------------------------------ #
    # Image Inference
    # ------------------------------------------------------------------ #

    def run_image(
        self,
        image: Union[str, Path],
        prompt: str = "Describe this image in detail. What do you observe?",
        vision_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> PipelineResult:
        """
        Full pipeline for a single image.

        Args:
            image: Path to image file
            prompt: Question/instruction for the LLM reasoning step
            vision_prompt: Override for the vision model step (optional)
            stream: If True, streams LLM output (prints tokens, returns final result)

        Returns:
            PipelineResult with vision description + LLM answer
        """
        from datetime import datetime, timezone

        t_start = time.perf_counter()
        image_path = Path(image)

        # -- Step 1: Vision --
        with self._get_span("vision_inference") as span:
            if hasattr(span, "set_attribute"):
                span.set_attribute("model.name", self.vision.name)
                span.set_attribute("image.path", str(image_path))

            logger.info(f"[Vision] Analyzing {image_path.name} with {self.vision.name}")
            v_prompt = vision_prompt or "Describe this image in comprehensive detail."

            t_vision = time.perf_counter()
            try:
                vision_result: VisionResult = self.vision.analyze(image_path, v_prompt)
            except Exception as e:
                if hasattr(span, "record_exception"):
                    span.record_exception(e)
                if self._metrics:
                    self._metrics.errors_total.labels(
                        component="vision", error_type=type(e).__name__
                    ).inc()
                raise

            vision_latency_s = time.perf_counter() - t_vision
            if hasattr(span, "set_attribute"):
                span.set_attribute("latency_ms", vision_result.latency_ms)

            # Record Prometheus vision latency
            if self._metrics:
                self._metrics.vision_latency.labels(
                    model=self.vision.name
                ).observe(vision_latency_s)

            logger.info(f"[Vision] Done. Latency={vision_result.latency_ms:.0f}ms")

        # -- Step 2: RAG --
        rag_context = None
        chunks_retrieved = 0
        rag_latency_ms = 0.0

        if self.enable_rag and self._retriever:
            with self._get_span("rag_retrieve") as span:
                if hasattr(span, "set_attribute"):
                    span.set_attribute("rag.top_k", self._settings.rag_top_k)

                t_rag = time.perf_counter()
                try:
                    rag_context = self._retrieve_context(
                        prompt + " " + vision_result.description
                    )
                    chunks_retrieved = len(rag_context.split("\n---\n")) if rag_context else 0
                except Exception as e:
                    logger.warning(f"RAG retrieval failed: {e}")
                    rag_context = None

                rag_latency_ms = (time.perf_counter() - t_rag) * 1000
                if hasattr(span, "set_attribute"):
                    span.set_attribute("rag.chunks_returned", chunks_retrieved)

                if self._metrics:
                    self._metrics.rag_latency.labels(
                        collection=self._settings.chroma_collection
                    ).observe(rag_latency_ms / 1000)
                    self._metrics.rag_retrievals_total.labels(
                        outcome="hit" if chunks_retrieved > 0 else "miss"
                    ).inc()

        # -- Step 3: LLM --
        with self._get_span("llm_generate") as span:
            llm_prompt = self._build_llm_prompt(prompt, vision_result, rag_context)
            if hasattr(span, "set_attribute"):
                span.set_attribute("model.name", self.llm.model)
                span.set_attribute("prompt.length", len(llm_prompt))

            logger.info(f"[LLM] Reasoning with {self.llm.model}")
            t_llm = time.perf_counter()

            # Step 3: Final LLM Answer Generation
            start_llm = time.perf_counter()
            if stream:
                full_answer = ""
                # Stream directly, ReAct agent doesn't natively stream intermediate thoughts in this version
                for token in self.llm.stream(llm_prompt, context=rag_context):
                    print(token, end="", flush=True)
                    full_answer += token
                print()
                tokens_used = 0
            else:
                if use_agent:
                    from core.agent.orchestrator import ReActOrchestrator
                    logger.info("Executing via ReAct Autonomous Agent...")
                    orchestrator = ReActOrchestrator(self.llm)
                    full_answer = orchestrator.run(vision_result.description, llm_prompt)
                    tokens_used = 0 # Approx tracking for multi-step agent
                else:
                    llm_resp: LLMResponse = self.llm.generate(llm_prompt, context=rag_context)
                    full_answer = llm_resp.content
                    tokens_used = llm_resp.total_tokens
            llm_latency_ms = (time.perf_counter() - start_llm) * 1000

            if hasattr(span, "set_attribute"):
                span.set_attribute("tokens.total", tokens_used)
                span.set_attribute("latency_ms", llm_latency_ms)

            if self._metrics:
                self._metrics.llm_latency.labels(
                    model=self.llm.model
                ).observe(llm_latency_ms / 1000)
                if tokens_used > 0:
                    self._metrics.llm_tokens_total.labels(
                        model=self.llm.model, type="completion"
                    ).inc(tokens_used)

        total_latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info(f"[Pipeline] Complete. Total latency={total_latency_ms:.0f}ms")

        # Record full pipeline run
        if self._metrics:
            self._metrics.record_pipeline_run(
                vision_model=self.vision.name,
                llm_model=self.llm.model,
                vision_latency_s=vision_latency_s,
                llm_latency_s=llm_latency_ms / 1000,
                total_latency_s=total_latency_ms / 1000,
                tokens_used=tokens_used,
                success=True,
                rag_hits=chunks_retrieved,
            )

        return PipelineResult(
            input_path=str(image_path),
            input_type="image",
            prompt=prompt,
            vision_model=self.vision.name,
            vision_description=vision_result.description,
            vision_latency_ms=vision_result.latency_ms,
            vision_memory_mb=vision_result.memory_mb,
            rag_context=rag_context,
            rag_chunks_retrieved=chunks_retrieved,
            rag_latency_ms=rag_latency_ms,
            llm_model=self.llm.model,
            final_answer=full_answer,
            llm_latency_ms=llm_latency_ms,
            llm_tokens_used=tokens_used,
            total_latency_ms=total_latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def stream_image(
        self,
        image: Union[str, Path],
        prompt: str = "Describe this image in detail.",
        vision_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Stream LLM tokens for an image analysis request.
        Yields tokens as strings for real-time WebSocket / SSE delivery.
        """
        image_path = Path(image)
        v_prompt = vision_prompt or "Describe this image in comprehensive detail."

        with self._get_span("vision_inference") as span:
            if hasattr(span, "set_attribute"):
                span.set_attribute("model.name", self.vision.name)
            vision_result: VisionResult = self.vision.analyze(image_path, v_prompt)

        # RAG context for streaming
        rag_context = None
        if self.enable_rag and self._retriever:
            rag_context = self._retrieve_context(prompt + " " + vision_result.description)

        llm_prompt = self._build_llm_prompt(prompt, vision_result, rag_context)

        # First yield the vision description as structured context
        yield f"[VISION]: {vision_result.description}\n\n[ANALYSIS]: "

        # Then stream the LLM response
        for token in self.llm.stream(llm_prompt):
            yield token

    # ------------------------------------------------------------------ #
    # Video Inference
    # ------------------------------------------------------------------ #

    def run_video(
        self,
        video: Union[str, Path],
        prompt: str = "Describe what is happening in this video.",
        fps: float = 0.5,
        max_frames: int = 20,
        use_keyframes: bool = False,
        output_dir: Optional[Path] = None,
    ) -> VideoNarration:
        """Full pipeline for a video: extract frames → analyze each → narrate."""
        t_start = time.perf_counter()
        video_path = Path(video)
        logger.info(f"[Video] Processing {video_path.name}")

        # Extract frames
        with self._get_span("video_frame_extraction") as span:
            if use_keyframes:
                frames: List[VideoFrame] = extract_keyframes(video_path, output_dir, max_frames)
            else:
                frames = extract_frames(video_path, output_dir, fps, max_frames)

            if hasattr(span, "set_attribute"):
                span.set_attribute("frames.extracted", len(frames))

        logger.info(f"[Video] Extracted {len(frames)} frames")

        # Analyze each frame
        frame_summaries = []
        all_descriptions = []

        for frame in frames:
            try:
                with self._get_span("vision_inference") as span:
                    if hasattr(span, "set_attribute"):
                        span.set_attribute("frame.index", frame.index)
                    vr = self.vision.analyze(
                        frame.image_path,
                        "Describe what is happening in this frame. Be specific.",
                    )
                frame_summaries.append({
                    "timestamp": frame.timestamp_sec,
                    "description": vr.description,
                    "latency_ms": vr.latency_ms,
                })
                all_descriptions.append(f"[{frame.timestamp_sec:.1f}s] {vr.description}")

                if self._metrics:
                    self._metrics.video_frames_total.labels(model=self.vision.name).inc()

            except Exception as e:
                logger.warning(f"Failed to analyze frame {frame.index}: {e}")

        # Build narration
        frame_context = "\n".join(all_descriptions)
        narration_prompt = (
            f"Here are descriptions of sequential video frames:\n\n{frame_context}\n\n"
            f"Based on these frames, {prompt}"
        )

        rag_context = None
        if self.enable_rag and self._retriever:
            rag_context = self._retrieve_context(prompt)

        with self._get_span("llm_generate") as span:
            if hasattr(span, "set_attribute"):
                span.set_attribute("operation", "video_narration")
            narration_resp = self.llm.generate(narration_prompt, context=rag_context)

        total_latency = (time.perf_counter() - t_start) * 1000
        logger.info(f"[Video] Narration complete. Total latency={total_latency:.0f}ms")

        return VideoNarration(
            video_path=str(video_path),
            total_frames=len(frames),
            narration=narration_resp.content,
            frame_summaries=frame_summaries,
            total_latency_ms=total_latency,
        )

    # ------------------------------------------------------------------ #
    # RAG Integration
    # ------------------------------------------------------------------ #

    def _init_rag(self) -> None:
        """Initialize the RAG retriever."""
        try:
            from .rag import SentinelRetriever
            settings = self._settings
            self._retriever = SentinelRetriever(
                collection_name=settings.chroma_collection,
                top_k=settings.rag_top_k,
            )
            logger.info("✅ RAG retriever initialized.")
            if self._metrics:
                self._metrics.rag_documents_count.labels(
                    collection=settings.chroma_collection
                ).set(self._retriever.document_count)
        except ImportError as e:
            logger.warning(f"RAG module unavailable: {e} — running without RAG.")
            self._retriever = None

    def _retrieve_context(self, query: str) -> Optional[str]:
        """Retrieve relevant context from the vector store."""
        if not self._retriever:
            return None
        try:
            return self._retriever.retrieve_as_context(query)
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_llm_prompt(
        user_prompt: str,
        vision_result: VisionResult,
        rag_context: Optional[str] = None,
    ) -> str:
        """Build the LLM prompt from the user question + vision description + RAG context."""
        prompt = (
            f"You are analyzing an image/video. "
            f"The visual content has been described by a vision model as follows:\n\n"
            f"VISION MODEL OUTPUT:\n{vision_result.description}\n\n"
        )

        if rag_context:
            prompt += (
                f"RELEVANT CONTEXT FROM KNOWLEDGE BASE:\n{rag_context}\n\n"
            )

        prompt += f"USER QUESTION: {user_prompt}\n\n"
        prompt += "Provide a detailed, accurate analysis based on the visual description"
        if rag_context:
            prompt += " and retrieve context"
        prompt += "."

        return prompt

    def health_check(self) -> dict:
        """Return health status of all pipeline components."""
        return {
            "vision_model": self.vision.name,
            "vision_loaded": self.vision._loaded,
            "llm_model": self.llm.model,
            "llm_available": self.llm.is_available(),
            "rag_enabled": self.enable_rag,
            "rag_docs": self._retriever.document_count if self._retriever else 0,
            "device": self.device,
            "otel_enabled": self._tracer is not None,
            "metrics_enabled": self._metrics is not None,
        }

    def __repr__(self) -> str:
        return (
            f"SentinelPipeline("
            f"vision={self.vision.name}, "
            f"llm={self.llm.model}, "
            f"device={self.device}, "
            f"rag={self.enable_rag})"
        )
