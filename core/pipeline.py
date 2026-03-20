"""
Sentinel AI — Main Pipeline Orchestrator

Coordinates the full multimodal inference pipeline:
    Input (Image / Video)
        → Vision Model → Description
        → [RAG retrieval → Context] (optional)
        → LLM Reasoning → Response
        → Observability events

Usage:
    from core.pipeline import SentinelPipeline
    
    pipeline = SentinelPipeline(vision_model="florence2", llm_model="llama3.1:8b")
    result = pipeline.run_image("path/to/image.jpg", "What defects do you see?")
    print(result.final_answer)
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

    Manages vision model + LLM client lifecycle, optionally integrates
    RAG retriever, and provides both image and video inference methods.
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

        # Resolve model names
        vision_name = vision_model or settings.default_vision_model
        llm_name = llm_model or settings.ollama_llm_model
        ollama_url = ollama_base_url or settings.ollama_base_url

        logger.info(f"Initializing SentinelPipeline: vision={vision_name}, llm={llm_name}, device={self.device}")

        # Build vision model (lazy-loaded on first use)
        self.vision = get_vision_model(vision_name, device=self.device)

        # Build LLM client
        self.llm = OllamaLLMClient(
            model=llm_name,
            base_url=ollama_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        # RAG retriever (initialized lazily)
        self._retriever = None
        if enable_rag:
            self._init_rag()

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

        # Step 1: Vision
        logger.info(f"[Vision] Analyzing {image_path.name} with {self.vision.name}")
        v_prompt = vision_prompt or "Describe this image in comprehensive detail."
        vision_result: VisionResult = self.vision.analyze(image_path, v_prompt)
        logger.info(f"[Vision] Done. Latency={vision_result.latency_ms:.0f}ms")

        # Step 2: RAG (optional)
        rag_context = None
        chunks_retrieved = 0
        if self.enable_rag and self._retriever:
            rag_context = self._retrieve_context(prompt + " " + vision_result.description)
            chunks_retrieved = len(rag_context.split("\n---\n")) if rag_context else 0

        # Step 3: LLM Reasoning
        llm_prompt = self._build_llm_prompt(prompt, vision_result)
        logger.info(f"[LLM] Reasoning with {self.llm.model}")

        if stream:
            full_answer = ""
            t_llm = time.perf_counter()
            for token in self.llm.stream(llm_prompt, context=rag_context):
                print(token, end="", flush=True)
                full_answer += token
            print()
            llm_latency = (time.perf_counter() - t_llm) * 1000
            tokens_used = 0
        else:
            t_llm = time.perf_counter()
            llm_resp: LLMResponse = self.llm.generate(llm_prompt, context=rag_context)
            llm_latency = (time.perf_counter() - t_llm) * 1000
            full_answer = llm_resp.content
            tokens_used = llm_resp.total_tokens

        total_latency = (time.perf_counter() - t_start) * 1000
        logger.info(f"[Pipeline] Complete. Total latency={total_latency:.0f}ms")

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
            llm_model=self.llm.model,
            final_answer=full_answer,
            llm_latency_ms=llm_latency,
            llm_tokens_used=tokens_used,
            total_latency_ms=total_latency,
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
        vision_result: VisionResult = self.vision.analyze(image_path, v_prompt)

        llm_prompt = self._build_llm_prompt(prompt, vision_result)

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
        """
        Full pipeline for a video: extract frames → analyze each → narrate.

        Args:
            video: Path to video file
            prompt: The overall question to answer about the video
            fps: Frames per second to extract (0.5 = 1 frame every 2 seconds)
            max_frames: Maximum frames to analyze
            use_keyframes: If True, extract semantic keyframes instead of uniform
            output_dir: Directory to save extracted frames

        Returns:
            VideoNarration with per-frame descriptions and full narration
        """
        t_start = time.perf_counter()
        video_path = Path(video)
        logger.info(f"[Video] Processing {video_path.name}")

        # Extract frames
        if use_keyframes:
            frames: List[VideoFrame] = extract_keyframes(video_path, output_dir, max_frames)
        else:
            frames = extract_frames(video_path, output_dir, fps, max_frames)

        logger.info(f"[Video] Extracted {len(frames)} frames")

        # Analyze each frame
        frame_summaries = []
        all_descriptions = []

        for frame in frames:
            try:
                vr = self.vision.analyze(
                    frame.image_path,
                    "Describe what is happening in this frame. Be specific.",
                )
                frame_summaries.append({
                    "timestamp": frame.timestamp_sec,
                    "description": vr.description,
                    "latency_ms": vr.latency_ms,
                })
                all_descriptions.append(
                    f"[{frame.timestamp_sec:.1f}s] {vr.description}"
                )
            except Exception as e:
                logger.warning(f"Failed to analyze frame {frame.index}: {e}")

        # Build narration prompt from all frame descriptions
        frame_context = "\n".join(all_descriptions)
        narration_prompt = (
            f"Here are descriptions of sequential video frames:\n\n{frame_context}\n\n"
            f"Based on these frames, {prompt}"
        )

        rag_context = None
        if self.enable_rag and self._retriever:
            rag_context = self._retrieve_context(prompt)

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
        """Initialize the RAG retriever (lazy)."""
        try:
            from .rag import SentinelRetriever
            self._retriever = SentinelRetriever()
            logger.info("✅ RAG retriever initialized.")
        except ImportError:
            logger.warning("RAG module not yet built — running without RAG.")
            self._retriever = None

    def _retrieve_context(self, query: str) -> Optional[str]:
        """Retrieve relevant context from the vector store."""
        if not self._retriever:
            return None
        try:
            docs = self._retriever.retrieve(query)
            return "\n---\n".join(d["text"] for d in docs)
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_llm_prompt(user_prompt: str, vision_result: VisionResult) -> str:
        """Build the LLM prompt from the user question + vision description."""
        return (
            f"You are analyzing an image/video. "
            f"The visual content has been described by a vision model as follows:\n\n"
            f"VISION MODEL OUTPUT:\n{vision_result.description}\n\n"
            f"USER QUESTION: {user_prompt}\n\n"
            f"Provide a detailed, accurate analysis based on the visual description."
        )

    def health_check(self) -> dict:
        """Return health status of all pipeline components."""
        return {
            "vision_model": self.vision.name,
            "vision_loaded": self.vision._loaded,
            "llm_model": self.llm.model,
            "llm_available": self.llm.is_available(),
            "rag_enabled": self.enable_rag,
            "device": self.device,
        }

    def __repr__(self) -> str:
        return (
            f"SentinelPipeline("
            f"vision={self.vision.name}, "
            f"llm={self.llm.model}, "
            f"device={self.device}, "
            f"rag={self.enable_rag})"
        )
