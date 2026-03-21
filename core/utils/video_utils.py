"""
Sentinel AI — Video Frame Extraction Utilities
Extract frames from video files for multimodal inference.
Uses OpenCV for frame reading and ffmpeg for format support.
"""
import logging
import tempfile
import subprocess
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

def _ensure_mp4_format(video_path: Path) -> Path:
    """
    Phase 26: Omni-Lens FFmpeg Transcoder.
    Intercepts physical enterprise codecs (MKV, AVI, WEBM, HEVC) and normalizes 
    them natively into MP4 geometry so the OpenCV pipeline does not violently crash.
    """
    valid_extensions = ['.mp4', '.mov']
    if video_path.suffix.lower() in valid_extensions:
        return video_path
        
    logger.warning(f"Phase 26 Transcoder: Intercepted unsupported codec [{video_path.suffix}]. Booting native FFmpeg matrix...")
    temp_dir = Path(tempfile.mkdtemp(prefix="sentinel_transcode_"))
    out_path = temp_dir / f"{video_path.stem}_normalized.mp4"
    
    try:
        # Ultra fast silent transcode omitting audio streams
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path), 
            "-c:v", "libx264", "-preset", "ultrafast", 
            "-an", str(out_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logger.info(f"Phase 26 Transcoder: Mathematical realignment successful. Output -> {out_path.name}")
        return out_path
    except Exception as e:
        logger.error(f"FATAL transcoder failure. FFmpeg sequence aborted: {e}")
        return video_path


@dataclass
class VideoFrame:
    """A single extracted video frame with metadata."""
    index: int               # Frame number (0-indexed)
    timestamp_sec: float     # Time position in video
    image_path: Path         # Path to saved frame (JPG)
    width: int
    height: int


@dataclass
class VideoMetadata:
    """Basic metadata about a video file."""
    path: str
    fps: float
    total_frames: int
    duration_sec: float
    width: int
    height: int
    codec: str


def get_video_metadata(video_path: Union[str, Path]) -> VideoMetadata:
    """Extract metadata from a video file using OpenCV."""
    import cv2
    video_path = _ensure_mp4_format(Path(video_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    cap.release()

    return VideoMetadata(
        path=str(video_path),
        fps=fps,
        total_frames=total_frames,
        duration_sec=total_frames / fps if fps > 0 else 0,
        width=width,
        height=height,
        codec=codec,
    )


def extract_frames(
    video_path: Path,
    output_dir: Optional[Path] = None,
    fps: float = 1.0,
    max_frames: int = 50,
    resize: Optional[tuple] = None,
) -> List[VideoFrame]:
    """
    Extract frames from a video file at a target frame rate.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save frames. Uses a temp dir if None.
        fps: Frames per second to extract (1.0 = 1 frame/sec)
        max_frames: Limit on number of frames to extract
        resize: Optional (width, height) tuple to resize frames

    Returns:
        List of VideoFrame objects with image paths

    Example:
        frames = extract_frames("clip.mp4", fps=0.5, max_frames=20)
        for frame in frames:
            result = vision_model.analyze(frame.image_path)
    """
    import cv2

    video_path = _ensure_mp4_format(Path(video_path))
    if output_dir is None:
        _tmp_dir = tempfile.mkdtemp(prefix="sentinel_frames_")
        output_dir = Path(_tmp_dir)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(video_fps / fps))

    frames: List[VideoFrame] = []
    frame_idx = 0
    saved_count = 0

    logger.info(
        f"Extracting frames from {video_path.name} "
        f"at {fps}fps (interval={frame_interval}), max={max_frames}"
    )

    while True:
        ret, frame = cap.read()
        if not ret or saved_count >= max_frames:
            break

        if frame_idx % frame_interval == 0:
            if resize:
                frame = cv2.resize(frame, resize)

            timestamp = frame_idx / video_fps
            filename = output_dir / f"frame_{saved_count:04d}_{timestamp:.2f}s.jpg"
            cv2.imwrite(str(filename), frame)

            h, w = frame.shape[:2]
            frames.append(VideoFrame(
                index=frame_idx,
                timestamp_sec=round(timestamp, 3),
                image_path=filename,
                width=w,
                height=h,
            ))
            saved_count += 1

        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {len(frames)} frames → {output_dir}")
    return frames


def extract_keyframes(
    video_path: Path,
    output_dir: Optional[Path] = None,
    max_frames: int = 20,
) -> List[VideoFrame]:
    """
    Extract scene-change keyframes using frame difference detection.
    More semantic than time-based extraction — captures scene transitions.

    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        max_frames: Maximum number of keyframes

    Returns:
        List of VideoFrame objects
    """
    import cv2
    import numpy as np

    video_path = _ensure_mp4_format(Path(video_path))
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="sentinel_keyframes_"))
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    prev_frame_gray = None
    keyframes: List[VideoFrame] = []
    frame_idx = 0
    DIFF_THRESHOLD = 30.0  # Mean absolute difference threshold

    while len(keyframes) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame_gray is not None:
            diff = cv2.absdiff(gray, prev_frame_gray)
            score = float(np.mean(diff))
            if score > DIFF_THRESHOLD:
                timestamp = frame_idx / fps
                filename = output_dir / f"keyframe_{len(keyframes):04d}_{timestamp:.2f}s.jpg"
                cv2.imwrite(str(filename), frame)
                h, w = frame.shape[:2]
                keyframes.append(VideoFrame(
                    index=frame_idx,
                    timestamp_sec=round(timestamp, 3),
                    image_path=filename,
                    width=w,
                    height=h,
                ))

        prev_frame_gray = gray
        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {len(keyframes)} keyframes from {video_path.name}")
    return keyframes


def frames_to_narration_input(frames: List[VideoFrame]) -> List[dict]:
    """
    Convert extracted frames into a structured list for pipeline narration.

    Returns:
        [{"timestamp": 1.5, "image_path": "/path/frame.jpg"}, ...]
    """
    return [
        {"timestamp": f.timestamp_sec, "image_path": str(f.image_path)}
        for f in frames
    ]

