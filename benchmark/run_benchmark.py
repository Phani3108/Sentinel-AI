"""
Sentinel AI — Benchmarking Suite
Measures latency, memory, and accuracy across vision models and hardware configs.

Usage:
    python benchmark/run_benchmark.py --models llava florence2 moondream --device cpu
"""
import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import psutil
from rich.console import Console
from rich.table import Table

from core.vision import get_vision_model

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class BenchmarkResult:
    """Result for a single model × image × prompt combination."""
    model_name: str
    device: str
    image_path: str
    prompt: str
    latency_ms: float
    memory_delta_mb: float
    output_length: int
    description_preview: str  # First 200 chars
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    """Full benchmark report across models."""
    timestamp: str
    device: str
    models: List[str]
    results: List[BenchmarkResult] = field(default_factory=list)
    system_info: dict = field(default_factory=dict)


def get_system_info() -> dict:
    """Collect hardware and OS info for context."""
    import platform
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "python": platform.python_version(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = torch.backends.mps.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = "not installed"
    return info


def run_single_benchmark(
    model_name: str,
    image_path: Path,
    prompt: str,
    device: str,
    warmup: bool = True,
) -> BenchmarkResult:
    """
    Run a single benchmark: load model → optionally warm up → time inference.
    """
    try:
        model = get_vision_model(model_name, device=device)

        # Warm-up run (don't count toward latency)
        if warmup:
            try:
                model.analyze(image_path, prompt)
            except Exception:
                pass  # Warm-up may fail on first load — that's ok

        # Timed run
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024

        t0 = time.perf_counter()
        result = model.analyze(image_path, prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        mem_after = process.memory_info().rss / 1024 / 1024

        return BenchmarkResult(
            model_name=result.model_name,
            device=device,
            image_path=str(image_path),
            prompt=prompt,
            latency_ms=round(elapsed_ms, 1),
            memory_delta_mb=round(mem_after - mem_before, 1),
            output_length=len(result.description),
            description_preview=result.description[:200],
            success=True,
        )

    except Exception as e:
        logger.error(f"Benchmark failed for {model_name}: {e}")
        return BenchmarkResult(
            model_name=model_name,
            device=device,
            image_path=str(image_path),
            prompt=prompt,
            latency_ms=0,
            memory_delta_mb=0,
            output_length=0,
            description_preview="",
            success=False,
            error=str(e),
        )


def run_benchmark(
    models: List[str],
    image_paths: List[Path],
    prompt: str = "Describe this image in detail.",
    device: str = "cpu",
    output_dir: Path = Path("benchmark/reports"),
) -> BenchmarkReport:
    """
    Run a full benchmark across multiple models and images.

    Args:
        models: List of model names to benchmark
        image_paths: List of image files to test on
        prompt: Common prompt for all models
        device: 'cpu', 'cuda', or 'mps'
        output_dir: Directory to save JSON + Markdown reports

    Returns:
        BenchmarkReport with all results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print(f"\n[bold cyan]🔬 Sentinel AI Benchmark[/bold cyan]")
    console.print(f"Models: {models}")
    console.print(f"Images: {len(image_paths)}")
    console.print(f"Device: {device}\n")

    report = BenchmarkReport(
        timestamp=timestamp,
        device=device,
        models=models,
        system_info=get_system_info(),
    )

    for model_name in models:
        for image_path in image_paths:
            console.print(f"  Testing [yellow]{model_name}[/yellow] on {image_path.name}...")
            result = run_single_benchmark(model_name, image_path, prompt, device)
            report.results.append(result)

            status = "✅" if result.success else "❌"
            console.print(
                f"  {status} {model_name} — "
                f"{result.latency_ms:.0f}ms | "
                f"{result.memory_delta_mb:+.0f}MB | "
                f"{result.output_length} chars"
            )

    # Print summary table
    _print_summary_table(report)

    # Save reports
    json_path = output_dir / f"benchmark_{timestamp}.json"
    md_path = output_dir / f"benchmark_{timestamp}.md"

    with open(json_path, "w") as f:
        json.dump(
            {**asdict(report), "results": [asdict(r) for r in report.results]},
            f,
            indent=2,
        )

    _save_markdown_report(report, md_path)
    console.print(f"\n[green]Reports saved:[/green]\n  {json_path}\n  {md_path}")

    return report


def _print_summary_table(report: BenchmarkReport) -> None:
    """Print a Rich table summarizing benchmark results."""
    table = Table(title="Benchmark Summary", show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Memory ΔMB", justify="right")
    table.add_column("Output Chars", justify="right")
    table.add_column("Status")

    for r in report.results:
        status = "✅" if r.success else f"❌ {r.error[:40]}"
        table.add_row(
            r.model_name,
            f"{r.latency_ms:.0f}",
            f"{r.memory_delta_mb:+.0f}",
            str(r.output_length),
            status,
        )

    console.print("\n")
    console.print(table)


def _save_markdown_report(report: BenchmarkReport, path: Path) -> None:
    """Save benchmark results as a Markdown report."""
    lines = [
        f"# Sentinel AI Benchmark Report",
        f"",
        f"**Date**: {report.timestamp}  ",
        f"**Device**: {report.device}  ",
        f"**Models tested**: {', '.join(report.models)}",
        f"",
        f"## System",
        f"",
        f"| Property | Value |",
        f"|---|---|",
    ]
    for k, v in report.system_info.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        f"",
        f"## Results",
        f"",
        f"| Model | Image | Latency (ms) | Memory ΔMB | Output Chars | Status |",
        f"|---|---|---|---|---|---|",
    ]
    for r in report.results:
        img = Path(r.image_path).name
        status = "✅" if r.success else f"❌"
        lines.append(
            f"| {r.model_name} | {img} | {r.latency_ms:.0f} | {r.memory_delta_mb:+.0f} | {r.output_length} | {status} |"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinel AI Benchmarking Suite")
    parser.add_argument(
        "--models", nargs="+",
        default=["moondream", "florence2"],
        help="Vision models to benchmark"
    )
    parser.add_argument("--images", type=str, default="data/sample_images",
                        help="Path to image file or directory")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output", type=str, default="benchmark/reports")
    parser.add_argument("--prompt", type=str,
                        default="Describe this image in detail.")
    args = parser.parse_args()

    images_path = Path(args.images)
    if images_path.is_dir():
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    else:
        image_files = [images_path]

    if not image_files:
        console.print("[red]No images found. Add images to data/sample_images/ first.[/red]")
        exit(1)

    run_benchmark(
        models=args.models,
        image_paths=image_files,
        prompt=args.prompt,
        device=args.device,
        output_dir=Path(args.output),
    )
