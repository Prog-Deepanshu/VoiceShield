#!/usr/bin/env python3
"""
VoiceGuard — Latency Benchmark
================================
Measures end-to-end inference latency for different model configurations.

Reports P50, P95, P99 latencies and compares configurations.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --model aasist --device cuda --iterations 200
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_benchmark(
    model_variant: str = "aasist_l",
    device: str = "auto",
    iterations: int = 100,
    include_preprocessing: bool = True,
):
    """Run latency benchmark."""
    import torch

    from config.settings import Settings
    from core.inference_engine import AASISTInferenceEngine
    from core.preprocessor import AudioPreprocessor

    print("=" * 60)
    print("  VoiceGuard Latency Benchmark")
    print("=" * 60)

    # Override settings for benchmark
    settings = Settings(
        model_variant=model_variant,
        device=device,
        enable_torchscript=True,
        enable_half_precision=device != "cpu" and torch.cuda.is_available(),
    )

    print(f"  Model:          {model_variant}")
    print(f"  Device:         {settings.device}")
    print(f"  TorchScript:    {settings.enable_torchscript}")
    print(f"  Half Precision: {settings.enable_half_precision}")
    print(f"  Iterations:     {iterations}")
    print(f"  Preprocessing:  {include_preprocessing}")
    print("-" * 60)

    # Load engine
    engine = AASISTInferenceEngine(settings)
    try:
        engine.load()
    except Exception as e:
        print(f"\n⚠ Model loading failed: {e}")
        print("  Ensure weights are downloaded: python scripts/download_weights.py")
        print("  Running benchmark with random weights...")

        # Force load with random weights for benchmarking
        from models.aasist import Model as AASISTModel
        model_config = settings.load_model_config()
        engine.model = AASISTModel(model_config).to(engine.device)
        engine.model.eval()
        engine._is_loaded = True

    preprocessor = AudioPreprocessor(target_sr=16000) if include_preprocessing else None

    # Generate synthetic test audio (white noise at 16kHz, 6 seconds)
    test_audio = np.random.randn(16000 * 6).astype(np.float32) * 0.5

    # Warmup
    print("\nWarmup (5 iterations)...")
    for _ in range(5):
        if include_preprocessing:
            engine.infer(test_audio, sample_rate=16000)
        else:
            tensor = torch.randn(1, settings.nb_samp).to(engine.device)
            engine.infer_tensor(tensor)

    # Benchmark
    print(f"\nRunning {iterations} iterations...")
    latencies = []
    preprocess_times = []
    inference_times = []

    for i in range(iterations):
        start = time.perf_counter()

        if include_preprocessing:
            # Full pipeline: preprocess + infer
            prep_start = time.perf_counter()
            audio_tensor = preprocessor.preprocess(test_audio, sample_rate=16000)
            prep_end = time.perf_counter()
            preprocess_times.append((prep_end - prep_start) * 1000)

            inf_start = time.perf_counter()
            result = engine.infer_tensor(audio_tensor)
            inf_end = time.perf_counter()
            inference_times.append((inf_end - inf_start) * 1000)
        else:
            tensor = torch.randn(1, settings.nb_samp).to(engine.device)
            result = engine.infer_tensor(tensor)

        total = (time.perf_counter() - start) * 1000
        latencies.append(total)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{iterations}] avg={np.mean(latencies):.2f}ms")

    # Report
    latencies = np.array(latencies)
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n  Total End-to-End Latency ({iterations} iterations):")
    print(f"    Mean:   {np.mean(latencies):8.2f} ms")
    print(f"    Median: {np.median(latencies):8.2f} ms")
    print(f"    P50:    {np.percentile(latencies, 50):8.2f} ms")
    print(f"    P95:    {np.percentile(latencies, 95):8.2f} ms")
    print(f"    P99:    {np.percentile(latencies, 99):8.2f} ms")
    print(f"    Min:    {np.min(latencies):8.2f} ms")
    print(f"    Max:    {np.max(latencies):8.2f} ms")
    print(f"    Std:    {np.std(latencies):8.2f} ms")

    if include_preprocessing and preprocess_times:
        preprocess_times = np.array(preprocess_times)
        inference_times = np.array(inference_times)
        print(f"\n  Preprocessing Breakdown:")
        print(f"    Mean:   {np.mean(preprocess_times):8.2f} ms")
        print(f"    P95:    {np.percentile(preprocess_times, 95):8.2f} ms")
        print(f"\n  Model Inference Breakdown:")
        print(f"    Mean:   {np.mean(inference_times):8.2f} ms")
        print(f"    P95:    {np.percentile(inference_times, 95):8.2f} ms")

    # Pass/Fail
    target_ms = 200
    p95 = np.percentile(latencies, 95)
    status = "✓ PASS" if p95 < target_ms else "✗ FAIL"
    print(f"\n  Target: < {target_ms}ms (P95)")
    print(f"  Result: {status} (P95 = {p95:.2f}ms)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="VoiceGuard Latency Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="aasist_l",
        choices=["aasist", "aasist_l"],
        help="Model variant to benchmark",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="Skip preprocessing (measure model inference only)",
    )
    args = parser.parse_args()

    run_benchmark(
        model_variant=args.model,
        device=args.device,
        iterations=args.iterations,
        include_preprocessing=not args.no_preprocessing,
    )


if __name__ == "__main__":
    main()
