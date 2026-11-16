#!/usr/bin/env python3
"""
VoiceGuard — Analyze File
=========================
A convenience script to run a single audio file through the VoiceGuard
pipeline without starting the FastAPI server.

Usage:
    python scripts/analyze_file.py /path/to/audio.wav
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from core.feature_extractor import FeatureExtractor
from core.inference_engine import AASISTInferenceEngine
from core.risk_classifier import RiskClassifier
import soundfile as sf
import numpy as np


def analyze_file(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found exactly at {path}")
        sys.exit(1)

    from core.ensemble import EnsembleScorer
    from core.inference_engine import Wav2Vec2InferenceEngine
    
    print(f"Loading AASIST model ({settings.model_variant.value}) on {settings.device}...")
    engine_aasist = AASISTInferenceEngine(settings)
    engine_aasist.load()
    
    print("Loading Wav2Vec2 model (ensemble)...")
    engine_w2v2 = Wav2Vec2InferenceEngine()
    engine_w2v2.load()
    
    scorer = EnsembleScorer(aasist_weight=0.3, wav2vec2_weight=0.7) # Prioritize W2V2 for cross-domain stability
    
    classifier = RiskClassifier(
        threshold_low=settings.risk_threshold_low,
        threshold_high=settings.risk_threshold_high,
    )

    print(f"Decoding audio file: {path.name}")
    try:
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1) # Convert to mono
    except Exception as e:
        print(f"Error reading audio file: {e}")
        sys.exit(1)

    duration = len(audio) / sr
    print(f"Audio length: {duration:.2f} seconds ({sr} Hz)")

    if duration < 1.0:
        print("Error: Audio is too short. Minimum 1 second length required.")
        sys.exit(1)

    print("Running inference...")
    res_aasist = engine_aasist.infer(audio, sample_rate=sr)
    res_w2v2 = engine_w2v2.infer(audio, sample_rate=sr)
    
    result = scorer.score(res_aasist, res_w2v2)
    
    risk = classifier.classify(result["deepfake_score"], confidence=result["confidence"])
    
    # Feature Extraction (Diagnostics)
    fe = FeatureExtractor(sample_rate=sr)
    spectral = fe.analyze(audio)

    print("\n" + "=" * 50)
    print("  Analysis Results")
    print("=" * 50)
    print(f"Deepfake Risk Label  : {risk.risk_level.value}")
    print(f"Deepfake Score (0-1) : {result['deepfake_score']:.4f}")
    print(f"Recommended Action   : {risk.recommended_action.value}")
    print(f"Inference Latency    : {result['total_latency_ms']:.2f} ms")
    
    print("-" * 50)
    print("  Spectral Diagnostics")
    print("-" * 50)
    print(f"Pitch Stability      : {spectral.pitch_stability:.4f} (lower is usually more stable/synthetic-like)")
    print(f"Spectral Flattening  : {spectral.spectral_flatness:.4f}")
    
    print("\nDescription:")
    print(risk.description)
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze an audio file for deepfakes.")
    parser.add_argument("file", type=str, help="Path to the WAV/FLAC audio file")
    args = parser.parse_args()
    
    analyze_file(args.file)