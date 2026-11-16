#!/usr/bin/env python3
"""
VoiceGuard — Dataset Benchmark
==============================
Runs the VoiceGuard ensemble pipeline over the
'UniDataPro/real-vs-fake-human-voice-deepfake-audio' HuggingFace dataset
and generates a structured tree-view report in a text file.

Usage:
    ./.venv/bin/python scripts/benchmark_dataset.py [--max-files 50] [--output report.txt]
"""

import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from config.settings import settings
from core.inference_engine import AASISTInferenceEngine, Wav2Vec2InferenceEngine
from core.ensemble import EnsembleScorer
from core.risk_classifier import RiskClassifier


def generate_tree_view(results: dict, summary: dict) -> str:
    """Generate a tree view representation of the benchmark results."""
    lines = []
    lines.append("VoiceGuard Dataset Benchmark Report")
    lines.append("===================================")
    lines.append("Dataset: UniDataPro/real-vs-fake-human-voice-deepfake-audio")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Build tree
    for label_str, items in results.items():
        lines.append(f"├── {label_str}")
        for i, item in enumerate(items):
            prefix = "│   ├──" if i < len(items) - 1 else "│   └──"
            
            # Format correctness
            mark = "✓ Correct" if item["correct"] else "✗ Incorrect"
            
            lines.append(f"{prefix} Entry {item['id']}: Risk={item['risk']} (Score: {item['score']:.4f})  {mark}")
    
    lines.append("")
    lines.append("===================================")
    lines.append("  Summary:")
    lines.append("===================================")
    
    for label_str, stat in summary.items():
        accuracy = (stat["correct"] / stat["total"]) * 100 if stat["total"] > 0 else 0
        lines.append(f"Total {label_str}: {stat['total']} -> Accuracy: {accuracy:.1f}%")
        
    total_samples = sum(stat["total"] for stat in summary.values())
    total_correct = sum(stat["correct"] for stat in summary.values())
    overall_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    
    lines.append("-" * 35)
    lines.append(f"Overall Accuracy: {overall_acc:.1f}% ({total_correct}/{total_samples})")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark HF Dataset")
    parser.add_argument("--max-files", type=int, default=50, help="Max files to test")
    parser.add_argument("--output", type=str, default="benchmark_dataset_report.txt", help="Output txt report path")
    args = parser.parse_args()

    print("Loading models...")
    engine_aasist = AASISTInferenceEngine(settings)
    engine_aasist.load()
    
    engine_w2v2 = Wav2Vec2InferenceEngine()
    engine_w2v2.load()
    
    scorer = EnsembleScorer(aasist_weight=0.3, wav2vec2_weight=0.7)
    classifier = RiskClassifier(
        threshold_low=settings.risk_threshold_low,
        threshold_high=settings.risk_threshold_high,
    )

    print("\nLoading dataset 'UniDataPro/real-vs-fake-human-voice-deepfake-audio'...")
    try:
        from datasets import Audio
        ds = load_dataset("UniDataPro/real-vs-fake-human-voice-deepfake-audio", split="train")
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    max_files = min(args.max_files, len(ds))
    print(f"Running inference on {max_files} samples...")
    
    results = defaultdict(list)
    summary = defaultdict(lambda: {"total": 0, "correct": 0})
    
    import soundfile as sf
    import io
    
    for i in range(max_files):
        item = ds[i]
        
        audio_data = item.get("audio", {})
        audio_path = audio_data.get("path")
        audio_bytes = audio_data.get("bytes")
        
        try:
            if audio_bytes is not None:
                audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            elif audio_path is not None:
                audio_array, sample_rate = sf.read(audio_path, dtype="float32")
            else:
                continue
        except Exception as e:
            print(f"Skip Entry {i} (decode error): {e}")
            continue
            
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
            
        # Parse label
        raw_label = item.get("label", item.get("class", 0))
        # Usually 0=Real/Bonafide, 1=Fake/Spoof
        is_fake = bool(raw_label == 1 or str(raw_label).lower() in ["fake", "spoof"])
        label_str = "Fake (Synthetic)" if is_fake else "Real (Human)"
        
        # Inference
        res_aasist = engine_aasist.infer(audio_array, sample_rate=sample_rate)
        res_w2v2 = engine_w2v2.infer(audio_array, sample_rate=sample_rate)
        
        ensemble_res = scorer.score(res_aasist, res_w2v2)
        score = ensemble_res["deepfake_score"]
        
        risk = classifier.classify(score, confidence=ensemble_res["confidence"])
        
        # Is the model correct? 
        # If it's fake, we want HIGH risk. If it's real, we want LOW (or at least not HIGH).
        if is_fake:
            correct = risk.risk_level.value == "HIGH"
        else:
            correct = risk.risk_level.value == "LOW"
            
        # Logging
        results[label_str].append({
            "id": i,
            "risk": risk.risk_level.value,
            "score": score,
            "correct": correct
        })
        
        summary[label_str]["total"] += 1
        if correct:
            summary[label_str]["correct"] += 1
            
        print(f"Processed {i+1}/{max_files} | True: {label_str} | Pred Risk: {risk.risk_level.value}")

    print("\nGenerating report...")
    report_text = generate_tree_view(results, summary)
    
    # Save to file
    out_path = Path(args.output)
    out_path.write_text(report_text, encoding="utf-8")
    
    print(f"Success! Report saved to {out_path.absolute()}")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
