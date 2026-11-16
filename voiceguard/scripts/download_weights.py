#!/usr/bin/env python3
"""
VoiceGuard — Download Pretrained Weights
==========================================
Downloads pretrained AASIST and optionally Wav2Vec2 model weights.

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --include-wav2vec2
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"
AASIST_REPO = "https://github.com/clovaai/aasist.git"


def download_aasist_weights():
    """Download AASIST pretrained weights from the official repository."""
    print("=" * 50)
    print("  Downloading AASIST Pretrained Weights")
    print("=" * 50)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if weights already exist
    aasist_path = WEIGHTS_DIR / "AASIST.pth"
    aasist_l_path = WEIGHTS_DIR / "AASIST-L.pth"

    if aasist_path.exists() and aasist_l_path.exists():
        print("✓ AASIST weights already downloaded.")
        print(f"  AASIST:   {aasist_path} ({aasist_path.stat().st_size / 1024:.1f} KB)")
        print(f"  AASIST-L: {aasist_l_path} ({aasist_l_path.stat().st_size / 1024:.1f} KB)")
        return

    # Clone the repo (sparse checkout for weights only)
    tmp_dir = PROJECT_ROOT / ".tmp_aasist"
    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        print(f"Cloning {AASIST_REPO} (sparse checkout)...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none",
             "--sparse", AASIST_REPO, str(tmp_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

        subprocess.run(
            ["git", "sparse-checkout", "set", "models/weights"],
            cwd=str(tmp_dir),
            check=True,
            capture_output=True,
            text=True,
        )

        # Copy weights
        src_weights = tmp_dir / "models" / "weights"
        if src_weights.exists():
            for weight_file in src_weights.glob("*.pth"):
                dest = WEIGHTS_DIR / weight_file.name
                shutil.copy2(weight_file, dest)
                print(f"✓ Downloaded: {weight_file.name} ({dest.stat().st_size / 1024:.1f} KB)")
        else:
            print("⚠ Weight files not found in the AASIST repository.")
            print("  The weights may need to be downloaded separately.")
            print("  Check: https://github.com/clovaai/aasist")

    except subprocess.CalledProcessError as e:
        print(f"⚠ Git clone failed: {e.stderr}")
        print("  Please manually download weights from:")
        print(f"  {AASIST_REPO}")
        print(f"  Place .pth files in: {WEIGHTS_DIR}")
    except FileNotFoundError:
        print("⚠ Git not found. Please install git or download weights manually.")
        print(f"  Place .pth files in: {WEIGHTS_DIR}")
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def download_wav2vec2_model(model_id: str = "Hemgg/Deepfake-audio-detection"):
    """Download and cache the Wav2Vec2 model from HuggingFace."""
    print("\n" + "=" * 50)
    print("  Downloading Wav2Vec2 Deepfake Detection Model")
    print("=" * 50)

    try:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        print(f"Downloading model: {model_id}")
        print("This may take a few minutes on first download...")

        # Download and cache
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModelForAudioClassification.from_pretrained(model_id)

        nb_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Wav2Vec2 model downloaded and cached")
        print(f"  Parameters: {nb_params:,}")
        print(f"  Labels: {model.config.id2label}")

    except ImportError:
        print("⚠ transformers library not installed.")
        print("  Install with: pip install transformers")
    except Exception as e:
        print(f"⚠ Download failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained weights for VoiceGuard"
    )
    parser.add_argument(
        "--include-wav2vec2",
        action="store_true",
        help="Also download the Wav2Vec2 ensemble model from HuggingFace",
    )
    parser.add_argument(
        "--wav2vec2-model-id",
        type=str,
        default="Hemgg/Deepfake-audio-detection",
        help="HuggingFace model ID for Wav2Vec2",
    )
    args = parser.parse_args()

    download_aasist_weights()

    if args.include_wav2vec2:
        download_wav2vec2_model(args.wav2vec2_model_id)

    print("\n" + "=" * 50)
    print("  Download Complete")
    print("=" * 50)
    print(f"\nWeight files location: {WEIGHTS_DIR}")
    print("\nTo start the server:")
    print("  uvicorn api.app:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
