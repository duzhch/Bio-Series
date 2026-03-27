#!/usr/bin/env python3
"""
Test GPU functionality and DeltaEngine initialization.
"""
import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def parse_args():
    ap = argparse.ArgumentParser(description='Smoke-test local Bio-Series runtime configuration')
    ap.add_argument('--config', default='config/global_config.yaml', help='Path to YAML config file')
    return ap.parse_args()


def test_gpu_functionality():
    print("=== Testing GPU Functionality ===")
    try:
        from src.features import get_best_device

        print("Testing device selection...")
        device = get_best_device('auto')
        print(f"Selected device: {device}")
        print(f"CPU device: {get_best_device('cpu')}")
        return True
    except Exception as e:
        print(f"GPU functionality test failed: {e}")
        return False


def test_delta_engine(config_path: str):
    print("\n=== Testing DeltaEngine Initialization ===")
    try:
        from DF_GSF_v5 import load_cfg
        from src.features import _DeltaEngine

        cfg = load_cfg(config_path)
        ref_genome = Path(cfg['resources']['reference_genome'])
        model_dir = Path(cfg['resources']['pigbert_model'])

        if not ref_genome.exists():
            print(f"Missing reference genome: {ref_genome}")
            return False
        if not model_dir.exists():
            print(f"Missing PigBERT model directory: {model_dir}")
            return False

        print("Testing DeltaEngine with CPU...")
        _DeltaEngine(str(ref_genome), str(model_dir), device='cpu')
        print("DeltaEngine initialized successfully on CPU")
        return True
    except Exception as e:
        print(f"DeltaEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    args = parse_args()
    success1 = test_gpu_functionality()
    success2 = test_delta_engine(args.config)

    if success1 and success2:
        print("\nAll GPU tests passed")
        sys.exit(0)

    print("\nSome tests failed")
    sys.exit(1)
