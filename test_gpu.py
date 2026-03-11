#!/usr/bin/env python3
"""
Test GPU functionality and device selection.
"""
import sys
from pathlib import Path

# Add current directory to Python path for module imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def test_gpu_functionality():
    """Test GPU device selection."""
    print("=== Testing GPU Functionality ===")

    try:
        from src.features import get_best_device

        print("Testing device selection...")
        device = get_best_device('auto')
        print(f"Selected device: {device}")

        cpu_device = get_best_device('cpu')
        print(f"CPU device: {cpu_device}")
        return True

    except Exception as e:
        print(f"GPU functionality test failed: {e}")
        return False


def test_delta_engine():
    """Test DeltaEngine initialization if local resources are configured."""
    print("\n=== Testing DeltaEngine Initialization ===")

    try:
        from DF_GSF_v5 import load_cfg
        from src.features import _DeltaEngine

        cfg = load_cfg('config/global_config.yaml')
        ref_genome = cfg['resources']['reference_genome']
        model_dir = cfg['resources']['pigbert_model']

        if '/path/to/' in ref_genome or '/path/to/' in model_dir:
            print("Skipping DeltaEngine initialization: configure local resource paths first.")
            return True

        if not Path(ref_genome).exists():
            print(f"Skipping DeltaEngine initialization: missing reference genome: {ref_genome}")
            return True

        if not Path(model_dir).exists():
            print(f"Skipping DeltaEngine initialization: missing PigBERT model directory: {model_dir}")
            return True

        print("Testing DeltaEngine with CPU...")
        _DeltaEngine(ref_genome, model_dir, device='cpu')
        print("DeltaEngine initialized successfully on CPU")
        return True

    except Exception as e:
        print(f"DeltaEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success1 = test_gpu_functionality()
    success2 = test_delta_engine()

    if success1 and success2:
        print("\nAll GPU tests passed")
        sys.exit(0)

    print("\nSome tests failed")
    sys.exit(1)
