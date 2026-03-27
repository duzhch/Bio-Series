#!/usr/bin/env python3
"""
Test GPU functionality and device selection
"""
import sys
from pathlib import Path

# Add current directory to Python path for module imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_gpu_functionality():
    """Test GPU device selection and model loading"""
    print("=== Testing GPU Functionality ===")
    
    try:
        from src.features import get_best_device
        
        # Test device selection
        print("Testing device selection...")
        device = get_best_device('auto')
        print(f"Selected device: {device}")
        
        # Test with CPU fallback
        cpu_device = get_best_device('cpu')
        print(f"CPU device: {cpu_device}")
        
        return True
        
    except Exception as e:
        print(f"GPU functionality test failed: {e}")
        return False

def test_delta_engine():
    """Test DeltaEngine initialization"""
    print("\n=== Testing DeltaEngine Initialization ===")
    
    try:
        from DF_GSF_v5 import load_cfg
        from src.features import _DeltaEngine
        
        cfg = load_cfg('config/global_config.yaml')
        
        print("Testing DeltaEngine with CPU...")
        engine = _DeltaEngine(
            cfg['resources']['reference_genome'],
            cfg['resources']['pigbert_model'],
            device='cpu'
        )
        print("✓ DeltaEngine initialized successfully on CPU")
        
        return True
        
    except Exception as e:
        print(f"✗ DeltaEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success1 = test_gpu_functionality()
    success2 = test_delta_engine()
    
    if success1 and success2:
        print("\n✅ All GPU tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)