#!/usr/bin/env python3
"""
Test script to verify the furniture placement system installation
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    required_modules = [
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('trimesh', 'Trimesh'),
        ('pybullet', 'PyBullet'),
        ('PIL', 'Pillow'),
        ('scipy', 'SciPy'),
        ('sklearn', 'Scikit-learn')
    ]
    
    optional_modules = [
        ('pyrender', 'PyRender (for 3D visualization)'),
        ('segment_anything', 'SAM (for furniture detection)')
    ]
    
    print("Testing required dependencies...")
    print("="*50)
    
    all_good = True
    for module_name, display_name in required_modules:
        try:
            importlib.import_module(module_name)
            print(f"✓ {display_name}: OK")
        except ImportError as e:
            print(f"✗ {display_name}: MISSING")
            all_good = False
    
    print("\nTesting optional dependencies...")
    print("="*50)
    
    for module_name, display_name in optional_modules:
        try:
            importlib.import_module(module_name)
            print(f"✓ {display_name}: OK")
        except ImportError:
            print(f"○ {display_name}: Not installed (optional)")
    
    print("\nTesting MoGe model...")
    print("="*50)
    try:
        from moge.model.v2 import MoGeModel
        print("✓ MoGe v2: OK")
    except ImportError:
        try:
            from moge.model.v1 import MoGeModel
            print("✓ MoGe v1: OK (consider upgrading to v2)")
        except ImportError:
            print("✗ MoGe: MISSING - Install with: pip install git+https://github.com/Ruicheng/moge.git")
            all_good = False
    
    print("\nTesting local modules...")
    print("="*50)
    
    local_modules = [
        'furniture_placement_engine',
        'furniture_renderer',
        'run_placement_pipeline'
    ]
    
    for module_name in local_modules:
        try:
            importlib.import_module(module_name)
            print(f"✓ {module_name}: OK")
        except ImportError as e:
            print(f"✗ {module_name}: ERROR - {e}")
            all_good = False
    
    return all_good


def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU availability...")
    print("="*50)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("○ No CUDA GPU available - will use CPU (slower)")
    except Exception as e:
        print(f"○ Could not check GPU: {e}")


def create_test_image():
    """Create a simple test image if none exists"""
    test_image_path = Path("test_room.jpg")
    
    if not test_image_path.exists():
        print("\nCreating test image...")
        print("="*50)
        
        try:
            import numpy as np
            import cv2
            
            # Create a simple room-like image
            img = np.ones((480, 640, 3), dtype=np.uint8) * 240  # Light gray
            
            # Add floor (darker)
            img[300:, :] = [180, 180, 180]
            
            # Add back wall (medium)
            img[:300, :] = [220, 220, 220]
            
            # Add some depth cues (corners)
            pts = np.array([[0, 300], [0, 480], [100, 400], [100, 280]], np.int32)
            cv2.fillPoly(img, [pts], (200, 200, 200))
            
            pts = np.array([[640, 300], [640, 480], [540, 400], [540, 280]], np.int32)
            cv2.fillPoly(img, [pts], (200, 200, 200))
            
            cv2.imwrite(str(test_image_path), img)
            print(f"✓ Created test image: {test_image_path}")
            
        except Exception as e:
            print(f"✗ Could not create test image: {e}")
            return None
    else:
        print(f"\n✓ Test image exists: {test_image_path}")
    
    return test_image_path


def main():
    print("\n" + "="*60)
    print("FURNITURE PLACEMENT SYSTEM - INSTALLATION TEST")
    print("="*60 + "\n")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test GPU
    test_gpu()
    
    # Create test image
    test_image = create_test_image()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if imports_ok:
        print("✓ All required dependencies installed")
        print("\nYou can now run:")
        if test_image:
            print(f"  python main.py --image {test_image}")
        else:
            print("  python main.py --image your_room.jpg")
    else:
        print("✗ Some dependencies missing")
        print("\nPlease run:")
        print("  pip install -r requirements.txt")
        print("  pip install git+https://github.com/Ruicheng/moge.git")
    
    print("\n")


if __name__ == "__main__":
    main()
