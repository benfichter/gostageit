#!/usr/bin/env python
"""
Test script to verify the furniture placement system is working
"""

import sys
from pathlib import Path
import numpy as np
import cv2


def create_test_room_image():
    """Create a simple test room image"""
    # Create a 800x600 image
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw floor (bottom portion)
    cv2.rectangle(img, (0, 400), (800, 600), (200, 180, 160), -1)
    
    # Draw walls
    cv2.rectangle(img, (0, 0), (800, 400), (240, 240, 230), -1)
    
    # Draw back wall edge
    cv2.line(img, (0, 400), (800, 400), (150, 150, 150), 2)
    
    # Draw some perspective lines for corners
    cv2.line(img, (0, 0), (100, 400), (180, 180, 180), 1)
    cv2.line(img, (800, 0), (700, 400), (180, 180, 180), 1)
    
    # Save test image
    test_path = Path("test_room.jpg")
    cv2.imwrite(str(test_path), img)
    
    return test_path


def main():
    print("="*60)
    print("FURNITURE PLACEMENT SYSTEM - TEST")
    print("="*60)
    
    # Check imports
    print("\nChecking imports...")
    try:
        import torch
        print("✓ PyTorch imported")
    except ImportError:
        print("✗ PyTorch not found - install with: pip install torch")
        sys.exit(1)
    
    try:
        import pybullet
        print("✓ PyBullet imported")
    except ImportError:
        print("✗ PyBullet not found - install with: pip install pybullet")
        sys.exit(1)
    
    try:
        import trimesh
        print("✓ Trimesh imported")
    except ImportError:
        print("✗ Trimesh not found - install with: pip install trimesh")
        sys.exit(1)
    
    try:
        import cv2
        print("✓ OpenCV imported")
    except ImportError:
        print("✗ OpenCV not found - install with: pip install opencv-python")
        sys.exit(1)
    
    # Check optional imports
    try:
        import pyrender
        print("✓ PyRender imported (3D rendering available)")
        has_3d = True
    except ImportError:
        print("○ PyRender not found (3D rendering disabled)")
        has_3d = False
    
    # Check for MoGe
    try:
        from moge.model.v2 import MoGeModel
        print("✓ MoGe model available")
    except ImportError:
        print("✗ MoGe not found")
        print("  Install with: pip install git+https://github.com/microsoft/MoGe.git")
        sys.exit(1)
    
    # Create test image
    print("\nCreating test room image...")
    test_image = create_test_room_image()
    print(f"✓ Test image created: {test_image}")
    
    # Test the pipeline
    print("\nTesting placement pipeline...")
    try:
        from run_placement_pipeline import run_placement_pipeline
        
        print("Running quick test (this may take a minute on first run)...")
        result = run_placement_pipeline(
            image_path=test_image,
            ceiling_height=2.4,
            use_3d_rendering=False  # Use 2D for test
        )
        
        print("✓ Pipeline executed successfully!")
        print(f"  Output directory: {result['output_dir']}")
        print(f"  Placements: {len(result['placements'])} items")
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*60)
    print("TEST COMPLETE - System is ready to use!")
    print("="*60)
    print("\nRun with your own image:")
    print("  python main.py --image your_room.jpg")


if __name__ == "__main__":
    main()
