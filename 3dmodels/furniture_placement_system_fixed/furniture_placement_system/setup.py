#!/usr/bin/env python
"""
Setup script for the Furniture Placement System
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("="*60)
    print("FURNITURE PLACEMENT SYSTEM - SETUP")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Install basic requirements
    print("\nInstalling core dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "numpy",
        "opencv-python",
        "torch",
        "trimesh",
        "pybullet",
        "scipy",
        "Pillow",
        "python-dotenv"
    ])
    
    # Check for optional packages
    print("\n" + "="*60)
    print("OPTIONAL PACKAGES")
    print("="*60)
    
    response = input("\nInstall 3D rendering support (pyrender)? [y/N]: ").lower()
    if response == 'y':
        print("Installing pyrender...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyrender", "pyglet"])
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print("\nTo run the system:")
    print("  python main.py --image your_room.jpg")
    print("\nFor help:")
    print("  python main.py --help")


if __name__ == "__main__":
    main()
