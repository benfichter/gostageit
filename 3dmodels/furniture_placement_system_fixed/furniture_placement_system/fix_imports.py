"""
FIX FOR MODULE IMPORT ERROR
Run this to fix the import issues
"""

import os
import shutil

print("Fixing import errors in furniture placement system...")

# Rename the fixed file
if os.path.exists("run_placement_pipeline_fixed.py"):
    if os.path.exists("run_placement_pipeline.py"):
        shutil.copy("run_placement_pipeline.py", "run_placement_pipeline_original.py")
        print("✓ Backed up original to run_placement_pipeline_original.py")
    
    shutil.copy("run_placement_pipeline_fixed.py", "run_placement_pipeline.py")
    print("✓ Replaced with fixed version")
else:
    print("✗ run_placement_pipeline_fixed.py not found")

print("\nDone! Try running again:")
print("  python main.py --image your_room.jpg")
