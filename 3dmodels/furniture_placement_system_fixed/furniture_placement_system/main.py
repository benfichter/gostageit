#!/usr/bin/env python
"""
Furniture Placement System - Main Entry Point
Places furniture in empty rooms using physics simulation and design rules
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from run_placement_pipeline import run_placement_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Automatically place furniture in empty room images using physics and design rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --image room.jpg
  
  # Specify ceiling height
  python main.py --image room.jpg --ceiling-height 2.7
  
  # Use 3D rendering (requires pyrender)
  python main.py --image room.jpg --use-3d
  
  # Custom output directory
  python main.py --image room.jpg --output-dir results/my_room
        """
    )
    
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to empty room image (jpg, png)"
    )
    
    parser.add_argument(
        "--ceiling-height",
        type=float,
        default=2.4,
        help="Room ceiling height in meters (default: 2.4)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: auto-generated timestamp)"
    )
    
    parser.add_argument(
        "--use-3d",
        action="store_true",
        help="Enable 3D rendering (requires pyrender package)"
    )
    
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        help="Style hints for furniture selection (future feature)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device for computation (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.image.exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Set device
    import torch
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("FURNITURE PLACEMENT SYSTEM")
    print("="*60)
    print(f"Input image: {args.image}")
    print(f"Ceiling height: {args.ceiling_height}m")
    print(f"Device: {device}")
    print(f"3D rendering: {'Enabled' if args.use_3d else 'Disabled'}")
    print("="*60)
    
    try:
        result = run_placement_pipeline(
            image_path=args.image,
            ceiling_height=args.ceiling_height,
            output_dir=args.output_dir,
            style_hints=args.style,
            device=device,
            use_3d_rendering=args.use_3d
        )
        
        print("\n" + "="*60)
        print("SUCCESS! Results saved to:")
        print(f"  {result['output_dir']}")
        print("\nGenerated files:")
        print(f"  - Visualization: {Path(result['rendered_path']).name}")
        print(f"  - Top-down view: {Path(result['topdown_path']).name}")
        print(f"  - Dimensions: {Path(result['annotated_path']).name}")
        print(f"  - Placement data: {Path(result['placement_data_path']).name}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
