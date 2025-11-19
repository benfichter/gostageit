"""
Main pipeline for furniture placement using MoGe room analysis
Standalone version without external dependencies
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Import MoGe model
try:
    from moge.model.v2 import MoGeModel
except ImportError:
    try:
        from moge.model.v1 import MoGeModel
        print("Warning: Using MoGe v1, consider upgrading to v2")
    except ImportError:
        print("ERROR: MoGe not installed. Please run:")
        print("pip install git+https://github.com/microsoft/MoGe.git")
        exit(1)

# Import placement modules
from furniture_placement_engine import (
    RuleBasedFurniturePlacement,
    FurnitureItem,
    InteriorDesignRules
)
from furniture_renderer import (
    FurnitureRenderer,
    SimpleRenderer
)


def log(message: str) -> None:
    """Consistent console logging"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[PLACEMENT {timestamp}] {message}")


def run_placement_pipeline(
    image_path: Path,
    ceiling_height: float = 2.4,
    output_dir: Optional[Path] = None,
    style_hints: Optional[str] = None,
    model: Optional[MoGeModel] = None,
    device: Optional[torch.device] = None,
    use_3d_rendering: bool = False,
    skip_detection: bool = True
) -> Dict:
    """
    Run the complete furniture placement pipeline
    
    Args:
        image_path: Path to empty room image
        ceiling_height: True ceiling height in meters for calibration
        output_dir: Output directory for results
        style_hints: Optional style description (for future AI integration)
        model: Optional pre-loaded MoGe model
        device: Device for computation
        use_3d_rendering: Whether to use full 3D rendering (requires pyrender)
        skip_detection: Skip existing furniture detection (assume empty room)
    
    Returns:
        Dictionary with paths to output files and placement data
    """
    
    # Setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("outputs") / f"placement_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load MoGe model if not provided
    if model is None:
        log(f"Loading MoGe model on {device}...")
        checkpoint_candidates = [
            "microsoft/moge-2-vitl-normal",
            "Ruicheng/moge-2-vitl-normal",
            "microsoft/moge-vitl",
            "Ruicheng/moge-vitl",
        ]
        last_error = None
        for checkpoint in checkpoint_candidates:
            try:
                log(f"  Trying checkpoint: {checkpoint}")
                model = MoGeModel.from_pretrained(checkpoint).to(device)
                log(f"MoGe weights loaded from {checkpoint}")
                break
            except Exception as exc:
                last_error = exc
                log(f"  Failed to load {checkpoint}: {exc}")
        else:
            raise RuntimeError(
                "Could not load any MoGe model checkpoint. "
                "Verify internet access or specify a local checkpoint."
            ) from last_error
        model.eval()
        log("MoGe model loaded")
    
    # Step 1: Analyze room
    log("Analyzing room geometry with MoGe...")
    room_analysis = run_moge_analysis(
        model=model,
        image_path=image_path,
        device=device,
        true_height=ceiling_height
    )
    
    # Save room analysis
    room_data_path = output_dir / "room_analysis.json"
    with open(room_data_path, 'w') as f:
        json.dump({
            'dimensions': room_analysis['dimensions'],
            'floor_bounds': room_analysis['floor_bounds'],
            'calibration_factor': room_analysis['calibration_factor'],
            'room_type': room_analysis['room_type']
        }, f, indent=2)
    log(f"Room analysis saved to {room_data_path}")
    
    # Step 2: For now, assume empty room (no existing furniture detection)
    log("Assuming empty room (no existing furniture)")
    existing_furniture = []
    
    # Step 3: Determine furniture to place
    log("Determining furniture set based on room analysis...")
    placement_engine = RuleBasedFurniturePlacement(log_fn=log)
    furniture_to_place = placement_engine.determine_furniture_set(room_analysis)
    
    log(f"Selected {len(furniture_to_place)} furniture items to place:")
    for item in furniture_to_place:
        dims = item.dimensions
        log(f"  - {item.type}: {dims['width']:.2f}m × {dims['depth']:.2f}m × {dims['height']:.2f}m")
    
    # Step 4: Optimize placement
    log("Computing optimal furniture placement...")
    placements = placement_engine.optimize_placement(furniture_to_place, room_analysis)
    
    log(f"Successfully placed {len(placements)} items")
    
    # Save placement data
    placement_data_path = output_dir / "placements.json"
    with open(placement_data_path, 'w') as f:
        json.dump(placements, f, indent=2, default=str)
    log(f"Placement data saved to {placement_data_path}")
    
    # Step 5: Render visualization
    log("Rendering furniture in room...")
    
    # Always create top-down view
    renderer = FurnitureRenderer()
    topdown_path = output_dir / "placement_topdown.png"
    topdown_image = renderer.render_placement_visualization(
        room_image=room_analysis["image_rgb"],
        placements=placements,
        room_analysis=room_analysis,
        output_path=topdown_path
    )
    log(f"Top-down visualization saved to {topdown_path}")
    
    # Create 2D or 3D visualization
    if use_3d_rendering:
        try:
            import pyrender
            rendered_path = output_dir / "rendered_room_3d.png"
            rendered_image = renderer.render_furniture_in_room(
                room_image=room_analysis["image_rgb"],
                placements=placements,
                room_analysis=room_analysis,
                output_path=rendered_path
            )
            log(f"3D rendered room saved to {rendered_path}")
        except ImportError as e:
            log(f"3D rendering unavailable ({e}), using 2D visualization")
            use_3d_rendering = False
    
    if not use_3d_rendering:
        # Use simple 2D visualization
        rendered_path = output_dir / "placement_2d.png"
        simple_vis = SimpleRenderer.render_2d_visualization(
            room_image=room_analysis["image_rgb"],
            placements=placements,
            room_analysis=room_analysis,
            output_path=rendered_path
        )
        log(f"2D visualization saved to {rendered_path}")
    
    # Step 6: Create annotated dimension overlay
    log("Creating dimension annotations...")
    annotated_path = output_dir / "annotated_dimensions.png"
    create_dimension_overlay(
        room_analysis["image_rgb"],
        placements,
        room_analysis,
        annotated_path
    )
    log(f"Annotated dimensions saved to {annotated_path}")
    
    # Create summary report
    summary_path = output_dir / "summary.txt"
    create_summary_report(room_analysis, placements, summary_path)
    log(f"Summary report saved to {summary_path}")
    
    # Summary
    log("\n" + "="*60)
    log("PLACEMENT COMPLETE")
    log("="*60)
    log(f"Room type: {room_analysis['room_type']}")
    log(f"Room dimensions: {room_analysis['dimensions']['width']:.2f}m × {room_analysis['dimensions']['depth']:.2f}m")
    log(f"Room area: {room_analysis['dimensions']['area']:.2f}m²")
    log(f"Furniture placed: {len(placements)} items")
    log(f"Output directory: {output_dir}")
    
    return {
        'output_dir': str(output_dir),
        'room_analysis': room_analysis,
        'placements': placements,
        'rendered_path': str(rendered_path),
        'topdown_path': str(topdown_path),
        'annotated_path': str(annotated_path),
        'placement_data_path': str(placement_data_path),
        'summary_path': str(summary_path)
    }


def run_moge_analysis(model: MoGeModel, image_path: Path, device: torch.device, 
                     true_height: float) -> Dict:
    """
    Run MoGe analysis on image to extract 3D room geometry
    """
    # Read image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    input_tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    
    # Run inference
    with torch.inference_mode():
        output = model.infer(input_tensor)
    
    # Extract outputs
    points = output["points"].cpu().numpy()
    depth = output["depth"].cpu().numpy()
    mask = output["mask"].cpu().numpy() > 0
    normals = output.get("normal")
    if normals is not None:
        normals = normals.cpu().numpy()
    
    intrinsics = output.get("intrinsics")
    if intrinsics is not None:
        intrinsics = intrinsics.cpu().numpy()
    
    # Compute floor and ceiling masks using normals or heuristics
    floor_mask, ceiling_mask = compute_floor_ceiling_masks(points, mask, normals)
    
    # Calculate room height and calibration
    valid_points = points[mask]
    if len(valid_points) == 0:
        raise RuntimeError("MoGe inference produced no valid points")
    
    # Get floor and ceiling Y coordinates
    floor_y, ceiling_y = estimate_room_height(points, floor_mask, ceiling_mask, valid_points)
    
    estimated_height = abs(floor_y - ceiling_y)
    if estimated_height < 1e-6:
        estimated_height = true_height
    
    calibration_factor = true_height / estimated_height
    points_calibrated = points * calibration_factor
    
    # Calculate room dimensions
    dimensions, floor_bounds = calculate_room_dimensions(
        points_calibrated, floor_mask, mask, true_height
    )
    
    # Determine room type
    room_type = infer_room_type(dimensions)
    
    return {
        'image_rgb': image_rgb,
        'points_calibrated': points_calibrated,
        'depth': depth,
        'mask': mask,
        'normals': normals,
        'intrinsics': intrinsics,
        'floor_mask': floor_mask,
        'ceiling_mask': ceiling_mask,
        'dimensions': dimensions,
        'floor_bounds': floor_bounds,
        'calibration_factor': calibration_factor,
        'estimated_height': estimated_height,
        'true_height': true_height,
        'room_type': room_type
    }


def compute_floor_ceiling_masks(points: np.ndarray, mask: np.ndarray, 
                               normals: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute floor and ceiling masks from normals or point distribution"""
    if normals is not None:
        # Use normal vectors to identify horizontal surfaces
        normal_y = normals[:, :, 1]
        floor_mask = (normal_y < -0.7) & mask  # Normals pointing up
        ceiling_mask = (normal_y > 0.7) & mask  # Normals pointing down
    else:
        # Fallback: use Y-coordinate distribution
        y_coords = points[:, :, 1]
        valid_y = y_coords[mask]
        if len(valid_y) > 0:
            floor_threshold = np.percentile(valid_y, 90)  # Bottom 10%
            ceiling_threshold = np.percentile(valid_y, 10)  # Top 10%
            floor_mask = mask & (y_coords >= floor_threshold)
            ceiling_mask = mask & (y_coords <= ceiling_threshold)
        else:
            floor_mask = np.zeros_like(mask)
            ceiling_mask = np.zeros_like(mask)
    
    return floor_mask, ceiling_mask


def estimate_room_height(points: np.ndarray, floor_mask: np.ndarray, 
                        ceiling_mask: np.ndarray, valid_points: np.ndarray) -> Tuple[float, float]:
    """Estimate floor and ceiling Y coordinates"""
    if np.any(floor_mask):
        floor_y = float(np.mean(points[floor_mask][:, 1]))
    else:
        floor_y = float(np.percentile(valid_points[:, 1], 95))
    
    if np.any(ceiling_mask):
        ceiling_y = float(np.mean(points[ceiling_mask][:, 1]))
    else:
        ceiling_y = float(np.percentile(valid_points[:, 1], 5))
    
    return floor_y, ceiling_y


def calculate_room_dimensions(points_calibrated: np.ndarray, floor_mask: np.ndarray,
                             mask: np.ndarray, true_height: float) -> Tuple[Dict, Dict]:
    """Calculate room dimensions and floor boundaries"""
    floor_points = points_calibrated[floor_mask] if np.any(floor_mask) else points_calibrated[mask]
    
    if len(floor_points) == 0:
        raise ValueError("No floor points detected")
    
    x_coords = floor_points[:, 0]
    z_coords = floor_points[:, 2]
    y_coords = floor_points[:, 1]
    
    dimensions = {
        'width': float(np.max(x_coords) - np.min(x_coords)),
        'depth': float(np.max(z_coords) - np.min(z_coords)),
        'height': float(true_height),
        'area': float((np.max(x_coords) - np.min(x_coords)) * (np.max(z_coords) - np.min(z_coords)))
    }
    
    floor_bounds = {
        'x_min': float(np.min(x_coords)),
        'x_max': float(np.max(x_coords)),
        'z_min': float(np.min(z_coords)),
        'z_max': float(np.max(z_coords)),
        'y_floor': float(np.mean(y_coords))
    }
    
    return dimensions, floor_bounds


def infer_room_type(dimensions: Dict[str, float]) -> str:
    """Infer room type from dimensions"""
    area = dimensions['area']
    width = dimensions['width']
    depth = dimensions['depth']
    aspect_ratio = max(width, depth) / min(width, depth) if min(width, depth) > 0 else 1
    
    if area < 10:
        return 'bedroom'
    elif aspect_ratio > 2.0:
        return 'hallway'
    elif area < 15:
        return 'bedroom' if area < 12 else 'dining_room'
    else:
        return 'living_room'


def create_dimension_overlay(image_rgb: np.ndarray, placements: List[Dict],
                            room_analysis: Dict, output_path: Path):
    """Create an overlay showing furniture dimensions with better positioning"""
    overlay = image_rgb.copy()
    h, w = overlay.shape[:2]
    
    # Create semi-transparent background for text
    info_box_height = min(200, h // 4)
    info_box = np.ones((info_box_height, w, 3), dtype=np.uint8) * 255
    info_box = cv2.addWeighted(info_box, 0.7, info_box, 0, 0)
    
    # Add room info at top
    y_offset = 30
    room_text = f"Room: {room_analysis['room_type'].replace('_', ' ').title()} | "
    room_text += f"Size: {room_analysis['dimensions']['width']:.1f}m × {room_analysis['dimensions']['depth']:.1f}m | "
    room_text += f"Area: {room_analysis['dimensions']['area']:.1f}m²"
    
    cv2.putText(overlay, room_text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, room_text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Add furniture list
    y_offset = 60
    cv2.putText(overlay, "Placed Furniture:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(overlay, "Placed Furniture:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # List each furniture item
    for i, placement in enumerate(placements):
        y_offset += 25
        dims = placement['dimensions']
        text = f"• {placement['type'].replace('_', ' ').title()}: "
        text += f"{dims['width']:.1f}m × {dims['depth']:.1f}m × {dims['height']:.1f}m"
        
        # Add position info
        pos = placement['position']
        text += f" @ ({pos[0]:.1f}, {pos[2]:.1f})"
        
        cv2.putText(overlay, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def create_summary_report(room_analysis: Dict, placements: List[Dict], output_path: Path):
    """Create a text summary of the placement results"""
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FURNITURE PLACEMENT SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("ROOM ANALYSIS:\n")
        f.write("-"*30 + "\n")
        f.write(f"Type: {room_analysis['room_type'].replace('_', ' ').title()}\n")
        f.write(f"Dimensions: {room_analysis['dimensions']['width']:.2f}m × {room_analysis['dimensions']['depth']:.2f}m\n")
        f.write(f"Height: {room_analysis['dimensions']['height']:.2f}m\n")
        f.write(f"Floor Area: {room_analysis['dimensions']['area']:.2f}m²\n")
        f.write(f"Volume: {room_analysis['dimensions']['area'] * room_analysis['dimensions']['height']:.2f}m³\n")
        f.write(f"Calibration Factor: {room_analysis['calibration_factor']:.4f}\n")
        f.write("\n")
        
        f.write("FURNITURE PLACEMENT:\n")
        f.write("-"*30 + "\n")
        f.write(f"Total Items Placed: {len(placements)}\n\n")
        
        for i, placement in enumerate(placements, 1):
            f.write(f"{i}. {placement['type'].replace('_', ' ').title()}\n")
            dims = placement['dimensions']
            f.write(f"   Dimensions: {dims['width']:.2f}m W × {dims['depth']:.2f}m D × {dims['height']:.2f}m H\n")
            
            # Convert to feet/inches for US users
            w_ft = dims['width'] * 3.28084
            d_ft = dims['depth'] * 3.28084
            h_ft = dims['height'] * 3.28084
            f.write(f"   ({w_ft:.1f}' × {d_ft:.1f}' × {h_ft:.1f}')\n")
            
            pos = placement['position']
            f.write(f"   Position: X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m\n")
            f.write(f"   Rotation: {np.degrees(placement['rotation']):.1f}°\n")
            f.write(f"   Placement Score: {placement.get('score', 0):.2f}\n")
            f.write("\n")
        
        # Calculate coverage
        total_furniture_area = sum(p['dimensions']['width'] * p['dimensions']['depth'] 
                                  for p in placements)
        coverage = (total_furniture_area / room_analysis['dimensions']['area']) * 100
        
        f.write("STATISTICS:\n")
        f.write("-"*30 + "\n")
        f.write(f"Floor Coverage: {coverage:.1f}%\n")
        f.write(f"Free Floor Space: {room_analysis['dimensions']['area'] - total_furniture_area:.2f}m²\n")
        
        # Add placement quality metrics
        avg_score = np.mean([p.get('score', 0) for p in placements])
        f.write(f"Average Placement Score: {avg_score:.2f}\n")


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--ceiling-height", type=float, default=2.4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--use-3d", action="store_true")
    args = parser.parse_args()
    
    run_placement_pipeline(
        image_path=args.image,
        ceiling_height=args.ceiling_height,
        output_dir=args.output_dir,
        use_3d_rendering=args.use_3d
    )
