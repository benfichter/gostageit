"""
Main pipeline for furniture placement using existing MoGe analysis
Integrates with your existing code while replacing BFL staging with placement
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import dotenv
from moge.model.v2 import MoGeModel

# Import your existing modules
from furniture_detection import FurnitureDetector
from sam_subject_extractor import SamSubjectExtractor

# Import new placement modules
from furniture_placement_engine import (
    RuleBasedFurniturePlacement,
    FurnitureItem,
    InteriorDesignRules
)
from furniture_renderer import (
    FurnitureRenderer,
    SimpleRenderer
)

dotenv.load_dotenv()


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
    use_3d_rendering: bool = False
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
    
    Returns:
        Dictionary with paths to output files and placement data
    """
    
    # Setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("output") / f"placement_{timestamp}"
    
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
    
    # Step 1: Analyze room using your existing code
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
    
    # Step 2: Detect existing furniture (if any)
    log("Checking for existing furniture...")
    sam_extractor = get_sam_extractor(device)
    if sam_extractor:
        try:
            furniture_detector = FurnitureDetector(
                sam_extractor=sam_extractor,
                log_fn=log
            )
            existing_furniture = furniture_detector.detect(
                image_rgb=room_analysis["image_rgb"],
                points_calibrated=room_analysis["points_calibrated"],
                depth_mask=room_analysis["mask"],
                normals=room_analysis["normals"]
            )
            log(f"Detected {len(existing_furniture)} existing furniture items")
        except Exception as e:
            log(f"Furniture detection failed: {e}")
            existing_furniture = []
    else:
        log("SAM not configured, skipping existing furniture detection")
        existing_furniture = []
    
    # Step 3: Determine furniture to place
    log("Determining furniture set based on room analysis...")
    placement_engine = RuleBasedFurniturePlacement(log_fn=log)
    furniture_to_place = placement_engine.determine_furniture_set(room_analysis)
    
    log(f"Selected {len(furniture_to_place)} furniture items to place:")
    for item in furniture_to_place:
        log(f"  - {item.type}: {item.dimensions['width']:.2f}m x {item.dimensions['depth']:.2f}m x {item.dimensions['height']:.2f}m")
    
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
    
    if use_3d_rendering:
        try:
            renderer = FurnitureRenderer()
            
            # Render furniture in room
            rendered_path = output_dir / "rendered_room.png"
            rendered_image = renderer.render_furniture_in_room(
                room_image=room_analysis["image_rgb"],
                placements=placements,
                room_analysis=room_analysis,
                output_path=rendered_path
            )
            log(f"3D rendered room saved to {rendered_path}")
            
            # Create top-down view
            topdown_path = output_dir / "placement_topdown.png"
            topdown_image = renderer.render_placement_visualization(
                room_image=room_analysis["image_rgb"],
                placements=placements,
                room_analysis=room_analysis,
                output_path=topdown_path
            )
            log(f"Top-down visualization saved to {topdown_path}")
            
        except ImportError as e:
            log(f"3D rendering unavailable ({e}), falling back to 2D visualization")
            use_3d_rendering = False
    
    if not use_3d_rendering:
        # Use simple 2D visualization
        simple_vis_path = output_dir / "placement_2d.png"
        simple_vis = SimpleRenderer.render_2d_visualization(
            room_image=room_analysis["image_rgb"],
            placements=placements,
            room_analysis=room_analysis,
            output_path=simple_vis_path
        )
        log(f"2D visualization saved to {simple_vis_path}")
        
        # Create top-down view using existing renderer
        renderer = FurnitureRenderer()
        topdown_path = output_dir / "placement_topdown.png"
        topdown_image = renderer.render_placement_visualization(
            room_image=room_analysis["image_rgb"],
            placements=placements,
            room_analysis=room_analysis,
            output_path=topdown_path
        )
        log(f"Top-down visualization saved to {topdown_path}")
    
    # Step 6: Create annotated dimension overlay
    log("Creating dimension annotations...")
    annotated_path = output_dir / "annotated_dimensions.png"
    create_dimension_overlay(
        room_analysis["image_rgb"],
        placements,
        annotated_path
    )
    log(f"Annotated dimensions saved to {annotated_path}")
    
    # Summary
    log("\n" + "="*60)
    log("PLACEMENT COMPLETE")
    log("="*60)
    log(f"Room type: {room_analysis['room_type']}")
    log(f"Room dimensions: {room_analysis['dimensions']['width']:.2f}m x {room_analysis['dimensions']['depth']:.2f}m")
    log(f"Furniture placed: {len(placements)} items")
    log(f"Output directory: {output_dir}")
    
    return {
        'output_dir': str(output_dir),
        'room_analysis': room_analysis,
        'placements': placements,
        'rendered_path': str(rendered_path) if use_3d_rendering else str(simple_vis_path),
        'topdown_path': str(topdown_path),
        'annotated_path': str(annotated_path),
        'placement_data_path': str(placement_data_path)
    }


def run_moge_analysis(model: MoGeModel, image_path: Path, device: torch.device, 
                     true_height: float) -> Dict:
    """
    Run MoGe analysis on image (adapted from your existing code)
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
    normals = output["normal"].cpu().numpy() if "normal" in output else None
    intrinsics = output.get("intrinsics")
    if intrinsics is not None:
        intrinsics = intrinsics.cpu().numpy()
    
    # Compute floor and ceiling masks
    if normals is not None:
        normal_y = normals[:, :, 1]
        floor_mask = (normal_y < -0.7) & mask
        ceiling_mask = (normal_y > 0.7) & mask
    else:
        y_coords = points[:, :, 1]
        valid_y = y_coords[mask]
        if len(valid_y) > 0:
            floor_threshold = np.percentile(valid_y, 90)
            ceiling_threshold = np.percentile(valid_y, 10)
            floor_mask = mask & (y_coords >= floor_threshold)
            ceiling_mask = mask & (y_coords <= ceiling_threshold)
        else:
            floor_mask = np.zeros_like(mask)
            ceiling_mask = np.zeros_like(mask)
    
    # Calculate room height and calibration
    valid_points = points[mask]
    if len(valid_points) == 0:
        raise RuntimeError("MoGe inference produced no valid points")
    
    if np.any(floor_mask):
        floor_y = float(np.mean(points[floor_mask][:, 1]))
    else:
        floor_y = float(np.percentile(valid_points[:, 1], 95))
    
    if np.any(ceiling_mask):
        ceiling_y = float(np.mean(points[ceiling_mask][:, 1]))
    else:
        ceiling_y = float(np.percentile(valid_points[:, 1], 5))
    
    estimated_height = abs(floor_y - ceiling_y)
    if estimated_height < 1e-6:
        estimated_height = true_height
    
    calibration_factor = true_height / estimated_height
    points_calibrated = points * calibration_factor
    
    # Calculate room dimensions
    floor_points = points_calibrated[floor_mask] if np.any(floor_mask) else points_calibrated[mask]
    
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


def infer_room_type(dimensions: Dict[str, float]) -> str:
    """Infer room type from dimensions"""
    area = dimensions['area']
    width = dimensions['width']
    depth = dimensions['depth']
    aspect_ratio = max(width, depth) / min(width, depth)
    
    if area < 10:
        return 'bedroom'
    elif aspect_ratio > 2.0:
        return 'hallway'
    elif area < 15:
        return 'bedroom' if area < 12 else 'dining_room'
    else:
        return 'living_room'


def get_sam_extractor(device: torch.device) -> Optional[SamSubjectExtractor]:
    """Get SAM extractor if configured (from your existing code)"""
    import os
    checkpoint = os.getenv("SAM_CHECKPOINT_PATH")
    if not checkpoint:
        log("SAM_CHECKPOINT_PATH not set, skipping SAM initialization")
        return None
    
    model_type = os.getenv("SAM_MODEL_TYPE", "vit_h")
    try:
        return SamSubjectExtractor(
            checkpoint_path=Path(checkpoint),
            model_type=model_type,
            device=device,
            log_fn=log
        )
    except Exception as e:
        log(f"Failed to initialize SAM: {e}")
        return None


def create_dimension_overlay(image_rgb: np.ndarray, placements: List[Dict], output_path: Path):
    """Create an overlay showing furniture dimensions"""
    overlay = image_rgb.copy()
    
    # Simplified 2D projection for annotation
    for i, placement in enumerate(placements):
        # Use simple heuristic for 2D position (this would be improved with proper projection)
        x = int(image_rgb.shape[1] * (0.2 + i * 0.15))
        y = int(image_rgb.shape[0] * (0.3 + (i % 3) * 0.2))
        
        # Draw furniture type and dimensions
        text = f"{placement['type']}: {placement['dimensions']['width']:.1f}m x {placement['dimensions']['depth']:.1f}m"
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Place furniture in empty room using physics and design rules")
    parser.add_argument("--image", type=Path, required=True, help="Path to empty room image")
    parser.add_argument("--ceiling-height", type=float, default=2.4, help="Ceiling height in meters")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--use-3d", action="store_true", help="Use 3D rendering (requires pyrender)")
    parser.add_argument("--style", type=str, help="Style hints (for future use)")
    
    args = parser.parse_args()
    
    run_placement_pipeline(
        image_path=args.image,
        ceiling_height=args.ceiling_height,
        output_dir=args.output_dir,
        style_hints=args.style,
        use_3d_rendering=args.use_3d
    )


if __name__ == "__main__":
    main()
