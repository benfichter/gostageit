from datetime import datetime
from pathlib import Path

import cv2
import torch
import dotenv
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2
from bfl_staging_pipeline import run_bfl_pipeline
# images of rooms will be uploaded and you need to find the dimensions of the room

dotenv.load_dotenv()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA for inference")
else:
    device = torch.device("cpu")
    print("CUDA not available, falling back to CPU inference")

# Load the model from huggingface hub (or load from local).
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)                             

# Ask for input file name
input_file = input("Enter the input image filename (e.g., test3.jpg): ").strip()
if not input_file:
    input_file = "test3.jpg"  # Default if user just presses enter
    print(f"Using default: {input_file}")

input_path = Path(input_file)
EXPORT_ROOT = Path("exports")
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
EXPORT_DIR = EXPORT_ROOT / f"{timestamp}_{input_path.stem}"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\nAll outputs will be stored in: {EXPORT_DIR.resolve()}")

def export_path(name: str) -> str:
    return str(EXPORT_DIR / name)

# Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
input_image = cv2.cvtColor(cv2.imread(input_file), cv2.COLOR_BGR2RGB)                       
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

# Infer 
output = model.infer(input_image)
# Save the output image

"""
`output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
The maps are in the same size as the input image. 
{
    "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
    "depth": (H, W),        # depth map
    "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
    "mask": (H, W),         # a binary mask for valid pixels. 
    "intrinsics": (3, 3),   # normalized camera intrinsics
}
"""

depth_path = export_path("output_depth.png")
cv2.imwrite(depth_path, (output["depth"].cpu().numpy() * 1000).astype("uint16"))  # Save depth in mm
# make it colorful as it is dark right now
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(output["depth"].cpu().numpy(), alpha=255/output["depth"].max().item()), cv2.COLORMAP_JET)
depth_colormap_path = export_path("output_depth_colormap.png")
cv2.imwrite(depth_colormap_path, depth_colormap)  # Save colorful depth


mask_path = export_path("output_mask.png")
cv2.imwrite(mask_path, (output["mask"].cpu().numpy() * 255).astype("uint8"))          # Save mask
if "normal" in output:
    normal_map = (output["normal"].cpu().numpy() + 1) / 2 * 255  # Convert normal from [-1, 1] to [0, 255]
    normal_map_path = export_path("output_normal.png")
    cv2.imwrite(normal_map_path, normal_map.astype("uint8"))

# find the height of the room using the straight lines in the room and find the distances of those lines, use edge outlining to find the floor line and the ceiling line and make a line between them to find the height

import numpy as np
def calculate_room_height(depth_map, mask, points_map, original_image, normal_map=None):
    # Use the 3D points to find floor and ceiling automatically
    # In OpenCV camera coordinates: x=right, y=down, z=forward
    # So y-coordinate represents vertical position (positive downward)

    # Get all valid 3D points
    valid_mask = mask > 0
    valid_points = points_map[valid_mask]

    if len(valid_points) == 0:
        return None  # No valid depth data

    # Use normal map if available for more accurate floor/ceiling detection
    if normal_map is not None:
        print("Using normal map for floor/ceiling detection...")

        # Normal map: Y component tells us surface orientation
        # Floor normals point UP (negative Y in camera coords, since Y+ is down)
        # Ceiling normals point DOWN (positive Y)
        normal_y = normal_map[:, :, 1]  # Y component of normals

        # Identify floor: normals pointing up (Y < -0.7, meaning strongly upward)
        floor_mask = (normal_y < -0.7) & valid_mask
        # Identify ceiling: normals pointing down (Y > 0.7, meaning strongly downward)
        ceiling_mask = (normal_y > 0.7) & valid_mask

        # Visualize the detected surfaces
        surface_vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        surface_vis[floor_mask] = [255, 0, 0]    # Red = floor
        surface_vis[ceiling_mask] = [0, 255, 0]  # Green = ceiling
        surface_vis[(~floor_mask) & (~ceiling_mask) & valid_mask] = [100, 100, 100]  # Gray = walls
        surface_path = export_path("surface_detection.png")
        cv2.imwrite(surface_path, surface_vis)
        print(f"Surface detection visualization saved as '{surface_path}'")
        print("(Red = floor, Green = ceiling, Gray = walls)")

        # Get Y coordinates from detected floor and ceiling
        if np.any(floor_mask):
            floor_points = points_map[floor_mask]
            floor_y_value = np.mean(floor_points[:, 1])
            print(f"Found {np.sum(floor_mask)} floor pixels")
        else:
            # Fallback to percentile method
            print("No floor detected by normals, using percentile method")
            y_coords = valid_points[:, 1]
            floor_y_value = np.percentile(y_coords, 95)

        if np.any(ceiling_mask):
            ceiling_points = points_map[ceiling_mask]
            ceiling_y_value = np.mean(ceiling_points[:, 1])
            print(f"Found {np.sum(ceiling_mask)} ceiling pixels")
        else:
            # Fallback to percentile method
            print("No ceiling detected by normals, using percentile method")
            y_coords = valid_points[:, 1]
            ceiling_y_value = np.percentile(y_coords, 5)
    else:
        # Fallback to old percentile method
        print("No normal map available, using percentile method...")
        y_coords = valid_points[:, 1]
        floor_y_value = np.percentile(y_coords, 95)
        ceiling_y_value = np.percentile(y_coords, 5)

    print(f"\nAutomatic detection:")
    print(f"Floor Y-coordinate: {floor_y_value:.3f} m")
    print(f"Ceiling Y-coordinate: {ceiling_y_value:.3f} m")
    print(f"Estimated room height: {abs(floor_y_value - ceiling_y_value):.2f} m")

    # Now let user manually select if they want to override
    # Convert to uint8 format (0-255) if needed
    if original_image.dtype == np.float32 or original_image.dtype == np.float64:
        image_uint8 = (original_image * 255).astype(np.uint8)
    else:
        image_uint8 = original_image

    # Create a visualization showing Y-coordinate heatmap
    height, width = points_map.shape[:2]
    y_heatmap = points_map[:, :, 1].copy()
    y_heatmap[~valid_mask] = np.nan

    # Normalize for visualization
    y_min, y_max = np.nanmin(y_heatmap), np.nanmax(y_heatmap)
    y_normalized = ((y_heatmap - y_min) / (y_max - y_min) * 255).astype(np.uint8)
    y_normalized[~valid_mask] = 0

    # Apply colormap
    y_colormap = cv2.applyColorMap(y_normalized, cv2.COLORMAP_JET)

    # Blend with original image
    overlay = cv2.addWeighted(image_uint8, 0.5, cv2.cvtColor(y_colormap, cv2.COLOR_BGR2RGB), 0.5, 0)
    height_map_path = export_path("height_map.png")
    cv2.imwrite(height_map_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"\nHeight visualization saved as {height_map_path}")
    print("(Blue = ceiling/top, Red = floor/bottom)")

    # Use automatic detection by default
    choice = "1"

    if choice == "2":
        print(f"\nImage dimensions: {height} rows x {width} columns")
        print("Row 0 is at the top, row {} is at the bottom".format(height-1))

        ceiling_row = int(input("Enter row number for ceiling line (smaller number, top of image): "))
        floor_row = int(input("Enter row number for floor line (larger number, bottom of image): "))

        if ceiling_row >= height or floor_row >= height or ceiling_row < 0 or floor_row < 0:
            print("Invalid row numbers")
            return None

        # Get points at these rows
        ceiling_points = points_map[ceiling_row, valid_mask[ceiling_row, :]]
        floor_points = points_map[floor_row, valid_mask[floor_row, :]]

        if len(ceiling_points) == 0 or len(floor_points) == 0:
            print("No valid points at selected rows")
            return None

        ceiling_y_value = np.mean(ceiling_points[:, 1])
        floor_y_value = np.mean(floor_points[:, 1])

        print(f"\nManual selection:")
        print(f"Floor Y-coordinate: {floor_y_value:.3f} m")
        print(f"Ceiling Y-coordinate: {ceiling_y_value:.3f} m")

    room_height = abs(floor_y_value - ceiling_y_value)

    # Ask for calibration
    print(f"\n{'='*60}")
    print("CALIBRATION")
    print(f"{'='*60}")

    # Ask user for true room height
    print(f"\nEstimated room height: {room_height:.2f}m")
    true_height_input = input("Enter the true room height in meters (or press Enter to use default 2.4m): ").strip()

    if true_height_input:
        try:
            true_height = float(true_height_input)
            print(f"Using room height: {true_height:.2f}m")
        except ValueError:
            print("Invalid input, using default 2.4m")
            true_height = 2.4
    else:
        true_height = 2.4  # Default room height in meters
        print(f"Using default room height: {true_height:.2f}m")

    calibration_factor = 1.0
    calibration_factor = true_height / room_height

    print(f"\nCalibration factor: {calibration_factor:.4f}")
    print(f"Scaling all 3D points by {calibration_factor:.4f}...")

    # Apply calibration to the points map
    points_map_calibrated = points_map * calibration_factor

    # Recalculate with calibrated values
    if choice == "2":
        ceiling_points_cal = points_map_calibrated[ceiling_row, valid_mask[ceiling_row, :]]
        floor_points_cal = points_map_calibrated[floor_row, valid_mask[floor_row, :]]
        ceiling_y_cal = np.mean(ceiling_points_cal[:, 1])
        floor_y_cal = np.mean(floor_points_cal[:, 1])
    else:
        valid_points_cal = points_map_calibrated[valid_mask]
        y_coords_cal = valid_points_cal[:, 1]
        floor_y_cal = np.percentile(y_coords_cal, 95)
        ceiling_y_cal = np.percentile(y_coords_cal, 5)

    room_height_calibrated = abs(floor_y_cal - ceiling_y_cal)

    print(f"\nCalibrated measurements:")
    print(f"Floor Y-coordinate: {floor_y_cal:.3f} m")
    print(f"Ceiling Y-coordinate: {ceiling_y_cal:.3f} m")
    print(f"Calibrated room height: {room_height_calibrated:.3f} m")

    # Save calibrated point map
    calibrated_points_path = export_path("calibrated_points.npy")
    np.save(calibrated_points_path, points_map_calibrated)
    print(f"\nCalibrated 3D point map saved to '{calibrated_points_path}'")

    return room_height_calibrated, calibration_factor, points_map_calibrated

# Pass normal map if available
normal_map_data = output["normal"].cpu().numpy() if "normal" in output else None
result = calculate_room_height(
    output["depth"].cpu().numpy(),
    output["mask"].cpu().numpy(),
    output["points"].cpu().numpy(),
    input_image.cpu().numpy().transpose(1, 2, 0),
    normal_map_data
)

room_height_value = None

if result is not None:
    room_height, calibration_factor, calibrated_points = result
    room_height_value = room_height
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Final room height: {room_height:.2f} meters")
    print(f"Calibration factor applied: {calibration_factor:.4f}")
    if calibration_factor != 1.0:
        print(f"\nAll dimensions in the 3D point map have been scaled by {calibration_factor:.4f}")
        print("You can now use the calibrated point map for accurate measurements!")

    # Calculate room dimensions (visible floor area)
    def calculate_room_dimensions(points_map, mask, normal_map=None):
        """Calculate the visible floor dimensions from calibrated 3D points"""
        # In OpenCV camera coordinates: x=right, y=down, z=forward

        valid_mask = mask > 0
        valid_points = points_map[valid_mask]

        if len(valid_points) == 0:
            return None

        # Use normal map to identify floor if available
        if normal_map is not None:
            normal_y = normal_map[:, :, 1]
            floor_mask = (normal_y < -0.7) & valid_mask
            if np.any(floor_mask):
                floor_points = points_map[floor_mask]
                print(f"Using normal-based floor detection: {np.sum(floor_mask)} floor pixels")
            else:
                # Fallback
                print("Normal-based floor detection found no pixels, using Y threshold")
                y_coords = valid_points[:, 1]
                y_threshold = np.percentile(y_coords, 80)
                floor_points = valid_points[valid_points[:, 1] > y_threshold]
        else:
            # Get Y-coordinates to identify floor points
            y_coords = valid_points[:, 1]
            # Define floor as points in the bottom 20% of Y range (high Y values)
            y_threshold = np.percentile(y_coords, 80)  # Top 20% of Y values
            floor_points = valid_points[valid_points[:, 1] > y_threshold]

        if len(floor_points) == 0:
            print("No floor points detected")
            return None

        # Extract X (width) and Z (depth) coordinates
        x_coords = floor_points[:, 0]  # Horizontal (left-right)
        z_coords = floor_points[:, 2]  # Depth (forward from camera)

        # Calculate dimensions
        floor_width = np.max(x_coords) - np.min(x_coords)
        floor_depth = np.max(z_coords) - np.min(z_coords)

        # Also calculate total room dimensions from all points
        all_x = valid_points[:, 0]
        all_z = valid_points[:, 2]

        total_width = np.max(all_x) - np.min(all_x)
        total_depth = np.max(all_z) - np.min(all_z)

        return {
            'floor_width': floor_width,
            'floor_depth': floor_depth,
            'total_width': total_width,
            'total_depth': total_depth,
            'floor_area': floor_width * floor_depth
        }

    print(f"\n{'='*60}")
    print("ROOM DIMENSIONS")
    print(f"{'='*60}")

    dimensions = calculate_room_dimensions(calibrated_points, output["mask"].cpu().numpy(), normal_map_data)

    if dimensions:
        print("\nVisible Floor Dimensions:")
        print(f"  Width (left-right):  {dimensions['floor_width']:.2f} meters")
        print(f"  Length (depth):      {dimensions['floor_depth']:.2f} meters")
        print(f"  Floor area:          {dimensions['floor_area']:.2f} m²")

        print("\nTotal Visible Space:")
        print(f"  Width (left-right):  {dimensions['total_width']:.2f} meters")
        print(f"  Depth (front-back):  {dimensions['total_depth']:.2f} meters")
        print(f"  Height:              {room_height:.2f} meters")
        print(f"  Volume (approx):     {dimensions['total_width'] * dimensions['total_depth'] * room_height:.2f} m³")

        # Create a proper top-down view of the floor
        print("\nGenerating top-down floor plan visualization...")

        # Get floor points using normal map if available
        valid_mask = output["mask"].cpu().numpy() > 0
        if normal_map_data is not None:
            normal_y = normal_map_data[:, :, 1]
            floor_mask_normal = (normal_y < -0.7) & valid_mask
            if np.any(floor_mask_normal):
                floor_points = calibrated_points[floor_mask_normal]
                print(f"Floor plan using {np.sum(floor_mask_normal)} normal-detected floor pixels")
            else:
                valid_points = calibrated_points[valid_mask]
                y_coords = valid_points[:, 1]
                floor_y_threshold = np.percentile(y_coords, 80)
                floor_points = valid_points[valid_points[:, 1] > floor_y_threshold]
        else:
            valid_points = calibrated_points[valid_mask]
            y_coords = valid_points[:, 1]
            floor_y_threshold = np.percentile(y_coords, 80)
            floor_points = valid_points[valid_points[:, 1] > floor_y_threshold]

        # Extract X and Z coordinates for top-down view
        x_coords = floor_points[:, 0]
        z_coords = floor_points[:, 2]

        # Create a grid for the floor plan
        # Define resolution (pixels per meter)
        pixels_per_meter = 100  # Higher = more detail

        # Get bounds
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)

        # Create image dimensions
        floor_width_px = int((x_max - x_min) * pixels_per_meter) + 100
        floor_depth_px = int((z_max - z_min) * pixels_per_meter) + 100

        # Create blank floor plan image with dark background
        floor_plan = np.ones((floor_depth_px, floor_width_px, 3), dtype=np.uint8) * 240

        # Create a filled floor area with better visualization
        # Draw filled circles to create a solid floor appearance
        for point in floor_points:
            x, y, z = point
            # Convert to pixel coordinates
            px = int((x - x_min) * pixels_per_meter) + 50
            pz = int((z - z_min) * pixels_per_meter) + 50

            if 0 <= px < floor_width_px and 0 <= pz < floor_depth_px:
                # Draw filled circle to make floor solid
                cv2.circle(floor_plan, (px, pz), 3, (200, 200, 200), -1)

        # Add grid lines every meter with darker color
        for i in range(50, floor_width_px, pixels_per_meter):
            cv2.line(floor_plan, (i, 50), (i, floor_depth_px - 50), (150, 150, 150), 1)
        for i in range(50, floor_depth_px, pixels_per_meter):
            cv2.line(floor_plan, (50, i), (floor_width_px - 50, i), (150, 150, 150), 1)

        # Draw border around floor area
        cv2.rectangle(floor_plan, (50, 50), (floor_width_px - 50, floor_depth_px - 50), (100, 100, 100), 2)

        # Add measurements on the edges with better positioning
        cv2.putText(floor_plan, f"{dimensions['floor_width']:.2f}m",
                    (floor_width_px//2 - 60, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        cv2.putText(floor_plan, f"{dimensions['floor_depth']:.2f}m",
                    (5, floor_depth_px//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

        # Add title
        cv2.putText(floor_plan, "Top-Down Floor Plan",
                    (floor_width_px//2 - 150, floor_depth_px - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        floor_plan_path = export_path("floor_plan.png")
        cv2.imwrite(floor_plan_path, floor_plan)
        print(f"Floor plan saved as '{floor_plan_path}'")
        print(f"Floor plan resolution: {floor_width_px}x{floor_depth_px} pixels ({pixels_per_meter} px/m)")

        # Save dimensions to file
        dimensions_text_path = export_path("room_dimensions.txt")
        with open(dimensions_text_path, "w") as f:
            f.write("ROOM DIMENSIONS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Height: {room_height:.2f} m\n\n")
            f.write("Visible Floor:\n")
            f.write(f"  Width:  {dimensions['floor_width']:.2f} m\n")
            f.write(f"  Length: {dimensions['floor_depth']:.2f} m\n")
            f.write(f"  Area:   {dimensions['floor_area']:.2f} m²\n\n")
            f.write("Total Visible Space:\n")
            f.write(f"  Width:  {dimensions['total_width']:.2f} m\n")
            f.write(f"  Depth:  {dimensions['total_depth']:.2f} m\n")
            f.write(f"  Height: {room_height:.2f} m\n")
            f.write(f"  Volume: {dimensions['total_width'] * dimensions['total_depth'] * room_height:.2f} m³\n")
            f.write(f"\nCalibration factor: {calibration_factor:.4f}\n")

        print(f"\nDimensions saved to '{dimensions_text_path}'")

        # Create annotated image with dimensions
        print("\nCreating annotated image with dimensions...")

        # Get original image
        annotated_image = cv2.cvtColor(cv2.imread(input_file), cv2.COLOR_BGR2RGB)
        h, w = annotated_image.shape[:2]

        # Draw complete room boundary outlines with measurements
        if normal_map_data is not None:
            normal_y = normal_map_data[:, :, 1]
            valid_mask = output["mask"].cpu().numpy() > 0

            # Find ceiling boundary
            ceiling_mask = (normal_y > 0.7) & valid_mask
            ceiling_uint8 = (ceiling_mask * 255).astype(np.uint8)
            contours_ceiling, _ = cv2.findContours(ceiling_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find floor boundary
            floor_mask = (normal_y < -0.7) & valid_mask
            floor_uint8 = (floor_mask * 255).astype(np.uint8)
            contours_floor, _ = cv2.findContours(floor_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw complete ceiling contour in magenta
            if len(contours_ceiling) > 0:
                cv2.drawContours(annotated_image, contours_ceiling, -1, (255, 0, 255), 5)

                largest_ceiling = max(contours_ceiling, key=cv2.contourArea)
                pts = largest_ceiling[:, 0, :]

                # Add DEPTH label on top edge
                min_y = np.min(pts[:, 1])
                top_pts = pts[pts[:, 1] < min_y + 30]
                if len(top_pts) > 1:
                    top_pts_sorted = top_pts[np.argsort(top_pts[:, 0])]
                    mid_x = (top_pts_sorted[0][0] + top_pts_sorted[-1][0]) // 2
                    mid_y = (top_pts_sorted[0][1] + top_pts_sorted[-1][1]) // 2
                    depth_label = f"{dimensions['total_depth']:.2f}m"
                    # Place label BELOW the line (inside the room)
                    cv2.putText(annotated_image, depth_label, (mid_x - 70, mid_y + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 4)

                # Add WIDTH label on left edge
                min_x = np.min(pts[:, 0])
                left_pts = pts[pts[:, 0] < min_x + 30]
                if len(left_pts) > 1:
                    left_pts_sorted = left_pts[np.argsort(left_pts[:, 1])]
                    mid_x = (left_pts_sorted[0][0] + left_pts_sorted[-1][0]) // 2
                    mid_y = (left_pts_sorted[0][1] + left_pts_sorted[-1][1]) // 2
                    width_label = f"{dimensions['total_width']:.2f}m"
                    # Place label to the RIGHT of the line (inside the room)
                    cv2.putText(annotated_image, width_label, (mid_x + 20, mid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 4)

            # Draw complete floor contour in magenta
            if len(contours_floor) > 0:
                cv2.drawContours(annotated_image, contours_floor, -1, (255, 0, 255), 5)

            # Draw HEIGHT measurement line on right edge
            if len(contours_ceiling) > 0 and len(contours_floor) > 0:
                largest_ceiling = max(contours_ceiling, key=cv2.contourArea)
                largest_floor = max(contours_floor, key=cv2.contourArea)

                ceiling_pts = largest_ceiling[:, 0, :]
                floor_pts = largest_floor[:, 0, :]

                # Find rightmost points
                max_x_ceil = np.max(ceiling_pts[:, 0])
                max_x_floor = np.max(floor_pts[:, 0])

                right_ceiling_pts = ceiling_pts[ceiling_pts[:, 0] > max_x_ceil - 30]
                right_floor_pts = floor_pts[floor_pts[:, 0] > max_x_floor - 30]

                if len(right_ceiling_pts) > 0 and len(right_floor_pts) > 0:
                    ceiling_pt = right_ceiling_pts[np.argmax(right_ceiling_pts[:, 0])]
                    floor_pt = right_floor_pts[np.argmax(right_floor_pts[:, 0])]

                    x_pos = min(ceiling_pt[0], floor_pt[0])
                    y_top = ceiling_pt[1]
                    y_bottom = floor_pt[1]

                    # Draw height line with arrows in red
                    cv2.line(annotated_image, (x_pos, y_top), (x_pos, y_bottom), (255, 0, 0), 5)
                    cv2.arrowedLine(annotated_image, (x_pos, y_top + 40), (x_pos, y_top), (255, 0, 0), 5, tipLength=0.3)
                    cv2.arrowedLine(annotated_image, (x_pos, y_bottom - 40), (x_pos, y_bottom), (255, 0, 0), 5, tipLength=0.3)

                    # Height label at midpoint
                    mid_y = (y_top + y_bottom) // 2
                    height_label = f"{room_height:.2f}m"
                    cv2.putText(annotated_image, height_label, (x_pos - 130, mid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)

        # Detect ceiling corner points automatically
        print(f"\n{'='*60}")
        print("AUTOMATIC CEILING CORNER DETECTION")
        print(f"{'='*60}")

        if normal_map_data is not None and len(contours_ceiling) > 0:
            largest_ceiling = max(contours_ceiling, key=cv2.contourArea)

            # Get convex hull to find the outer boundary
            hull = cv2.convexHull(largest_ceiling)
            hull_perimeter = cv2.arcLength(hull, True)

            print(f"\nCeiling contour points: {len(largest_ceiling)}")
            print(f"Convex hull points: {len(hull)}")

            # Use polygon approximation on the convex hull to find corners
            # Try different epsilon values to get exactly 4 corners
            corners = None
            for epsilon_factor in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
                epsilon = epsilon_factor * hull_perimeter
                approx = cv2.approxPolyDP(hull, epsilon, True)
                approx_corners = approx.reshape(-1, 2)

                print(f"Epsilon {epsilon_factor}: {len(approx_corners)} corners")

                if len(approx_corners) == 4:
                    corners = approx_corners
                    print(f"Found exactly 4 corners with epsilon={epsilon_factor}")
                    break
                elif len(approx_corners) >= 4 and corners is None:
                    # Keep this as backup if we don't find exactly 4
                    corners = approx_corners

            if corners is None:
                print("Fallback: Using hull corners directly")
                corners = hull.reshape(-1, 2)

            # If we have more than 4 corners, select the 4 with the most extreme positions
            if len(corners) > 4:
                print(f"Selecting 4 corners from {len(corners)} candidates")
                # Find the 4 corner points that are most extreme
                x_coords = corners[:, 0]
                y_coords = corners[:, 1]

                # Find indices of extreme points
                left_idx = np.argmin(x_coords)
                right_idx = np.argmax(x_coords)
                top_idx = np.argmin(y_coords)
                bottom_idx = np.argmax(y_coords)

                # Get unique indices
                indices = list(set([left_idx, right_idx, top_idx, bottom_idx]))

                # If we have exactly 4 unique indices, use them
                if len(indices) == 4:
                    corners = corners[indices]
                else:
                    # Otherwise take first 4
                    corners = corners[:4]

            # Sort corners to identify top-left, top-right, bottom-left, bottom-right
            sorted_by_y = corners[np.argsort(corners[:, 1])]

            # Get top two and bottom two points
            if len(sorted_by_y) >= 4:
                top_two = sorted_by_y[:2]
                bottom_two = sorted_by_y[-2:]
            else:
                # Use what we have
                mid_idx = len(sorted_by_y) // 2
                top_two = sorted_by_y[:mid_idx] if mid_idx > 0 else sorted_by_y[:1]
                bottom_two = sorted_by_y[mid_idx:] if mid_idx < len(sorted_by_y) else sorted_by_y[-1:]

            # Sort each pair by X coordinate
            top_left = top_two[np.argmin(top_two[:, 0])]
            top_right = top_two[np.argmax(top_two[:, 0])]
            bottom_left = bottom_two[np.argmin(bottom_two[:, 0])]
            bottom_right = bottom_two[np.argmax(bottom_two[:, 0])]

            print(f"\nIdentified 4 ceiling corners where green line changes direction:")
            print(f"  Top-left: ({top_left[0]}, {top_left[1]})")
            print(f"  Top-right: ({top_right[0]}, {top_right[1]})")
            print(f"  Bottom-left: ({bottom_left[0]}, {bottom_left[1]})")
            print(f"  Bottom-right: ({bottom_right[0]}, {bottom_right[1]})")

            # Calculate all possible distances between corners to find the correct diagonals
            depth_mask = output["mask"].cpu().numpy()

            corners_list = [
                ("top-left", top_left),
                ("top-right", top_right),
                ("bottom-left", bottom_left),
                ("bottom-right", bottom_right)
            ]

            # Calculate distances between all corner pairs
            distances = []
            for i in range(len(corners_list)):
                for j in range(i+1, len(corners_list)):
                    name1, pt1 = corners_list[i]
                    name2, pt2 = corners_list[j]

                    if depth_mask[pt1[1], pt1[0]] > 0 and depth_mask[pt2[1], pt2[0]] > 0:
                        pt1_3d = calibrated_points[pt1[1], pt1[0]]
                        pt2_3d = calibrated_points[pt2[1], pt2[0]]
                        dist = np.linalg.norm(pt1_3d - pt2_3d)
                        distances.append((dist, name1, name2, pt1, pt2))

            # Sort by distance
            distances.sort(reverse=True)

            print(f"\nAll ceiling corner distances:")
            for dist, n1, n2, _, _ in distances:
                print(f"  {n1} to {n2}: {dist:.2f}m")

            # Use the longest diagonal for green line and a good diagonal for measurement
            if len(distances) >= 2:
                # Find the diagonal (opposite corners, not adjacent)
                diagonal_good = None
                for dist, n1, n2, pt1, pt2 in distances:
                    # Check if it's a diagonal (not adjacent corners)
                    if ("top" in n1 and "bottom" in n2) or ("top" in n2 and "bottom" in n1):
                        if ("left" in n1 and "right" in n2) or ("left" in n2 and "right" in n1):
                            diagonal_good = (dist, n1, n2, pt1, pt2)
                            break

                # Get another diagonal
                diagonal2_data = None
                for dist, n1, n2, pt1, pt2 in distances:
                    if (n1, n2) != (diagonal_good[1], diagonal_good[2]):
                        if ("top" in n1 and "bottom" in n2) or ("top" in n2 and "bottom" in n1):
                            if ("left" in n1 and "right" in n2) or ("left" in n2 and "right" in n1):
                                diagonal2_data = (dist, n1, n2, pt1, pt2)
                                break

            # Create clean visualization with corner points (use original image)
            corner_vis = cv2.cvtColor(cv2.imread(input_file), cv2.COLOR_BGR2RGB)

            # Draw ceiling boundary contour in green (thicker line)
            cv2.drawContours(corner_vis, [largest_ceiling], -1, (0, 255, 0), 3)

            # Draw all 4 corners with different colors
            cv2.circle(corner_vis, tuple(top_left), 12, (255, 0, 0), -1)  # Red
            cv2.circle(corner_vis, tuple(top_right), 12, (0, 0, 255), -1)  # Blue
            cv2.circle(corner_vis, tuple(bottom_left), 12, (255, 255, 0), -1)  # Yellow
            cv2.circle(corner_vis, tuple(bottom_right), 12, (255, 0, 255), -1)  # Magenta

            # Measure distances along the green contour between consecutive corners
            # Find the positions of corners in the contour
            contour_points = largest_ceiling[:, 0, :]

            # Find indices of corners in the contour
            corner_coords = [top_left, top_right, bottom_left, bottom_right]
            corner_names = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
            corner_colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

            # Find closest contour point indices for each corner
            corner_indices = []
            for corner in corner_coords:
                distances_to_contour = np.sqrt(np.sum((contour_points - corner)**2, axis=1))
                closest_idx = np.argmin(distances_to_contour)
                corner_indices.append(closest_idx)

            # Sort corners by their position along the contour
            sorted_corner_data = sorted(zip(corner_indices, corner_coords, corner_names, corner_colors))

            print(f"\nMeasuring along green contour between consecutive corners:")

            # Measure distances between consecutive corners along the contour
            for i in range(len(sorted_corner_data)):
                idx1, pt1, name1, color1 = sorted_corner_data[i]
                idx2, pt2, name2, color2 = sorted_corner_data[(i + 1) % len(sorted_corner_data)]

                # Get contour segment between these two corners
                if idx1 < idx2:
                    segment_indices = list(range(idx1, idx2 + 1))
                else:
                    # Wrap around
                    segment_indices = list(range(idx1, len(contour_points))) + list(range(0, idx2 + 1))

                # Calculate distance by summing up 3D distances along the contour
                total_dist = 0
                for j in range(len(segment_indices) - 1):
                    idx_a = segment_indices[j]
                    idx_b = segment_indices[j + 1]

                    pt_a = contour_points[idx_a]
                    pt_b = contour_points[idx_b]

                    if depth_mask[pt_a[1], pt_a[0]] > 0 and depth_mask[pt_b[1], pt_b[0]] > 0:
                        pt_a_3d = calibrated_points[pt_a[1], pt_a[0]]
                        pt_b_3d = calibrated_points[pt_b[1], pt_b[0]]
                        total_dist += np.linalg.norm(pt_a_3d - pt_b_3d)

                # Place label at midpoint of segment
                mid_idx = segment_indices[len(segment_indices) // 2]
                mid_pt = contour_points[mid_idx]

                # Use bright color for label
                label_color = (255, 255, 255)  # White
                cv2.putText(corner_vis, f"{total_dist:.2f}m",
                           (mid_pt[0] - 40, mid_pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)

                print(f"  {name1} to {name2} (along contour): {total_dist:.2f}m")

            ceiling_corners_path = export_path("ceiling_corners.png")
            cv2.imwrite(ceiling_corners_path, cv2.cvtColor(corner_vis, cv2.COLOR_RGB2BGR))
            print(f"\nCeiling corner visualization saved as '{ceiling_corners_path}'")

        # Save annotated image
        annotated_dimensions_path = export_path("annotated_dimensions.png")
        cv2.imwrite(annotated_dimensions_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"\nAnnotated image saved as '{annotated_dimensions_path}'")

        # Add interactive distance measurement tool
        print(f"\n{'='*60}")
        print("POINT-TO-POINT DISTANCE MEASUREMENT")
        print(f"{'='*60}")
        print("You can measure the distance between any two points in the image.")
        print(f"Image dimensions: {h} rows (height) x {w} columns (width)")
        print("Coordinates: X is horizontal (0 to {}), Y is vertical (0 to {})".format(w-1, h-1))

        # Skip interactive point-to-point measurement
        measure = 'n'
        while True:
            if measure != 'y':
                break

            try:
                # Get first point
                x1 = int(input("Enter X coordinate of point 1 (horizontal, 0-{}): ".format(w-1)))
                y1 = int(input("Enter Y coordinate of point 1 (vertical, 0-{}): ".format(h-1)))

                # Get second point
                x2 = int(input("Enter X coordinate of point 2 (horizontal, 0-{}): ".format(w-1)))
                y2 = int(input("Enter Y coordinate of point 2 (vertical, 0-{}): ".format(h-1)))

                # Validate coordinates
                if x1 < 0 or x1 >= w or y1 < 0 or y1 >= h or x2 < 0 or x2 >= w or y2 < 0 or y2 >= h:
                    print("Error: Coordinates out of bounds!")
                    continue

                # Get 3D points from calibrated point map
                point1_3d = calibrated_points[y1, x1]
                point2_3d = calibrated_points[y2, x2]

                # Check if points are valid (have depth data)
                mask = output["mask"].cpu().numpy()
                if mask[y1, x1] == 0 or mask[y2, x2] == 0:
                    print("Error: One or both points have no depth data!")
                    continue

                # Calculate Euclidean distance
                distance = np.linalg.norm(point1_3d - point2_3d)

                # Get depth values
                depth1 = point1_3d[2]  # Z coordinate is depth
                depth2 = point2_3d[2]

                print(f"\n{'='*40}")
                print(f"Point 1 (x={x1}, y={y1}):")
                print(f"  3D position: [{point1_3d[0]:.2f}, {point1_3d[1]:.2f}, {point1_3d[2]:.2f}]")
                print(f"  Depth: {depth1:.2f}m")
                print(f"\nPoint 2 (x={x2}, y={y2}):")
                print(f"  3D position: [{point2_3d[0]:.2f}, {point2_3d[1]:.2f}, {point2_3d[2]:.2f}]")
                print(f"  Depth: {depth2:.2f}m")
                print(f"\nDistance between points: {distance:.2f}m")
                print(f"{'='*40}")

                # Create visualization
                vis_image = annotated_image.copy()
                cv2.circle(vis_image, (x1, y1), 8, (255, 0, 0), -1)  # Red circle for point 1
                cv2.circle(vis_image, (x2, y2), 8, (0, 0, 255), -1)  # Blue circle for point 2
                cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Red line

                # Add label
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                cv2.putText(vis_image, f"{distance:.2f}m", (mid_x - 50, mid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

                distance_measurement_path = export_path("distance_measurement.png")
                cv2.imwrite(distance_measurement_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                print(f"\nVisualization saved as '{distance_measurement_path}'")

            except ValueError:
                print("Error: Invalid input! Please enter integer coordinates.")
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue

else:
    print("Room height calculation failed.")

print("\nStarting BFL staging to generate a furnished version of the room.")
style_prompt = input("Enter staging style description (default: modern organic furniture with layered textures): ").strip()
if not style_prompt:
    style_prompt = "modern organic furniture with layered textures"

ceiling_height_for_bfl = room_height_value if room_height_value is not None else 2.4

try:
    bfl_results = run_bfl_pipeline(
        image_path=input_path,
        style_prompt=style_prompt,
        ceiling_height=ceiling_height_for_bfl,
        output_dir=EXPORT_DIR,
        model=model,
        device=device,
    )
    print("\nBFL staging completed!")
    print(f"  Staged render: {bfl_results['staged_path']}")
    print(f"  Annotated furniture dimensions: {bfl_results['annotated_path']}")
    print(f"  Furniture metadata: {bfl_results['metadata_path']}")
except Exception as exc:
    print(f"\nBFL staging failed: {exc}")
