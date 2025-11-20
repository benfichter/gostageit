# Real Estate Room Staging & Measurement System

A comprehensive system for automated room measurement and AI-powered virtual staging using MoGe-2 depth estimation and BFL FLUX 1.1 image generation.

## Overview

This project provides two complementary tools:
1. **main.py** - Interactive CLI tool for detailed room analysis and measurement
2. **api.py** - REST API for automated virtual staging with room measurements

Both tools use **MoGe-2** (Metric 3D Geometry Estimation) for accurate depth and normal estimation, enabling precise real-world measurements from single images.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Main.py - Interactive Room Analysis](#mainpy---interactive-room-analysis)
- [API.py - Virtual Staging API](#apipy---virtual-staging-api)
- [Technical Details](#technical-details)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

---

## Features

### Room Measurement (Both Tools)
- **Metric 3D Reconstruction** - MoGe-2 provides metric-scale depth and normal maps
- **Automatic Floor/Ceiling Detection** - Uses surface normals for accurate identification
- **Ceiling Corner Detection** - Identifies 4 corner points with distance measurements
- **Calibrated Measurements** - User-specified ceiling height for precise calibration
- **Multiple Visualizations** - Depth maps, floor plans, annotated images, corner diagrams

### Virtual Staging (API Only)
- **AI-Powered Staging** - BFL FLUX 1.1 Pro for photorealistic furniture placement
- **Room-Type Aware** - Specialized staging for bedrooms, living rooms, offices, etc.
- **Style Customization** - Detailed style descriptions (modern, traditional, Scandinavian, etc.)
- **Scale-Aware Placement** - Furniture sized appropriately for room dimensions
- **Non-Destructive** - Preserves original architecture and lighting

---

## Installation

### 1. Install Dependencies

```bash
# Core dependencies
pip install torch torchvision opencv-python numpy pillow python-dotenv

# MoGe-2 model
pip install 'moge @ git+https://github.com/microsoft/moge.git'

# For API only
pip install flask flask-cors werkzeug requests
```

### 2. Configure API Keys

Create a `.env` file in the workspace directory:

```bash
# Required for api.py virtual staging
BFL_API_KEY=your-bfl-api-key-here
```

**Important:** The `.env` file is gitignored to protect your API keys.

### 3. GPU Support (Recommended)

Both tools support CUDA for faster processing:
- MoGe-2 inference is significantly faster on GPU
- CPU mode is supported but slower

---

## Quick Start

### Interactive Room Analysis
```bash
python main.py
# Follow prompts to enter image filename and ceiling height
```
All generated images, point clouds, and text summaries are saved under `exports/<timestamp>_<input_name>/`. When the analysis finishes, the script automatically calls the BFL **FLUX.1 Kontext [pro]** editing pipeline (requires `BFL_API_KEY`) to add furniture directly onto the original photo, then re-measures the staged result to produce furniture dimension overlays.

### Virtual Staging API
```bash
# Start the server
python api.py

# In another terminal, test with curl
curl -X POST \
  -F "image=@room.jpg" \
  -F "style_prompt=Scandinavian style with light wood and minimalist decor" \
  -F "ceiling_height=2.7" \
  -F "return_format=url" \
  http://localhost:5000/stage
```

### BFL Staging + Dimensioning Pipeline (Kontext)
```bash
# Requires BFL_API_KEY in your environment
export BFL_API_KEY=sk-your-key

# Generate staged furnishings and furniture dimensions in one go
python bfl_staging_pipeline.py \
  --image test3.jpg \
  --style "modern organic furniture with layered neutrals" \
  --ceiling-height 2.6 \
  --output-dir output/bfl
```

This pipeline:
- Runs MoGe on the source room for accurate measurements
- Sends the original room photo + a structured “staged furniture” prompt to **FLUX.1 Kontext [pro]** (`/flux-kontext-pro`) so the model edits the photo in place instead of re-generating it
- Polls Kontext until the edited (staged) image is ready
- Runs MoGe again on the staged output to recover furniture geometry
- Outputs the staged render, a dimensioned bounding-box overlay, and JSON metadata under `output/bfl`

Example Kontext request body (the script fills this in automatically using your photo and style prompt):

```json
{
  "prompt": "Add staged furniture to this unfurnished living room photo while keeping the architecture and lighting identical...",
  "input_image": "<base64-encoded PNG of your original room>",
  "output_format": "png"
}
```

### Furniture bounding boxes via MoGe (default)

Furniture segmentation is now purely geometry-driven—no SAM checkpoints or extra downloads required. The detector:

1. Uses MoGe normals to carve out the floor, ceiling, and planar walls.
2. Clusters the remaining calibrated points (plus depth discontinuities) into connected components.
3. Filters clusters by physical size and point-count to reject artifacts.
4. Measures width × depth × height directly from the calibrated points and fits PCA-aligned cuboids for visualization.

Outputs from this step include:

- `*_detections.png` – depth-based overlay that numbers every detected furniture blob.
- `*_key_furniture_dimensions.png` – annotations for the key furniture (rug, table, sofas, TV console / shelves) with MoGe W×D×H values rendered next to the detections. If you set `GOOGLE_API_KEY`, Gemini chooses the five most important regions; otherwise we fall back to heuristics.

Optional Gemini setup for key-item selection:

```bash
export GOOGLE_API_KEY=sk-your-genai-key
# Or configure Vertex routing as usual:
# export GOOGLE_CLOUD_PROJECT=...
# export GOOGLE_GENAI_USE_VERTEXAI=True
# Optional override (defaults to gemini-2.0-flash-lite):
# export GEMINI_SELECTION_MODEL=gemini-1.5-flash
```

With that in place, the pipeline uploads a numbered geometry overlay and candidate metadata, asks Gemini to choose the five most important furniture items, and only labels those pieces.
The JSON response should look like `{"selected_items": [{"region_id": 4, "label": "main sofa"}, ...]}`; those labels show up verbatim on `_key_furniture_dimensions.png`.

### 3D furniture outlines

After MoGe segments each furniture cluster, we project the calibrated 3D bounding boxes back into the staged photo. This produces perspective-correct outlines - even for angled rugs or tall shelves - and the pipeline writes:

- `*_3d_boxes.png` – staged image with 3D wireframes
- `*_3d_data.json` – list of centers/corners/metrics for every furniture box

Optional extras:

- `pip install alphashape` to tighten outlines for irregular objects
- `pip install open3d` to emit meshes/PLY files (hooks live in `furniture_3d_visualization.py`)

---

## Main.py - Interactive Room Analysis

### What It Does

An interactive command-line tool that performs comprehensive room analysis from a single image:

1. **Depth & Normal Estimation** - Uses MoGe-2 to generate metric-scale 3D point cloud
2. **Surface Detection** - Identifies floor, ceiling, and walls using surface normals
3. **Height Calibration** - User provides actual ceiling height for accurate scaling
4. **Dimension Calculation** - Measures room width, depth, height, area, and volume
5. **Ceiling Corner Detection** - Automatically finds 4 ceiling corners with measurements
6. **Visualization Generation** - Creates multiple annotated images and diagrams

### Usage

```bash
python main.py
```

**Interactive Prompts:**
1. Enter image filename (e.g., `room.jpg`)
2. Enter actual ceiling height in meters (default: 2.4m)

### How It Works

#### Step 1: Load & Infer
```python
# Load MoGe-2 model
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")

# Process image
output = model.infer(input_tensor)
# Returns: points (H,W,3), depth (H,W), normal (H,W,3), mask (H,W)
```

#### Step 2: Surface Detection
- Floor: Normals pointing UP (Y < -0.7 in camera coords)
- Ceiling: Normals pointing DOWN (Y > 0.7)
- Walls: Other surfaces

#### Step 3: Height Calibration
```python
# Detect floor/ceiling Y-coordinates
estimated_height = abs(floor_y - ceiling_y)

# User provides true ceiling height
calibration_factor = true_height / estimated_height

# Scale entire point cloud
calibrated_points = points_map * calibration_factor
```

#### Step 4: Dimension Calculation
- **Width** - X-axis extent of floor points
- **Depth** - Z-axis extent of floor points
- **Height** - Calibrated ceiling height
- **Area** - Width × Depth
- **Volume** - Width × Depth × Height

#### Step 5: Ceiling Corner Detection
1. Extract ceiling contour from normal map
2. Find convex hull
3. Use polygon approximation to find 4 corners
4. Measure distances along contour between corners
5. Generate annotated visualization

### Output Files

| File | Description |
|------|-------------|
| `output_depth.png` | Raw depth map (16-bit, millimeters) |
| `output_depth_colormap.png` | Colorized depth visualization |
| `output_normal.png` | Surface normal map |
| `surface_detection.png` | Floor (red), ceiling (green), walls (gray) |
| `height_map.png` | Y-coordinate heatmap overlay |
| `floor_plan.png` | Top-down view with measurements |
| `ceiling_corners.png` | Corner detection with distance labels |
| `annotated_dimensions.png` | Original image with dimension annotations |
| `room_dimensions.txt` | Text summary of all measurements |
| `calibrated_points.npy` | Numpy array of calibrated 3D points |

---

## API.py - Virtual Staging API

### What It Does

A Flask REST API that provides automated room staging with measurements:

1. **Room Measurement** - Detects dimensions with custom ceiling height
2. **Virtual Staging** - Generates furnished room using BFL FLUX 1.1
3. **Ceiling Corner Visualization** - Creates annotated corner diagram
4. **Flexible Output** - Returns images as base64 or URLs

### Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ POST /stage
       │ (image, room_type, style_description, ceiling_height)
       ▼
┌─────────────────────────────────────┐
│         Flask API (api.py)          │
├─────────────────────────────────────┤
│  1. detect_dimensions()             │
│     └─ MoGe-2 inference             │
│     └─ Floor/ceiling detection      │
│     └─ Calibration & measurement    │
│                                     │
│  2. generate_staged_image()         │
│     └─ Build detailed prompt        │
│     └─ BFL FLUX 1.1 generation  │
│                                     │
│  3. generate_ceiling_corners()      │
│     └─ Corner detection             │
│     └─ Distance measurement         │
│     └─ Visualization creation       │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│         JSON Response               │
├─────────────────────────────────────┤
│ - dimensions                        │
│ - staged_image (base64/URL)         │
│ - ceiling_corners_image             │
│ - ceiling_measurements              │
│ - room_type, style_description      │
└─────────────────────────────────────┘
```

### API Endpoints

#### POST /stage
Full staging with dimensions, staged image, and ceiling corners.

**Request:**
```bash
curl -X POST http://localhost:5000/stage \
  -F "image=@room.jpg" \
  -F "room_type=bedroom" \
  -F "style_description=Modern minimal with neutral tones" \
  -F "ceiling_height=2.7" \
  -F "return_format=base64"
```

**Parameters:**
- `image` (required) - Image file (PNG, JPG, JPEG)
- `room_type` (optional) - Type of room (default: "living room")
  - Examples: "bedroom", "office", "kitchen", "dining room", "nursery"
- `style_description` (optional) - Detailed style preferences
  - Examples: "Scandinavian with light wood", "Industrial loft", "Coastal blues and whites"
- `ceiling_height` (optional) - Ceiling height in meters (default: 2.4)
- `return_format` (optional) - "base64" or "url" (default: "base64")

**Response:**
```json
{
  "success": true,
  "request_id": "a5d74334-a1bd-4ff6-b2ed-7f527510e3a3",
  "timestamp": "2025-11-16T16:20:22.123Z",
  "original_image": {
    "width": 1920,
    "height": 1080
  },
  "dimensions": {
    "height": 2.7,
    "width": 5.03,
    "depth": 3.52,
    "area": 17.71,
    "unit": "meters"
  },
  "staged_image": "base64_encoded_string_or_url",
  "room_type": "bedroom",
  "style_description": "Modern minimal with neutral tones",
  "ceiling_corners_image": "base64_encoded_string_or_url",
  "ceiling_measurements": [
    {"from": "top_left", "to": "top_right", "distance": 5.12},
    {"from": "top_right", "to": "bottom_left", "distance": 6.23},
    {"from": "bottom_left", "to": "bottom_right", "distance": 4.98},
    {"from": "bottom_right", "to": "top_left", "distance": 6.15}
  ],
  "processing_time_seconds": 11.4
}
```

#### POST /dimensions-only
Get room dimensions without generating staged image.

**Request:**
```bash
curl -X POST http://localhost:5000/dimensions-only \
  -F "image=@room.jpg" \
  -F "ceiling_height=2.7"
```

**Response:**
```json
{
  "success": true,
  "dimensions": {
    "height": 2.7,
    "width": 5.03,
    "depth": 3.52,
    "area": 17.71,
    "unit": "meters"
  }
}
```

#### GET /health
Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "moge_loaded": true,
  "gemini_loaded": true
}
```

#### GET /outputs/{filename}
Retrieve generated images (when using `return_format=url`).

### How It Works

#### 1. detect_dimensions(image_path, ceiling_height, return_raw_output)
```python
# Infer with MoGe-2
output = moge_model.infer(input_tensor)

# Detect floor/ceiling using normals
floor_mask = (normal_y < -0.7) & valid_mask
ceiling_mask = (normal_y > 0.7) & valid_mask

# Calibrate to user-provided ceiling height
calibration_factor = ceiling_height / estimated_height
points_map_calibrated = points_map * calibration_factor

# Calculate floor dimensions
floor_width = max(x_coords) - min(x_coords)
floor_depth = max(z_coords) - min(z_coords)

return {
    'height': ceiling_height,
    'width': floor_width,
    'depth': floor_depth,
    'area': floor_width * floor_depth,
    'raw_output': {'points_map_calibrated', 'mask', 'normal_map'}
}
```

#### 2. generate_staged_image(image_path, dimensions, room_type, style_description)
```python
# Build detailed prompt
prompt = f"""
ROOM TYPE: {room_type}
ROOM MEASUREMENTS:
- Height: {dimensions['height']}m
- Width: {dimensions['width']}m
- Depth: {dimensions['depth']}m
- Floor Area: {dimensions['area']}m²

STAGING STYLE REQUESTED:
{style_description}

REQUIREMENTS:
- Create DSLR-quality professionally staged real estate photo for a {room_type}
- Add furniture appropriate for a {room_type}, scaled to room measurements
- Maintain exact camera angle and perspective
- Keep original architecture intact
...
"""

# Generate with BFL FLUX 1.1
response = gemini_client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=[prompt, image]
)

# Extract and save generated image
generated_image.save(output_path)
```

#### 3. generate_ceiling_corners(input_image_path, points_map, mask, normal_map)
```python
# Detect ceiling using normals
ceiling_mask = (normal_y > 0.7) & valid_mask
contours_ceiling = cv2.findContours(ceiling_mask, ...)

# Find 4 corners using polygon approximation
hull = cv2.convexHull(largest_ceiling)
corners = cv2.approxPolyDP(hull, epsilon, True)

# Sort into top-left, top-right, bottom-left, bottom-right
# Measure distances along contour between consecutive corners
# Generate visualization with labels

return {
    'success': True,
    'measurements': [
        {'from': 'top_left', 'to': 'top_right', 'distance': 5.12},
        ...
    ]
}
```

### Running the API

```bash
# Start server
python api.py

# Server runs on http://0.0.0.0:5000
# Outputs stored in ./outputs/
# Uploaded files temporarily in ./uploads/
```

### Example Usage Scenarios

**Bedroom Staging:**
```bash
curl -X POST http://localhost:5000/stage \
  -F "image=@empty_bedroom.jpg" \
  -F "room_type=bedroom" \
  -F "style_description=Scandinavian style with light wood bed frame, white linens, minimal nightstands, and potted plants"
```

**Home Office:**
```bash
curl -X POST http://localhost:5000/stage \
  -F "image=@empty_office.jpg" \
  -F "room_type=home office" \
  -F "style_description=Professional modern office with ergonomic desk, comfortable chair, bookshelf, and indirect lighting"
```

**Living Room:**
```bash
curl -X POST http://localhost:5000/stage \
  -F "image=@empty_living.jpg" \
  -F "room_type=living room" \
  -F "style_description=Warm traditional living room with comfortable sofa, area rug, coffee table, and accent chairs"
```

---

## Technical Details

### Camera Coordinate System
- **OpenCV convention**: X=right, Y=down, Z=forward
- **Floor normals**: Y < -0.7 (pointing upward)
- **Ceiling normals**: Y > 0.7 (pointing downward)

### MoGe-2 Model
- **Model**: `Ruicheng/moge-2-vitl-normal`
- **Inputs**: RGB image (H, W, 3), normalized to [0, 1]
- **Outputs**:
  - `points`: (H, W, 3) - Metric 3D points in camera coordinates
  - `depth`: (H, W) - Depth map in meters
  - `normal`: (H, W, 3) - Surface normals in camera coordinates
  - `mask`: (H, W) - Binary mask for valid pixels
  - `intrinsics`: (3, 3) - Camera intrinsics

### BFL FLUX 1.1
- **Endpoint**: `https://api.bfl.ai/v1/flux-pro-1.1`
- **Capabilities**: Fast, reliable text-to-image generation with strong prompt adherence
- **Input**: Text prompt (derived from MoGe measurements + style request)
- **Output**: Signed URL to a staged PNG image (valid for ~10 minutes)

### Measurement Accuracy
- **Calibration**: Required for metric accuracy
- **Floor Detection**: Uses normal-based surface classification
- **Dimension Calculation**: Based on calibrated 3D point cloud
- **Typical Accuracy**: ±5-10cm depending on image quality and calibration

### Processing Performance
- **MoGe-2 Inference**: ~2-5 seconds (GPU) / ~10-30 seconds (CPU)
- **BFL Generation**: ~5-15 seconds
- **Corner Detection**: ~0.5-1 second
- **Total API Response**: ~10-20 seconds per request

---

## Output Files

### Main.py Outputs

Each run stores every artifact under `exports/<timestamp>_<input_name>/` so exports never overwrite previous analyses.

| File | Description |
|------|-------------|
| `output_depth.png` | 16-bit depth map (mm) |
| `output_depth_colormap.png` | Colorized depth (Jet colormap) |
| `output_mask.png` | Valid pixel mask |
| `output_normal.png` | RGB-encoded normal map |
| `surface_detection.png` | Red=floor, Green=ceiling, Gray=walls |
| `height_map.png` | Y-coordinate heatmap overlay |
| `floor_plan.png` | Top-down view with grid and measurements |
| `ceiling_corners.png` | Ceiling with 4 corners marked and labeled |
| `annotated_dimensions.png` | Original with width/height/depth annotations |
| `*_3d_boxes.png` | Perspective-correct 3D furniture outlines |
| `*_3d_data.json` | JSON dump of 3D box centers and corners |
| `room_dimensions.txt` | Text summary of measurements |
| `calibrated_points.npy` | Calibrated 3D point cloud (NumPy array) |

### API.py Outputs

Files stored in `./outputs/` directory:

| File Pattern | Description |
|--------------|-------------|
| `{request_id}_staged.png` | Generated staged room image |
| `{request_id}_corners.png` | Ceiling corners visualization |

Uploaded files temporarily stored in `./uploads/` and cleaned up after processing.

---

## Troubleshooting

### Common Issues

**Issue: "No floor/ceiling detected"**
- **Cause**: Image doesn't show clear floor or ceiling surfaces
- **Solution**: Ensure image includes visible floor and ceiling, try different viewing angle

**Issue: "Dimensions seem incorrect"**
- **Cause**: Incorrect ceiling height calibration
- **Solution**: Provide accurate ceiling height in meters (measure with tape measure if possible)

**Issue: "API returns 500 error"**
- **Cause**: Missing GEMINI_API_KEY or invalid image
- **Solution**: Check `.env` file exists with valid API key, verify image is valid JPG/PNG

**Issue: "Slow processing on CPU"**
- **Cause**: MoGe-2 is compute-intensive
- **Solution**: Use GPU if available, or expect 10-30 second processing time

**Issue: "Ceiling corners not detected"**
- **Cause**: Ceiling not visible or irregular shape
- **Solution**: Ensure ceiling is visible in image, works best with rectangular rooms

**Issue: "Staged image doesn't match room"**
- **Cause**: Generic or unclear style description
- **Solution**: Provide detailed style_description with specific furniture and colors

### Error Messages

**"GEMINI_API_KEY not found in environment"**
```bash
# Create .env file
echo "GEMINI_API_KEY=your-key-here" > .env
```

**"Could not detect room dimensions"**
- Image lacks sufficient depth information
- Try different image with better lighting and visibility

**"No image file provided"**
```bash
# Ensure -F "image=@filename.jpg" in curl command
curl -X POST -F "image=@room.jpg" http://localhost:5000/dimensions-only
```

### Performance Optimization

**For faster API responses:**
1. Use GPU (CUDA) if available
2. Reduce image resolution before upload (1920x1080 recommended)
3. Use `return_format=url` to avoid base64 encoding overhead
4. Use `/dimensions-only` endpoint if staging not needed

**For better staging results:**
1. Use high-quality input images with good lighting
2. Provide detailed style_description with specific furniture items
3. Specify accurate room_type for appropriate furniture
4. Provide accurate ceiling_height for proper furniture scaling

---

## Configuration

### Environment Variables

Create `.env` file:
```bash
# Required for api.py
GEMINI_API_KEY=your-gemini-api-key-here
```

### API Configuration

Edit [api.py](api.py) constants:
```python
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
```

### Model Configuration

Both tools use:
- **MoGe-2 Model**: `Ruicheng/moge-2-vitl-normal` (auto-downloaded from HuggingFace)
- **Device**: Auto-detects CUDA GPU, falls back to CPU

---

## License

MIT

## Acknowledgments

- **MoGe-2**: Microsoft Research - [GitHub](https://github.com/microsoft/moge)
- **BFL**: FLUX 1.1 Pro text-to-image API
- **OpenCV**: Computer vision operations
- **Flask**: REST API framework
