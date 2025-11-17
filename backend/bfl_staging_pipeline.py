import argparse
import base64
import colorsys
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import torch
import dotenv
from moge.model.v2 import MoGeModel
from furniture_detection import (
    FurnitureDetector,
    FurnitureDetectorUnavailable,
)
from sam_subject_extractor import (
    SamSubjectExtractor,
    SubjectExtractorUnavailable,
)
from furniture_3d_visualization import integrate_3d_visualization

dotenv.load_dotenv()


def log(message: str) -> None:
    """Consistent console logging."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[BFL {timestamp}] {message}")


def resolve_device(model: Optional[MoGeModel], override: Optional[torch.device]) -> torch.device:
    if override is not None:
        return override
    if model is not None:
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_reference_image(image_rgb: np.ndarray) -> str:
    """Encode RGB numpy array to base64 PNG string for BFL upload."""
    bgr_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", bgr_image)
    if not success:
        raise RuntimeError("Failed to encode reference image for BFL request.")
    return base64.b64encode(buffer).decode("utf-8")


def make_json_serializable(data):
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    if isinstance(data, list):
        return [make_json_serializable(v) for v in data]
    if isinstance(data, tuple):
        return [make_json_serializable(v) for v in data]
    if isinstance(data, np.ndarray):
        return make_json_serializable(data.tolist())
    if isinstance(data, np.generic):
        return data.item()
    return data


@dataclass
class RoomDescription:
    """Lightweight summary of key room appearance cues used for prompting."""

    wall_color_name: str
    wall_color_hex: str
    floor_color_name: str
    floor_color_hex: str
    lighting: str
    orientation: str
    brightness: float


def load_moge_model(device: torch.device) -> MoGeModel:
    log(f"Loading MoGe-2 model on {device}...")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.eval()
    log("MoGe-2 model loaded")
    return model




"""
git init                 # if you haven’t already
git remote add origin git@github.com:<user>/<private-repo>.git
git add .
git commit -m "Initial staging pipeline"
git push -u origin main  # or whatever branch name you prefer


"""
def read_image_rgb(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_color_name(color: np.ndarray) -> Tuple[str, str]:
    r, g, b = color / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue_deg = h * 360
    hex_code = "#{:02x}{:02x}{:02x}".format(
        int(color[0]), int(color[1]), int(color[2])
    )

    if v < 0.2:
        base = "near-black"
    elif v < 0.4:
        base = "dark"
    elif v < 0.75:
        base = "mid-tone"
    else:
        base = "light"

    if s < 0.15:
        if v > 0.85:
            tone = "bright white"
        elif v > 0.65:
            tone = "soft cream"
        elif v > 0.45:
            tone = "warm gray"
        else:
            tone = "charcoal gray"
    else:
        if hue_deg < 15 or hue_deg >= 345:
            tone = "rich red"
        elif hue_deg < 45:
            tone = "warm ochre"
        elif hue_deg < 75:
            tone = "sunlit yellow"
        elif hue_deg < 150:
            tone = "fresh green"
        elif hue_deg < 210:
            tone = "cool teal"
        elif hue_deg < 270:
            tone = "deep blue"
        elif hue_deg < 300:
            tone = "vibrant violet"
        else:
            tone = "rosy mauve"

    descriptive = f"{base} {tone}".strip()
    return descriptive, hex_code


def describe_room_colors(image_rgb: np.ndarray) -> RoomDescription:
    h, w, _ = image_rgb.shape
    top = image_rgb[: max(1, h // 3)]
    bottom = image_rgb[h - max(1, h // 3) :]

    wall_color = np.mean(top, axis=(0, 1))
    floor_color = np.mean(bottom, axis=(0, 1))

    wall_name, wall_hex = rgb_to_color_name(wall_color)
    floor_name, floor_hex = rgb_to_color_name(floor_color)

    brightness = float(np.mean(image_rgb))
    if brightness > 190:
        lighting = "bright daylight"
    elif brightness > 150:
        lighting = "soft natural light"
    elif brightness > 110:
        lighting = "even ambient light"
    else:
        lighting = "moody low light"

    orientation = "landscape" if w >= h else "portrait"

    return RoomDescription(
        wall_color_name=wall_name,
        wall_color_hex=wall_hex,
        floor_color_name=floor_name,
        floor_color_hex=floor_hex,
        lighting=lighting,
        orientation=orientation,
        brightness=brightness,
    )


def infer_room_type(dimensions: Dict[str, float]) -> str:
    area = dimensions.get("area", 0.0)
    width = dimensions.get("width", 0.0)
    depth = dimensions.get("depth", 0.0)

    if area < 10 or max(width, depth) < 3:
        return "bedroom"
    if area >= 25 or width >= 4.5:
        return "living room"
    if depth > width * 1.3:
        return "hallway"
    return "living room"


def compute_floor_ceiling_masks(
    points: np.ndarray,
    valid_mask: np.ndarray,
    normals: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if normals is not None:
        normal_y = normals[:, :, 1]
        floor_mask = (normal_y < -0.7) & valid_mask
        ceiling_mask = (normal_y > 0.7) & valid_mask
    else:
        y_coords = points[:, :, 1]
        valid_y = y_coords[valid_mask]
        if len(valid_y) == 0:
            return np.zeros_like(valid_mask), np.zeros_like(valid_mask)

        floor_threshold = np.percentile(valid_y, 90)
        ceiling_threshold = np.percentile(valid_y, 10)
        floor_mask = valid_mask & (y_coords >= floor_threshold)
        ceiling_mask = valid_mask & (y_coords <= ceiling_threshold)

    return floor_mask, ceiling_mask


def run_moge_analysis(
    model: MoGeModel,
    image_path: Path,
    device: torch.device,
    true_height: float,
    label: str = "image",
) -> Dict:
    log(f"Running MoGe inference on {label} ({image_path})")
    start = time.perf_counter()
    image_rgb = read_image_rgb(image_path)
    tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(
        2, 0, 1
    )

    with torch.inference_mode():
        output = model.infer(tensor)

    points = output["points"].detach().cpu().numpy()
    mask = output["mask"].detach().cpu().numpy() > 0
    depth = output["depth"].detach().cpu().numpy()
    normals = (
        output["normal"].detach().cpu().numpy() if "normal" in output else None
    )

    floor_mask, ceiling_mask = compute_floor_ceiling_masks(points, mask, normals)

    valid_points = points[mask]
    if len(valid_points) == 0:
        raise RuntimeError("MoGe inference did not return any valid points.")

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

    floor_points_cal = points_calibrated[floor_mask]
    if len(floor_points_cal) == 0:
        floor_points_cal = points_calibrated[mask]

    x_coords = floor_points_cal[:, 0]
    z_coords = floor_points_cal[:, 2]

    width = float(np.max(x_coords) - np.min(x_coords)) if len(x_coords) else 0.0
    depth_span = float(np.max(z_coords) - np.min(z_coords)) if len(z_coords) else 0.0
    area = width * depth_span

    dimensions = {
        "width": round(width, 3),
        "depth": round(depth_span, 3),
        "height": round(true_height, 3),
        "area": round(area, 3),
    }

    floor_bounds = {
        "x_min": float(np.min(x_coords)) if len(x_coords) else 0.0,
        "x_max": float(np.max(x_coords)) if len(x_coords) else 0.0,
        "z_min": float(np.min(z_coords)) if len(z_coords) else 0.0,
        "z_max": float(np.max(z_coords)) if len(z_coords) else 0.0,
    }

    elapsed = time.perf_counter() - start
    log(
        f"MoGe inference for {label} finished in {elapsed:.2f}s "
        f"(W≈{width:.2f}m, D≈{depth_span:.2f}m, H≈{true_height:.2f}m)"
    )

    return {
        "image_rgb": image_rgb,
        "image_path": str(image_path),
        "points_calibrated": points_calibrated,
        "depth": depth,
        "mask": mask,
        "normals": normals,
        "floor_mask": floor_mask,
        "ceiling_mask": ceiling_mask,
        "dimensions": dimensions,
        "floor_bounds": floor_bounds,
        "calibration_factor": calibration_factor,
        "true_height": true_height,
        "estimated_height": estimated_height,
        "room_type": infer_room_type(dimensions),
    }


def suggest_furniture(room_type: str, area: float) -> List[str]:
    if room_type == "bedroom":
        items = [
            "upholstered bed with headboard",
            "matching nightstands with lamps",
            "low dresser",
            "soft area rug",
        ]
    elif room_type == "hallway":
        items = [
            "slim console table",
            "decorative bench",
            "wall art",
            "accent lighting",
        ]
    else:
        items = [
            "generous sofa",
            "accent chairs",
            "coffee table",
            "media console",
            "floor lamp",
        ]

    if area < 12:
        return items[:3]
    if area > 25:
        return items + ["decorative plants"]
    return items


def build_bfl_prompt(
    room_analysis: Dict,
    appearance: RoomDescription,
    style_prompt: str,
) -> str:
    dims = room_analysis["dimensions"]
    room_type = room_analysis["room_type"]
    suggestions = suggest_furniture(room_type, dims["area"])

    prompt = f"""Photorealistic real estate photograph taken with a {appearance.orientation} composition after professional virtual staging.

ROOM CHARACTERISTICS:
- Dimensions approx {dims['width']:.2f}m W x {dims['depth']:.2f}m D x {dims['height']:.2f}m H
- Walls: {appearance.wall_color_name} ({appearance.wall_color_hex})
- Flooring: {appearance.floor_color_name} ({appearance.floor_color_hex})
- Lighting: {appearance.lighting}
- Room type: {room_type}

STAGING DIRECTIVE:
- Base reference photo is attached. Treat it as the exact architecture and perspective to preserve.
- Keep every architectural detail identical to the reference room (walls, floor, ceiling, trim, windows, doors, lighting).
- Maintain the exact camera perspective and lens feel.
- Only add furniture and decor matching this request: {style_prompt}.
- Focus on: {', '.join(suggestions)}.
- Furniture must respect the measured proportions so nothing feels oversized.
- Avoid any construction changes, no new walls, no removal of built features."""

    return prompt


def request_kontext_edit(
    prompt: str,
    api_key: str,
    init_image_b64: str,
    aspect_ratio: Optional[str] = None,
    timeout: int = 240,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[bytes, str]:
    reporter = status_callback or log
    headers = {
        "accept": "application/json",
        "x-key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": prompt,
        "input_image": init_image_b64,
        "output_format": "png",
        "safety_tolerance": 2,
    }

    if aspect_ratio:
        payload["aspect_ratio"] = aspect_ratio

    endpoint = "https://api.bfl.ai/v1/flux-kontext-pro"

    reporter("Submitting FLUX Kontext edit request")
    response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    polling_url = data.get("polling_url")
    if not polling_url:
        raise RuntimeError(f"BFL response missing polling_url: {data}")

    start = time.time()
    last_status = None
    while True:
        poll_resp = requests.get(polling_url, headers=headers, timeout=30)
        poll_resp.raise_for_status()
        poll_data = poll_resp.json()

        status = poll_data.get("status")
        if status != last_status:
            reporter(f"FLUX status: {status}")
            last_status = status
        if status == "Ready":
            result = poll_data.get("result", {})
            sample_url = result.get("sample")
            if not sample_url:
                raise RuntimeError("BFL result missing sample URL.")
            image_resp = requests.get(sample_url, timeout=60)
            image_resp.raise_for_status()
            reporter("FLUX result ready, downloading image")
            return image_resp.content, sample_url

        if status in {"Error", "Failed"}:
            raise RuntimeError(f"BFL generation failed: {poll_data}")

        if time.time() - start > timeout:
            raise TimeoutError("Timed out waiting for BFL generation.")

        time.sleep(1.0)


def annotate_furniture_boxes(
    image_rgb: np.ndarray,
    furniture_regions: List[Dict],
    output_path: Path,
) -> None:
    if not furniture_regions:
        cv2.imwrite(str(output_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        log("No furniture regions detected – saved plain staged image")
        return

    overlay = image_rgb.copy()
    color_palette = [
        (255, 99, 71),
        (60, 179, 113),
        (65, 105, 225),
        (218, 165, 32),
        (238, 130, 238),
    ]

    for idx, region in enumerate(furniture_regions, start=1):
        bbox = region["pixel_bbox"]
        dims = region["dimensions_m"]
        color = color_palette[(idx - 1) % len(color_palette)]

        contour = region.get("contour")
        if contour:
            pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], True, color, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(
                overlay,
                (bbox["x1"], bbox["y1"]),
                (bbox["x2"], bbox["y2"]),
                color,
                2,
            )

        label_parts = []
        item_name = region.get("type")
        if item_name:
            label_parts.append(f"{item_name}")
        label_parts.append(
            f"{dims['width']:.2f}m W x {dims['depth']:.2f}m D x {dims['height']:.2f}m H"
        )
        confidence = region.get("confidence")
        if confidence is not None:
            label_parts.append(f"{confidence*100:.0f}%")
        label_text = " | ".join(label_parts) or f"Item {idx}"
        text_origin = (bbox["x1"] + 4, max(20, bbox["y1"] - 10))

        cv2.putText(
            overlay,
            label_text,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    log(f"Annotated furniture overlay saved to {output_path}")


def save_sam_outline_visualization(
    image_rgb: np.ndarray,
    furniture_regions: List[Dict],
    output_path: Path,
) -> None:
    if not furniture_regions:
        cv2.imwrite(str(output_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        log("No SAM furniture regions detected – saved staged image without outlines.")
        return

    overlay = image_rgb.copy()
    color_palette = [
        (255, 99, 71),
        (60, 179, 113),
        (65, 105, 225),
        (218, 165, 32),
        (238, 130, 238),
        (0, 255, 255),
    ]

    for idx, region in enumerate(furniture_regions, start=1):
        contour = region.get("contour")
        if not contour:
            continue
        color = color_palette[(idx - 1) % len(color_palette)]
        pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], True, color, 2, cv2.LINE_AA)
        text_origin = tuple(pts[0, 0])
        cv2.putText(
            overlay,
            f"SAM {idx}",
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    log(f"SAM outline visualization saved to {output_path}")


PRIMARY_DIMENSION_TARGETS = [
    ("Rug", {"rug"}, 1),
    ("Table", {"table"}, 1),
    ("Sofa", {"sofa"}, 2),
    ("TV", {"tv"}, 1),
    ("Cabinet/Shelves", {"cabinet", "shelving", "console"}, 1),
]


def save_primary_dimension_overlay(
    image_rgb: np.ndarray,
    furniture_regions: List[Dict],
    output_path: Path,
) -> None:
    def area(region: Dict) -> float:
        dims = region["dimensions_m"]
        return dims["width"] * dims["depth"]

    overlay = image_rgb.copy()
    colors = [
        (255, 255, 255),
        (255, 215, 0),
        (173, 255, 47),
        (0, 255, 255),
        (255, 182, 193),
    ]
    color_iter = iter(colors)

    items_drawn = 0
    for label, type_set, count in PRIMARY_DIMENSION_TARGETS:
        candidates = [
            r for r in furniture_regions if r.get("type") in type_set and r.get("dimensions_in")
        ]
        if not candidates:
            continue
        candidates.sort(key=area, reverse=True)
        color = next(color_iter, (255, 255, 255))
        for idx, region in enumerate(candidates[:count], start=1):
            contour = region.get("contour")
            if contour is not None:
                if region.get("type") == "rug":
                    rect = cv2.minAreaRect(np.array(contour, dtype=np.float32))
                    box = cv2.boxPoints(rect).astype(np.int32)
                    cv2.polylines(overlay, [box], True, color, 2, cv2.LINE_AA)
                else:
                    pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(overlay, [pts], True, color, 2, cv2.LINE_AA)
            dims_in = region["dimensions_in"]
            text_label = label
            if count > 1:
                text_label = f"{label} {idx}"
            text = f"{text_label}: {dims_in['width']:.0f}\" W x {dims_in['depth']:.0f}\" D x {dims_in['height']:.0f}\" H"
            bbox = region["pixel_bbox"]
            anchor = (bbox["x1"], max(20, bbox["y1"] - 10 - 20 * (idx - 1)))
            cv2.putText(
                overlay,
                text,
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            items_drawn += 1

    if items_drawn == 0:
        cv2.imwrite(str(output_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        log("No primary furniture dimensions drawn; saved staged image instead.")
        return
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    log(f"Primary furniture dimensions saved to {output_path}")


def save_furniture_metadata(
    metadata_path: Path,
    furniture_regions: List[Dict],
    staged_image_path: Path,
    room_dimensions: Dict[str, float],
    prompt: str,
) -> None:
    payload = {
        "staged_image": str(staged_image_path),
        "furniture_regions": furniture_regions,
        "room_dimensions_m": room_dimensions,
        "prompt_used": prompt,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_serializable(payload), f, indent=2)
    log(f"Furniture metadata saved to {metadata_path}")


def save_surface_visualization(
    normals: Optional[np.ndarray],
    mask: np.ndarray,
    output_path: Path,
) -> None:
    if normals is None:
        return

    normal_y = normals[:, :, 1]
    valid_mask = mask.astype(bool)
    floor_mask = (normal_y < -0.7) & valid_mask
    ceiling_mask = (normal_y > 0.7) & valid_mask

    surface_vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    surface_vis[floor_mask] = [255, 0, 0]
    surface_vis[ceiling_mask] = [0, 255, 0]
    surface_vis[(~floor_mask) & (~ceiling_mask) & valid_mask] = [100, 100, 100]

    cv2.imwrite(str(output_path), surface_vis)
    log(f"Surface detection visualization saved to {output_path}")


_SAM_EXTRACTOR: Optional[SamSubjectExtractor] = None


def get_sam_extractor(
    device: Optional[torch.device] = None,
) -> Optional[SamSubjectExtractor]:
    """
    Lazily initialize Segment Anything subject extraction if configured.
    """
    global _SAM_EXTRACTOR
    if _SAM_EXTRACTOR is False:
        return None
    if _SAM_EXTRACTOR is None:
        checkpoint = os.getenv("SAM_CHECKPOINT_PATH")
        if not checkpoint:
            log("SAM subject extraction disabled: SAM_CHECKPOINT_PATH not set.")
            _SAM_EXTRACTOR = False
            return None
        model_type = os.getenv("SAM_MODEL_TYPE", "vit_h")
        target_device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        try:
            _SAM_EXTRACTOR = SamSubjectExtractor(
                checkpoint_path=Path(checkpoint),
                model_type=model_type,
                device=target_device,
                log_fn=log,
            )
        except SubjectExtractorUnavailable as exc:
            log(f"SAM subject extraction unavailable: {exc}")
            _SAM_EXTRACTOR = False
    return _SAM_EXTRACTOR or None


def run_bfl_pipeline(
    image_path: Path,
    style_prompt: str,
    ceiling_height: float,
    output_dir: Path,
    model: Optional[MoGeModel] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    if ceiling_height <= 0:
        raise ValueError("Ceiling height must be positive.")

    api_key = os.environ.get("BFL_API_KEY")
    if not api_key:
        raise EnvironmentError("BFL_API_KEY is not set. Please export your API key before running.")

    resolved_device = resolve_device(model, device)
    owns_model = model is None
    if owns_model:
        model = load_moge_model(resolved_device)
    else:
        log("Using caller-provided MoGe model for staging")
        model = model.to(resolved_device)
        model.eval()

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_path = output_dir / f"{image_path.stem}_bfl_staged.png"
    annotated_path = output_dir / f"{image_path.stem}_bfl_dimensions.png"
    metadata_path = output_dir / f"{image_path.stem}_bfl_boxes.json"

    overall_start = time.perf_counter()
    log("Starting BFL staging pipeline")
    log(f"Input image: {image_path}")
    sam_extractor = get_sam_extractor(device=resolved_device)
    if sam_extractor is None:
        raise RuntimeError(
            "SAM-based furniture detection requires SAM_CHECKPOINT_PATH. "
            "Please download a SAM checkpoint and set SAM_CHECKPOINT_PATH before running."
        )
    try:
        furniture_detector = FurnitureDetector(
            sam_extractor=sam_extractor,
            log_fn=log,
        )
    except FurnitureDetectorUnavailable as exc:
        raise RuntimeError(f"SAM furniture detector unavailable: {exc}") from exc

    original_analysis = run_moge_analysis(
        model=model,
        image_path=image_path,
        device=resolved_device,
        true_height=ceiling_height,
        label="original",
    )

    room_description = describe_room_colors(original_analysis["image_rgb"])
    log(
        f"Room appearance: walls {room_description.wall_color_name} / "
        f"floor {room_description.floor_color_name} / lighting {room_description.lighting}"
    )
    prompt = build_bfl_prompt(original_analysis, room_description, style_prompt)
    reference_b64 = encode_reference_image(original_analysis["image_rgb"])

    log("Requesting staged furnishing edits from FLUX Kontext...")
    request_start = time.perf_counter()
    image_bytes, sample_url = request_kontext_edit(
        prompt=prompt,
        api_key=api_key,
        init_image_b64=reference_b64,
        status_callback=log,
    )
    with staged_path.open("wb") as f:
        f.write(image_bytes)
    log(
        f"Staged image saved to {staged_path} (source URL expires soon). "
        f"Generation took {time.perf_counter() - request_start:.2f}s"
    )

    log("Running MoGe on staged image for furniture measurements...")
    staged_analysis = run_moge_analysis(
        model=model,
        image_path=staged_path,
        device=resolved_device,
        true_height=ceiling_height,
        label="staged",
    )

    staged_surface_path = output_dir / f"{image_path.stem}_surface_staged.png"
    save_surface_visualization(
        normals=staged_analysis["normals"],
        mask=staged_analysis["mask"],
        output_path=staged_surface_path,
    )

    if sam_extractor:
        subject_outputs = sam_extractor.save_subject_assets(
            image_rgb=staged_analysis["image_rgb"],
            output_dir=output_dir,
            stem=staged_path.stem,
        )
        if subject_outputs:
            log(f"SAM subject mask saved to {subject_outputs.mask_path}")
            log(f"SAM subject cutout saved to {subject_outputs.cutout_path}")

    furniture_regions = furniture_detector.detect(
        image_rgb=staged_analysis["image_rgb"],
        points_calibrated=staged_analysis["points_calibrated"],
        depth_mask=staged_analysis["mask"],
        normals=staged_analysis["normals"],
    )

    sam_outline_path = output_dir / f"{image_path.stem}_sam_outlines.png"
    save_sam_outline_visualization(
        image_rgb=staged_analysis["image_rgb"],
        furniture_regions=furniture_regions,
        output_path=sam_outline_path,
    )

    primary_dims_path = output_dir / f"{image_path.stem}_key_furniture_dimensions.png"
    save_primary_dimension_overlay(
        image_rgb=staged_analysis["image_rgb"],
        furniture_regions=furniture_regions,
        output_path=primary_dims_path,
    )

    viz_outputs = integrate_3d_visualization(
        image_rgb=staged_analysis["image_rgb"],
        points_calibrated=staged_analysis["points_calibrated"],
        furniture_regions=furniture_regions,
        output_dir=output_dir,
    )
    if viz_outputs.get("overlay"):
        log(f"3D bounding boxes saved to {viz_outputs['overlay']}")
    if viz_outputs.get("spec"):
        log(f"Spec view saved to {viz_outputs['spec']}")
    if viz_outputs.get("comparison"):
        log(f"Comparison view saved to {viz_outputs['comparison']}")
    if viz_outputs.get("data"):
        log(f"3D box data saved to {viz_outputs['data']}")

    for region in furniture_regions:
        region.pop("mask", None)

    annotate_furniture_boxes(
        image_rgb=staged_analysis["image_rgb"],
        furniture_regions=furniture_regions,
        output_path=annotated_path,
    )
    save_furniture_metadata(
        metadata_path=metadata_path,
        furniture_regions=furniture_regions,
        staged_image_path=staged_path,
        room_dimensions=staged_analysis["dimensions"],
        prompt=prompt,
    )

    if not furniture_regions:
        log("Warning: No furniture regions were detected in the staged image.")

    log(f"Annotated dimension image saved to {annotated_path}")
    log(f"Furniture metadata written to {metadata_path}")
    log(f"Pipeline finished in {time.perf_counter() - overall_start:.2f}s")

    return {
        "staged_path": str(staged_path),
        "annotated_path": str(annotated_path),
        "metadata_path": str(metadata_path),
        "furniture_regions": furniture_regions,
        "room_dimensions": staged_analysis["dimensions"],
        "bfl_url": sample_url,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a room with BFL FLUX-1.1 and re-dimension furnishings with MoGe."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the original (unfurnished) room image.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="warm contemporary furniture with layered textures",
        help="Furniture style to request from BFL.",
    )
    parser.add_argument(
        "--ceiling-height",
        type=float,
        default=2.4,
        help="True ceiling height in meters used to calibrate MoGe.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "bfl",
        help="Directory where staged and annotated images will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_bfl_pipeline(
        image_path=args.image,
        style_prompt=args.style,
        ceiling_height=args.ceiling_height,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
