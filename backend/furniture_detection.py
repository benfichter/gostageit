from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency
    from google import genai
    from google.genai.types import (
        GenerateContentConfig,
        HarmBlockThreshold,
        HarmCategory,
        HttpOptions,
        Part,
        SafetySetting,
    )
except ImportError:  # pragma: no cover - optional dependency
    genai = None
    GenerateContentConfig = None
    Part = None


class FurnitureLabelerUnavailable(RuntimeError):
    """Raised when Gemini labeler cannot be initialised."""


def default_log(message: str) -> None:
    print(message)


def detect_geometric_regions(
    points_calibrated: np.ndarray,
    mask: np.ndarray,
    normals: Optional[np.ndarray],
    room_height: float,
    log_fn: Callable[[str], None] = default_log,
    min_area_pixels: int = 400,
) -> List[Dict]:
    """Pure MoGe-based detection to guarantee metric fidelity."""
    log = log_fn
    log("Detecting furniture regions using MoGe geometry...")
    valid_mask = mask.astype(bool)
    if normals is not None:
        normal_y = normals[:, :, 1]
        floor_mask = (normal_y < -0.7) & valid_mask
        ceiling_mask = (normal_y > 0.7) & valid_mask
        vertical_component = np.abs(normal_y)
    else:
        y_coords = points_calibrated[:, :, 1]
        valid_y = y_coords[valid_mask]
        if len(valid_y) == 0:
            return []
        floor_thresh = np.percentile(valid_y, 90)
        ceil_thresh = np.percentile(valid_y, 10)
        floor_mask = valid_mask & (y_coords >= floor_thresh)
        ceiling_mask = valid_mask & (y_coords <= ceil_thresh)
        vertical_component = np.zeros_like(y_coords)

    structure_mask = valid_mask & ~(floor_mask | ceiling_mask)
    if normals is not None:
        wall_mask = structure_mask & (vertical_component <= 0.45)
    else:
        wall_mask = np.zeros_like(structure_mask)

    furniture_mask = structure_mask & ~wall_mask
    furniture_uint8 = (furniture_mask * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        furniture_uint8, connectivity=8
    )

    regions: List[Dict] = []
    kept = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area_pixels:
            continue

        component_mask = labels == label
        object_points = points_calibrated[component_mask]
        if len(object_points) < 30:
            continue

        x_vals = object_points[:, 0]
        y_vals = object_points[:, 1]
        z_vals = object_points[:, 2]

        dims = {
            "width": float(np.max(x_vals) - np.min(x_vals)),
            "height": float(np.max(y_vals) - np.min(y_vals)),
            "depth": float(np.max(z_vals) - np.min(z_vals)),
        }

        if dims["height"] > room_height * 0.92:
            continue
        if dims["width"] < 0.18 and dims["depth"] < 0.18:
            continue

        x1 = int(stats[label, cv2.CC_STAT_LEFT])
        y1 = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        regions.append(
            {
                "pixel_bbox": {"x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h},
                "dimensions_m": dims,
                "center": {
                    "x": float(np.mean(x_vals)),
                    "y": float(np.mean(y_vals)),
                    "z": float(np.mean(z_vals)),
                },
                "point_count": int(len(object_points)),
                "confidence": None,
                "type": "furniture",
                "mask": component_mask,
            }
        )
        kept += 1

    log(f"Detected {kept} furniture region(s) from MoGe geometry.")
    return regions


@dataclass
class GeminiLabelerConfig:
    project: str
    location: str = "global"
    model: str = "gemini-2.5-flash"
    max_objects: int = 25
    temperature: float = 0.2


class GeminiLabeler:
    def __init__(
        self,
        config: GeminiLabelerConfig,
        log_fn: Callable[[str], None] = default_log,
    ) -> None:
        if genai is None or GenerateContentConfig is None or Part is None:
            raise FurnitureLabelerUnavailable("google-genai must be installed.")
        self.config = config
        self.log = log_fn
        self._init_client()

    def _init_client(self) -> None:
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self.config.project)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", self.config.location)
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
        self.client = genai.Client(http_options=HttpOptions(api_version="v1"))
        self.response_config = GenerateContentConfig(
            system_instruction=(
                "Return bounding boxes as JSON array. Format:\n"
                '[{"box_2d":[y_min,x_min,y_max,x_max],"label":"sofa"}]\n'
                "Coordinates use 0-1000 normalization. No masks."
            ),
            temperature=self.config.temperature,
            safety_settings=[
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                )
            ],
            response_mime_type="application/json",
        )
        self.log(
            f"Gemini labeler initialised (project={self.config.project}, "
            f"location={self.config.location})."
        )

    def label(self, image_rgb: np.ndarray, regions: List[Dict]) -> None:
        if not regions:
            return

        rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".jpg", rgb)
        if not success:
            self.log("Failed to encode staged image for Gemini labeler.")
            return

        part = Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg")
        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=[part, "Describe each piece of furniture with bounding boxes."],
                config=self.response_config,
            )
        except Exception as exc:  # pragma: no cover - external call
            self.log(f"Gemini label request failed: {exc}")
            return

        detections = self._parse_response(response.text or "[]")
        if not detections:
            self.log("Gemini returned no boxes for labelling.")
            return

        height, width, _ = image_rgb.shape
        for region in regions:
            best = None
            best_iou = 0.0
            region_box = region["pixel_bbox"]
            for det in detections:
                pixel_box = self._normalised_to_pixels(det["box_2d"], width, height)
                iou = compute_iou(region_box, pixel_box)
                if iou > best_iou:
                    best_iou = iou
                    best = det
            if best and best_iou > 0.2:
                region["type"] = best.get("label", region.get("type"))
                region["gemini_bbox"] = best["box_2d"]
                region["confidence"] = best_iou

    def _parse_response(self, text: str) -> List[Dict]:
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [
                    det
                    for det in data
                    if isinstance(det, dict) and "box_2d" in det and len(det["box_2d"]) == 4
                ]
        except json.JSONDecodeError:
            self.log("Gemini returned malformed JSON for label response.")
        return []

    def _normalised_to_pixels(
        self, bbox: List[int], width: int, height: int
    ) -> Dict[str, int]:
        y_min, x_min, y_max, x_max = bbox
        x1 = int((x_min / 1000) * width)
        x2 = int((x_max / 1000) * width)
        y1 = int((y_min / 1000) * height)
        y2 = int((y_max / 1000) * height)
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def compute_iou(box_a: Dict[str, int], box_b: Dict[str, int]) -> float:
    xa1, ya1, xa2, ya2 = box_a["x1"], box_a["y1"], box_a["x2"], box_a["y2"]
    xb1, yb1, xb2, yb2 = box_b["x1"], box_b["y1"], box_b["x2"], box_b["y2"]
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter_area / float(area_a + area_b - inter_area)


def build_gemini_labeler(
    log_fn: Callable[[str], None] = default_log,
) -> GeminiLabeler:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise FurnitureLabelerUnavailable("GOOGLE_CLOUD_PROJECT must be set.")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    config = GeminiLabelerConfig(project=project, location=location)
    return GeminiLabeler(config, log_fn=log_fn)


__all__ = [
    "detect_geometric_regions",
    "GeminiLabeler",
    "GeminiLabelerConfig",
    "FurnitureLabelerUnavailable",
    "build_gemini_labeler",
    "compute_iou",
]
