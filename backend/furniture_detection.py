from __future__ import annotations

import cv2
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

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


class FurnitureDetectionUnavailable(RuntimeError):
    """Raised when the Gemini bounding-box API is not available."""


def default_log(message: str) -> None:
    print(message)


@dataclass
class FurnitureDetectorConfig:
    project: str
    location: str = "global"
    model: str = "gemini-2.5-flash"
    prompt: str = (
        "Identify every furniture item (chairs, sofas, tables, lamps, shelves, decor) "
        "visible in this staged room photo. Return only bounding boxes and labels in JSON."
    )
    temperature: float = 0.2
    max_objects: int = 25


class FurnitureDetector:
    """Uses Gemini bounding-box detection (google-genai) for object localization."""

    def __init__(
        self,
        config: FurnitureDetectorConfig,
        log_fn: Callable[[str], None] = default_log,
    ) -> None:
        if genai is None or GenerateContentConfig is None or Part is None:
            raise FurnitureDetectionUnavailable("google-genai must be installed.")
        self.config = config
        self.log = log_fn
        self._init_client()

    def _init_client(self) -> None:
        self.log(
            f"Initialising Gemini bounding-box detector "
            f"(project={self.config.project}, location={self.config.location})..."
        )
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self.config.project)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", self.config.location)
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

        self.client = genai.Client(http_options=HttpOptions(api_version="v1"))
        self.response_config = GenerateContentConfig(
            system_instruction=(
                "Return bounding boxes as a JSON array. Format is:\n"
                '[{"box_2d":[y_min,x_min,y_max,x_max],"label":"chair"}]\n'
                "Never return segmentation masks. Limit detections to "
                f"{self.config.max_objects} objects."
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
        self.log("Gemini detector ready.")

    def detect(
        self,
        image_rgb: np.ndarray,
        points_calibrated: np.ndarray,
        mask: np.ndarray,
        room_height: float,
    ) -> List[Dict]:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".jpg", image_bgr)
        if not success:
            self.log("Failed to encode staged image for Gemini request.")
            return []

        part = Part.from_data(data=buffer.tobytes(), mime_type="image/jpeg")
        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=[part, self.config.prompt],
                config=self.response_config,
            )
        except Exception as exc:  # pragma: no cover - external call
            self.log(f"Gemini detection failed: {exc}")
            return []

        detections = self._parse_response(response.text or "[]")
        if not detections:
            self.log("Gemini returned no bounding boxes.")
            return []

        height, width, _ = image_rgb.shape
        valid_mask = mask.astype(bool)
        results: List[Dict] = []

        for det in detections:
            bbox = det.get("box_2d")
            if not self._bbox_valid(bbox):
                continue

            x1, y1, x2, y2 = self._normalised_to_pixels(bbox, width, height)
            if x2 <= x1 or y2 <= y1:
                continue

            roi_mask = valid_mask[y1:y2, x1:x2]
            if not roi_mask.any():
                continue
            roi_points = points_calibrated[y1:y2, x1:x2][roi_mask]
            if len(roi_points) < 25:
                continue

            dims, center = self._metrics_from_points(roi_points)
            if dims is None:
                continue

            results.append(
                {
                    "pixel_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "dimensions_m": dims,
                    "center": center,
                    "point_count": int(len(roi_points)),
                    "type": det.get("label", "furniture"),
                    "confidence": None,
                    "vertex_box": bbox,
                }
            )

        self.log(f"Kept {len(results)} Gemini detections after depth filtering.")
        return results

    def _parse_response(self, text: str) -> List[Dict]:
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            self.log("Gemini returned malformed JSON for bounding boxes.")
        return []

    def _bbox_valid(self, bbox: Optional[List[int]]) -> bool:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        return True

    def _normalised_to_pixels(
        self, bbox: List[int], width: int, height: int
    ) -> tuple[int, int, int, int]:
        y_min, x_min, y_max, x_max = bbox
        x1 = int((x_min / 1000) * width)
        x2 = int((x_max / 1000) * width)
        y1 = int((y_min / 1000) * height)
        y2 = int((y_max / 1000) * height)
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))
        return x1, y1, x2, y2

    def _metrics_from_points(
        self, object_points: np.ndarray
    ) -> Optional[tuple[Dict[str, float], Dict[str, float]]]:
        if len(object_points) == 0:
            return None

        x_vals = object_points[:, 0]
        y_vals = object_points[:, 1]
        z_vals = object_points[:, 2]
        dims = {
            "width": float(np.max(x_vals) - np.min(x_vals)),
            "height": float(np.max(y_vals) - np.min(y_vals)),
            "depth": float(np.max(z_vals) - np.min(z_vals)),
        }
        center = {
            "x": float(np.mean(x_vals)),
            "y": float(np.mean(y_vals)),
            "z": float(np.mean(z_vals)),
        }
        return dims, center


def build_detector_from_env(log_fn: Callable[[str], None] = default_log) -> FurnitureDetector:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise FurnitureDetectionUnavailable(
            "GOOGLE_CLOUD_PROJECT must be set to use Gemini bounding boxes."
        )
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    config = FurnitureDetectorConfig(project=project, location=location)
    return FurnitureDetector(config, log_fn=log_fn)


__all__ = [
    "FurnitureDetector",
    "FurnitureDetectorConfig",
    "FurnitureDetectionUnavailable",
    "build_detector_from_env",
]
