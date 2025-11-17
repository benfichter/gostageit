from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from sam_subject_extractor import SamSubjectExtractor


def default_log(message: str) -> None:
    print(message)


def classify_region(dimensions: Dict[str, float]) -> str:
    width = dimensions["width"]
    depth = dimensions["depth"]
    height = dimensions["height"]

    footprint = width * depth
    flatness = height / max(width, depth, 1e-6)

    if footprint > 4.0 and flatness < 0.05:
        return "rug"
    if height > 1.4 and footprint < 2.0:
        return "shelving"
    if depth < 0.25 and width > 0.9 and height < 1.2:
        return "tv"
    if height < 1.2 and width > 0.8 and depth > 0.25:
        return "cabinet"
    if height > 0.8 and width > 1.2 and depth > 0.4:
        return "sofa"
    if height < 0.6 and footprint < 1.5:
        return "table"
    if width < 0.8 and depth < 0.8 and height > 0.6:
        return "chair"
    return "furniture"


class FurnitureDetectorUnavailable(RuntimeError):
    """Raised when SAM is not configured for furniture detection."""


def _bbox_iou(box_a: Dict[str, int], box_b: Dict[str, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a["x1"], box_a["y1"], box_a["x2"], box_a["y2"]
    bx1, by1, bx2, by2 = box_b["x1"], box_b["y1"], box_b["x2"], box_b["y2"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / max(area_a + area_b - inter_area, 1e-6)


class FurnitureDetector:
    """
    Uses Segment Anything masks for segmentation and MoGe points for dimensions.
    """

    def __init__(
        self,
        sam_extractor: SamSubjectExtractor,
        log_fn: Callable[[str], None] = default_log,
        min_area_ratio: float = 0.003,
        max_area_ratio: float = 0.4,
        min_points: int = 200,
        max_regions: int = 30,
        max_iou: float = 0.7,
    ) -> None:
        if sam_extractor is None:
            raise FurnitureDetectorUnavailable(
                "SAM extractor is required for furniture detection."
            )
        self.sam = sam_extractor
        self.log = log_fn or default_log
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_points = min_points
        self.max_regions = max_regions
        self.max_iou = max_iou

    def detect(
        self,
        image_rgb: np.ndarray,
        points_calibrated: np.ndarray,
        depth_mask: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        self.log("Detecting furniture via Segment Anything masks...")
        image_h, image_w = image_rgb.shape[:2]
        valid_mask = depth_mask.astype(bool)
        valid_mask &= np.isfinite(points_calibrated[:, :, 0])
        valid_mask &= np.isfinite(points_calibrated[:, :, 1])
        valid_mask &= np.isfinite(points_calibrated[:, :, 2])

        image_area = image_h * image_w
        min_area = image_area * self.min_area_ratio
        max_area = image_area * self.max_area_ratio

        masks = self.sam.generate_masks(image_rgb)
        if not masks:
            self.log("SAM returned no masks for this image.")
            return []

        masks = sorted(masks, key=lambda m: m["area"], reverse=True)
        regions: List[Dict] = []
        selected_bboxes: List[Dict[str, int]] = []

        region_counter = 1
        for mask_data in masks:
            area = mask_data.get("area", 0)
            if area < min_area or area > max_area:
                continue

            segmentation = mask_data["segmentation"].astype(bool)
            contour = self._extract_primary_contour(segmentation)
            if contour is None:
                continue

            bbox = self._contour_to_bbox(contour, image_w, image_h)
            if self._touches_border(bbox, image_w, image_h):
                continue

            object_mask = segmentation & valid_mask
            if np.count_nonzero(object_mask) < self.min_points:
                continue

            dims_center = self._compute_dimensions(points_calibrated, object_mask)
            if dims_center is None:
                continue
            dims, center = dims_center

            if any(_bbox_iou(bbox, existing) > self.max_iou for existing in selected_bboxes):
                continue

            region = {
                "region_id": region_counter,
                "pixel_bbox": bbox,
                "contour": contour.tolist(),
                "dimensions_m": dims,
                "center": center,
                "point_count": int(np.count_nonzero(object_mask)),
                "mask": object_mask,
                "confidence": float(mask_data.get("predicted_iou", 0.0)),
                "type": classify_region(dims),
                "sam_area": int(area),
            }
            regions.append(region)
            selected_bboxes.append(bbox)
            region_counter += 1

            if len(regions) >= self.max_regions:
                break

        self.log(f"Detected {len(regions)} furniture region(s) via SAM.")
        return regions

    def _compute_dimensions(
        self,
        points_calibrated: np.ndarray,
        mask: np.ndarray,
    ) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
        object_points = points_calibrated[mask]
        if len(object_points) < self.min_points:
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

    @staticmethod
    def _extract_primary_contour(mask: np.ndarray) -> Optional[np.ndarray]:
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 50:
            return None
        return largest.squeeze(axis=1).astype(np.int32)

    @staticmethod
    def _contour_to_bbox(contour: np.ndarray, image_w: int, image_h: int) -> Dict[str, int]:
        x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
        x1 = int(np.clip(x, 0, image_w - 1))
        y1 = int(np.clip(y, 0, image_h - 1))
        x2 = int(np.clip(x + w, 0, image_w - 1))
        y2 = int(np.clip(y + h, 0, image_h - 1))
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    @staticmethod
    def _touches_border(
        bbox: Dict[str, int],
        image_w: int,
        image_h: int,
        margin: int = 2,
        full_ratio: float = 0.6,
    ) -> bool:
        width = bbox["x2"] - bbox["x1"]
        height = bbox["y2"] - bbox["y1"]
        touches_left = bbox["x1"] <= margin and width > image_w * full_ratio
        touches_right = bbox["x2"] >= image_w - margin and width > image_w * full_ratio
        touches_top = bbox["y1"] <= margin and height > image_h * full_ratio
        touches_bottom = bbox["y2"] >= image_h - margin and height > image_h * full_ratio
        return touches_left or touches_right or touches_top or touches_bottom


def detect_geometric_regions(
    points_calibrated: np.ndarray,
    mask: np.ndarray,
    normals: Optional[np.ndarray],
    room_height: float,
    log_fn: Callable[[str], None] = default_log,
    min_area_pixels: int = 400,
) -> List[Dict]:
    """
    Legacy depth-only fallback remaining for reference/testing.
    """
    log = log_fn
    log("Detecting furniture using MoGe depth/normal geometry...")

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

    depth_grad = cv2.Laplacian(points_calibrated[:, :, 2], cv2.CV_32F)
    discontinuity = np.abs(depth_grad) > 0.03
    furniture_mask = (structure_mask & ~wall_mask) | discontinuity

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

        if dims["height"] > room_height * 0.95:
            continue
        if dims["width"] < 0.15 and dims["depth"] < 0.15:
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
                "mask": component_mask,
                "confidence": None,
                "type": classify_region(dims),
            }
        )
        kept += 1

    log(f"Detected {kept} furniture region(s) from MoGe geometry only.")
    return regions


__all__ = [
    "FurnitureDetector",
    "FurnitureDetectorUnavailable",
    "detect_geometric_regions",
]
