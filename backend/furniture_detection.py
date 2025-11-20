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
    """Raised when Segment Anything masks are required but unavailable."""


class FurnitureDetector:
    """
    Prefer SAM masks for crisp furniture segmentation; fall back to MoGe-only geometry.
    """

    def __init__(
        self,
        log_fn: Callable[[str], None] = default_log,
        sam_extractor: Optional[SamSubjectExtractor] = None,
        min_area_ratio: float = 0.003,
        max_area_ratio: float = 0.35,
        min_points: int = 200,
        max_regions: int = 30,
        max_iou: float = 0.65,
        geom_min_area_pixels: int = 400,
        geom_min_point_count: int = 150,
        smoothing_kernel: int = 5,
        dilation_iterations: int = 1,
    ) -> None:
        self.sam = sam_extractor
        self.log = log_fn or default_log
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_points = max(1, min_points)
        self.max_regions = max_regions
        self.max_iou = max_iou
        self.geom_min_area_pixels = geom_min_area_pixels
        self.geom_min_point_count = geom_min_point_count
        self.smoothing_kernel = max(1, smoothing_kernel)
        self.dilation_iterations = max(0, dilation_iterations)

    def detect(
        self,
        image_rgb: np.ndarray,
        points_calibrated: np.ndarray,
        depth_mask: np.ndarray,
        normals: Optional[np.ndarray],
        room_height: float,
    ) -> List[Dict]:
        if self.sam is not None:
            try:
                return self._detect_with_sam(
                    image_rgb=image_rgb,
                    points_calibrated=points_calibrated,
                    depth_mask=depth_mask,
                    room_height=room_height,
                )
            except FurnitureDetectorUnavailable as exc:
                self.log(f"SAM furniture detector unavailable: {exc}")

        return detect_geometric_regions(
            points_calibrated=points_calibrated,
            mask=depth_mask,
            normals=normals,
            room_height=room_height,
            log_fn=self.log,
            min_area_pixels=self.geom_min_area_pixels,
            min_point_count=self.geom_min_point_count,
            smoothing_kernel=self.smoothing_kernel,
            dilation_iterations=self.dilation_iterations,
        )

    def _detect_with_sam(
        self,
        image_rgb: np.ndarray,
        points_calibrated: np.ndarray,
        depth_mask: np.ndarray,
        room_height: float,
    ) -> List[Dict]:
        masks = self.sam.generate_masks(image_rgb) if self.sam else None
        if not masks:
            raise FurnitureDetectorUnavailable("SAM returned no masks.")

        image_h, image_w = image_rgb.shape[:2]
        image_area = image_h * image_w
        min_area = image_area * self.min_area_ratio
        max_area = image_area * self.max_area_ratio

        valid_mask = depth_mask.astype(bool)
        valid_mask &= np.isfinite(points_calibrated[:, :, 0])
        valid_mask &= np.isfinite(points_calibrated[:, :, 1])
        valid_mask &= np.isfinite(points_calibrated[:, :, 2])

        masks = sorted(masks, key=lambda m: m.get("area", 0), reverse=True)
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

            bbox = self._contour_to_bbox(np.array(contour), image_w, image_h)
            if self._touches_border(bbox, image_w, image_h):
                continue

            object_mask = segmentation & valid_mask
            if object_mask.sum() < self.min_points:
                continue

            dims_center = self._compute_dimensions(points_calibrated, object_mask)
            if dims_center is None:
                continue

            dims = dims_center["dimensions"]
            if dims["height"] > room_height * 0.98:
                continue

            if any(
                self._bbox_iou(bbox, existing) > self.max_iou
                for existing in selected_bboxes
            ):
                continue

            regions.append(
                {
                    "region_id": region_counter,
                    "pixel_bbox": bbox,
                    "mask": object_mask,
                    "contour": contour,
                    "dimensions_m": dims,
                    "center": dims_center["center"],
                    "point_count": int(object_mask.sum()),
                    "confidence": mask_data.get("predicted_iou"),
                    "type": classify_region(dims),
                }
            )
            selected_bboxes.append(bbox)
            region_counter += 1
            if self.max_regions and len(regions) >= self.max_regions:
                break

        if not regions:
            raise FurnitureDetectorUnavailable("SAM produced masks but none passed filters.")

        self.log(f"Detected {len(regions)} furniture region(s) via SAM.")
        return regions

    @staticmethod
    def _extract_primary_contour(segmentation: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        contours, _ = cv2.findContours(
            segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 10:
            return None
        contour = contour.reshape(-1, 2)
        return [(int(x), int(y)) for x, y in contour]

    @staticmethod
    def _compute_dimensions(
        points_calibrated: np.ndarray,
        mask: np.ndarray,
    ) -> Optional[Dict[str, Dict[str, float]]]:
        object_points = points_calibrated[mask]
        if len(object_points) < 20:
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
        return {"dimensions": dims, "center": center}

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

    @staticmethod
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


def detect_geometric_regions(
    points_calibrated: np.ndarray,
    mask: np.ndarray,
    normals: Optional[np.ndarray],
    room_height: float,
    log_fn: Callable[[str], None] = default_log,
    min_area_pixels: int = 400,
    min_point_count: int = 50,
    smoothing_kernel: int = 5,
    dilation_iterations: int = 1,
) -> List[Dict]:
    """
    Segment volumetric furniture regions from MoGe geometry without SAM assistance.
    """
    log = log_fn
    log("Detecting furniture using MoGe depth/normal geometry...")

    valid_mask = mask.astype(bool)
    valid_mask &= np.isfinite(points_calibrated[:, :, 0])
    valid_mask &= np.isfinite(points_calibrated[:, :, 1])
    valid_mask &= np.isfinite(points_calibrated[:, :, 2])
    if not np.any(valid_mask):
        log("MoGe output contains no valid points for detection.")
        return []

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
        wall_mask = np.zeros_like(structure_mask, dtype=bool)

    depth_channel = np.where(
        np.isfinite(points_calibrated[:, :, 2]), points_calibrated[:, :, 2], 0.0
    ).astype(np.float32)
    depth_grad = cv2.Laplacian(depth_channel, cv2.CV_32F)
    discontinuity = (np.abs(depth_grad) > 0.03) & valid_mask

    furniture_mask = (structure_mask & ~wall_mask) | discontinuity

    furniture_uint8 = (furniture_mask * 255).astype(np.uint8)
    if smoothing_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (smoothing_kernel, smoothing_kernel)
        )
        furniture_uint8 = cv2.morphologyEx(furniture_uint8, cv2.MORPH_CLOSE, kernel)
        furniture_uint8 = cv2.morphologyEx(furniture_uint8, cv2.MORPH_OPEN, kernel)
        if dilation_iterations > 0:
            furniture_uint8 = cv2.dilate(
                furniture_uint8, kernel, iterations=dilation_iterations
            )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        furniture_uint8, connectivity=8
    )

    regions: List[Dict] = []
    region_counter = 1

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area_pixels:
            continue

        component_mask = labels == label
        object_points = points_calibrated[component_mask]
        if len(object_points) < min_point_count:
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
        bbox = {"x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h}

        mask_uint8 = (component_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_points: Optional[List[Tuple[int, int]]]
        if contours:
            contour_points = (
                max(contours, key=cv2.contourArea).reshape(-1, 2).astype(int).tolist()
            )
        else:
            contour_points = None

        regions.append(
            {
                "region_id": region_counter,
                "pixel_bbox": bbox,
                "dimensions_m": dims,
                "center": {
                    "x": float(np.mean(x_vals)),
                    "y": float(np.mean(y_vals)),
                    "z": float(np.mean(z_vals)),
                },
                "point_count": int(len(object_points)),
                "mask": component_mask,
                "contour": contour_points,
                "confidence": None,
                "type": classify_region(dims),
            }
        )
        region_counter += 1

    log(f"Detected {len(regions)} furniture region(s) from MoGe geometry only.")
    return regions


__all__ = [
    "FurnitureDetector",
    "detect_geometric_regions",
]
