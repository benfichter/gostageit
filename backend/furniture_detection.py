from __future__ import annotations

import cv2
import numpy as np
from typing import Callable, Dict, List, Optional


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
    if height > 0.8 and width > 1.2 and depth > 0.4:
        return "sofa"
    if height < 0.6 and footprint < 1.5:
        return "table"
    if width < 0.8 and depth < 0.8 and height > 0.6:
        return "chair"
    return "furniture"


def detect_geometric_regions(
    points_calibrated: np.ndarray,
    mask: np.ndarray,
    normals: Optional[np.ndarray],
    room_height: float,
    log_fn: Callable[[str], None] = default_log,
    min_area_pixels: int = 400,
) -> List[Dict]:
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


__all__ = ["detect_geometric_regions"]
