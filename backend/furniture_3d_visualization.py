from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.decomposition import PCA


BOX_EDGES: Sequence[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),
]


@dataclass
class BoundingBox3D:
    corners_3d: np.ndarray
    corners_2d: np.ndarray
    center: np.ndarray
    dimensions: Dict[str, float]


class Furniture3DVisualizer:
    def __init__(self, intrinsics: Optional[np.ndarray] = None) -> None:
        self.intrinsics = intrinsics

    def _estimate_intrinsics(self, image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape[:2]
        focal = max(h, w)
        cx, cy = w / 2, h / 2
        return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])

    def _project(self, pts_3d: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        if self.intrinsics is None:
            self.intrinsics = self._estimate_intrinsics(image_shape)
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]
        z = np.clip(pts_3d[:, 2], 1e-6, None)
        x = (pts_3d[:, 0] / z) * fx + cx
        y = (pts_3d[:, 1] / z) * fy + cy
        pts_2d = np.stack([x, y], axis=1)
        return pts_2d.astype(int)

    def _axis_aligned_box(self, pts: np.ndarray) -> BoundingBox3D:
        min_pt = pts.min(axis=0)
        max_pt = pts.max(axis=0)
        corners = np.array([
            [min_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]],
            [min_pt[0], max_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], max_pt[2]],
            [min_pt[0], max_pt[1], max_pt[2]],
        ])
        dims = {
            "width": float(max_pt[0] - min_pt[0]),
            "height": float(max_pt[1] - min_pt[1]),
            "depth": float(max_pt[2] - min_pt[2]),
        }
        return BoundingBox3D(
            corners_3d=corners,
            corners_2d=None,
            center=(min_pt + max_pt) / 2,
            dimensions=dims,
        )

    def _oriented_box(self, pts: np.ndarray) -> BoundingBox3D:
        center = pts.mean(axis=0)
        centered = pts - center
        pca = PCA(n_components=3)
        rotated = pca.fit_transform(centered)
        min_pt = rotated.min(axis=0)
        max_pt = rotated.max(axis=0)
        corners = np.array([
            [min_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]],
            [min_pt[0], max_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], max_pt[2]],
            [min_pt[0], max_pt[1], max_pt[2]],
        ])
        corners = pca.inverse_transform(corners) + center
        dims = {
            "width": float(max_pt[0] - min_pt[0]),
            "height": float(max_pt[1] - min_pt[1]),
            "depth": float(max_pt[2] - min_pt[2]),
        }
        return BoundingBox3D(
            corners_3d=corners,
            corners_2d=None,
            center=center,
            dimensions=dims,
        )

    def _create_box(self, pts: np.ndarray, method: str) -> Optional[BoundingBox3D]:
        if pts.size == 0:
            return None
        if method == "axis":
            return self._axis_aligned_box(pts)
        return self._oriented_box(pts)

    def draw_boxes(
        self,
        image_rgb: np.ndarray,
        furniture_regions: List[Dict],
        points_calibrated: np.ndarray,
        method: str = "oriented",
    ) -> Tuple[np.ndarray, List[BoundingBox3D]]:
        vis = image_rgb.copy()
        boxes: List[BoundingBox3D] = []
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        for idx, region in enumerate(furniture_regions):
            mask = region.get("mask")
            if mask is None:
                continue
            region_pts = points_calibrated[mask]
            if len(region_pts) < 10:
                continue
            bbox = self._create_box(region_pts, method)
            if bbox is None:
                continue
            bbox.corners_2d = self._project(bbox.corners_3d, image_rgb.shape)
            boxes.append(bbox)
            color = colors[idx % len(colors)]
            for edge in BOX_EDGES:
                pt1 = tuple(bbox.corners_2d[edge[0]])
                pt2 = tuple(bbox.corners_2d[edge[1]])
                cv2.line(vis, pt1, pt2, color, 2)
            label = region.get("type", f"item {idx+1}")
            dims = bbox.dimensions
            text = f"{label}: {dims['width']:.2f}×{dims['depth']:.2f}×{dims['height']:.2f}m"
            top = bbox.corners_2d[bbox.corners_2d[:, 1].argmin()].astype(int)
            cv2.putText(
                vis,
                text,
                (top[0], max(15, top[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        return vis, boxes

    def export_json(self, boxes: List[BoundingBox3D], furniture_regions: List[Dict], path: Path) -> None:
        payload = []
        for bbox, region in zip(boxes, furniture_regions):
            payload.append(
                {
                    "type": region.get("type"),
                    "dimensions_m": bbox.dimensions,
                    "center_xyz": bbox.center.tolist(),
                    "corners_xyz": bbox.corners_3d.tolist(),
                }
            )
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def integrate_3d_visualization(
    image_rgb: np.ndarray,
    points_calibrated: np.ndarray,
    furniture_regions: List[Dict],
    output_dir: Path,
    method: str = "oriented",
) -> Tuple[Optional[Path], Optional[Path], List[BoundingBox3D]]:
    if not furniture_regions:
        return None, None, []
    visualizer = Furniture3DVisualizer()
    vis_image, boxes = visualizer.draw_boxes(
        image_rgb=image_rgb,
        furniture_regions=furniture_regions,
        points_calibrated=points_calibrated,
        method=method,
    )
    vis_path = output_dir / "furniture_3d_boxes.png"
    cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    data_path = output_dir / "furniture_3d_data.json"
    visualizer.export_json(boxes, furniture_regions, data_path)
    return vis_path, data_path, boxes
