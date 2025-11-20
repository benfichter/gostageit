from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class FurnitureBox:
    label: str
    pixel_bbox: Dict[str, int]
    corners_3d: np.ndarray
    corners_2d: np.ndarray
    dimensions_m: Dict[str, float]
    dimensions_in: Dict[str, float]


BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),
]
HIDDEN_EDGES = {(4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)}


class Furniture3DVisualizer:
    def __init__(self, meters_to_inches: float = 39.3701) -> None:
        self.meters_to_inches = meters_to_inches
        self.intrinsics: Optional[np.ndarray] = None

    def _estimate_intrinsics(self, image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape[:2]
        focal = max(h, w)
        cx, cy = w / 2, h / 2
        return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])

    def _project_points(self, pts_3d: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        if self.intrinsics is None:
            self.intrinsics = self._estimate_intrinsics(image_shape)
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]
        z = np.clip(pts_3d[:, 2], 1e-6, None)
        x = (pts_3d[:, 0] / z) * fx + cx
        y = (pts_3d[:, 1] / z) * fy + cy
        return np.stack([x, y], axis=1).astype(int)

    def _build_box(self, points: np.ndarray, room_height: float) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
        if len(points) < 10:
            return None
        pts = points.copy()
        # Reduce influence of near-ceiling noise
        pts[:, 1] = np.clip(pts[:, 1], np.min(pts[:, 1]), np.min(pts[:, 1]) + room_height)
        width = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        depth = float(np.max(pts[:, 2]) - np.min(pts[:, 2]))
        height = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        if width <= 0 or depth <= 0 or height <= 0:
            return None
        corners = np.array(
            [
                [np.min(pts[:, 0]), np.min(pts[:, 1]), np.min(pts[:, 2])],
                [np.max(pts[:, 0]), np.min(pts[:, 1]), np.min(pts[:, 2])],
                [np.max(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 2])],
                [np.min(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 2])],
                [np.min(pts[:, 0]), np.max(pts[:, 1]), np.min(pts[:, 2])],
                [np.max(pts[:, 0]), np.max(pts[:, 1]), np.min(pts[:, 2])],
                [np.max(pts[:, 0]), np.max(pts[:, 1]), np.max(pts[:, 2])],
                [np.min(pts[:, 0]), np.max(pts[:, 1]), np.max(pts[:, 2])],
            ]
        )
        dims = {"width": width, "height": height, "depth": depth}
        return corners, dims

    def build_boxes(
        self,
        image_rgb: np.ndarray,
        points_calibrated: np.ndarray,
        furniture_regions: List[Dict],
        room_height: float,
    ) -> List[FurnitureBox]:
        boxes: List[FurnitureBox] = []
        for region in furniture_regions:
            mask = region.get("mask")
            if mask is None:
                continue
            pts = points_calibrated[mask]
            result = self._build_box(pts, room_height=region.get("room_height", room_height))
            if result is None:
                continue
            corners_3d, dims = result
            corners_2d = self._project_points(corners_3d, image_rgb.shape)
            dims_in = {k: v * self.meters_to_inches for k, v in dims.items()}
            boxes.append(
                FurnitureBox(
                    label=region.get("selected_label")
                    or region.get("type", "furniture"),
                    pixel_bbox=region["pixel_bbox"],
                    corners_3d=corners_3d,
                    corners_2d=corners_2d,
                    dimensions_m=dims,
                    dimensions_in=dims_in,
                )
            )
        return boxes

    def _draw_dashed_line(self, canvas, pt1, pt2, color, thickness=1, dash_length=8):
        pt1 = np.array(pt1, dtype=float)
        pt2 = np.array(pt2, dtype=float)
        dist = np.linalg.norm(pt2 - pt1)
        if dist == 0:
            return
        direction = (pt2 - pt1) / dist
        num_dashes = int(dist / dash_length)
        for i in range(0, num_dashes, 2):
            start = pt1 + direction * dash_length * i
            end = pt1 + direction * dash_length * (i + 1)
            cv2.line(canvas, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness, cv2.LINE_AA)

    def render_camera_view(self, image_rgb: np.ndarray, boxes: List[FurnitureBox], path: Path) -> None:
        view = image_rgb.copy()
        colors = [
            (255, 99, 71),
            (60, 179, 113),
            (65, 105, 225),
            (255, 215, 0),
            (238, 130, 238),
            (0, 255, 255),
        ]
        for idx, box in enumerate(boxes):
            color = colors[idx % len(colors)]
            for edge in BOX_EDGES:
                pt1 = tuple(box.corners_2d[edge[0]])
                pt2 = tuple(box.corners_2d[edge[1]])
                if edge in HIDDEN_EDGES:
                    self._draw_dashed_line(view, pt1, pt2, color, 1)
                else:
                    cv2.line(view, pt1, pt2, color, 2, cv2.LINE_AA)
            dims = box.dimensions_in
            text = f"{box.label}: {dims['width']:.1f}\" W × {dims['depth']:.1f}\" D × {dims['height']:.1f}\" H"
            anchor = (box.pixel_bbox["x1"], max(box.pixel_bbox["y1"] - 10, 15))
            cv2.putText(view, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(path), cv2.cvtColor(view, cv2.COLOR_RGB2BGR))

    def render_spec_view(self, boxes: List[FurnitureBox], path: Path) -> None:
        if not boxes:
            return
        cols = 2
        rows = (len(boxes) + cols - 1) // cols
        cell_h = 360
        cell_w = 540
        canvas = np.ones((rows * cell_h + 120, cols * cell_w + 80, 3), dtype=np.uint8) * 255
        gradient = np.linspace(230, 180, canvas.shape[0], dtype=np.float32).reshape(-1, 1, 1)
        canvas = (canvas.astype(np.float32) * (gradient / 255.0)).astype(np.uint8)
        cv2.putText(canvas, "DIMENSIONS  UNIT:INCH", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 80, 80), 3, cv2.LINE_AA)
        for idx, box in enumerate(boxes):
            row = idx // cols
            col = idx % cols
            origin_x = 60 + col * cell_w
            origin_y = 140 + row * cell_h
            scale = 100
            corners = box.corners_3d
            iso_points = []
            angle = np.radians(35)
            for pt in corners:
                x = (pt[0] - pt[2]) * np.cos(angle) * scale
                y = (-pt[1]) * scale - (pt[0] + pt[2]) * np.sin(angle) * scale
                iso_points.append([int(origin_x + x), int(origin_y + y)])
            iso_points = np.array(iso_points)
            for edge in BOX_EDGES:
                pt1 = tuple(iso_points[edge[0]])
                pt2 = tuple(iso_points[edge[1]])
                if edge in HIDDEN_EDGES:
                    self._draw_dashed_line(canvas, pt1, pt2, (255, 255, 255), 2)
                else:
                    cv2.line(canvas, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)
            dims = box.dimensions_in
            cv2.putText(canvas, f"W {dims['width']:.1f}\"", (origin_x - 40, origin_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 70, 70), 2)
            cv2.putText(canvas, f"D {dims['depth']:.1f}\"", (origin_x + 120, origin_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 70, 70), 2)
            cv2.putText(canvas, f"H {dims['height']:.1f}\"", (origin_x - 100, origin_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 70, 70), 2)
            cv2.putText(canvas, box.label, (origin_x - 40, origin_y + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        cv2.imwrite(str(path), canvas)

    def render_comparison(
        self,
        image_rgb: np.ndarray,
        furniture_regions: List[Dict],
        boxes: List[FurnitureBox],
        path: Path,
    ) -> None:
        h, w = image_rgb.shape[:2]
        canvas = np.ones((h, w * 2, 3), dtype=np.uint8) * 255
        left = image_rgb.copy()
        for region in furniture_regions:
            contour = region.get("contour")
            if contour:
                pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(left, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                box = region["pixel_bbox"]
                cv2.rectangle(left, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (0, 0, 255), 2)
        right = image_rgb.copy()
        colors = [
            (255, 99, 71),
            (60, 179, 113),
            (65, 105, 225),
        ]
        for idx, box in enumerate(boxes):
            color = colors[idx % len(colors)]
            for edge in BOX_EDGES:
                pt1 = tuple(box.corners_2d[edge[0]])
                pt2 = tuple(box.corners_2d[edge[1]])
                if edge in HIDDEN_EDGES:
                    self._draw_dashed_line(right, pt1, pt2, color, 1)
                else:
                    cv2.line(right, pt1, pt2, color, 2, cv2.LINE_AA)
        canvas[:, :w] = left
        canvas[:, w:] = right
        cv2.putText(canvas, "2D detections", (w // 2 - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(canvas, "MoGe 3D boxes", (w + w // 2 - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    def export_json(self, boxes: List[FurnitureBox], path: Path) -> None:
        payload = []
        for box in boxes:
            payload.append(
                {
                    "label": box.label,
                    "pixel_bbox": box.pixel_bbox,
                    "dimensions_m": box.dimensions_m,
                    "dimensions_in": box.dimensions_in,
                    "corners_3d": box.corners_3d.tolist(),
                }
            )
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def integrate_3d_visualization(
    image_rgb: np.ndarray,
    points_calibrated: np.ndarray,
    furniture_regions: List[Dict],
    output_dir: Path,
    room_height: float,
) -> Dict[str, Optional[Path]]:
    visualizer = Furniture3DVisualizer()
    boxes = visualizer.build_boxes(
        image_rgb, points_calibrated, furniture_regions, room_height=room_height
    )
    outputs = {"overlay": None, "spec": None, "comparison": None, "data": None}
    if not boxes:
        return outputs
    overlay_path = output_dir / "furniture_3d_overlay.png"
    visualizer.render_camera_view(image_rgb, boxes, overlay_path)
    outputs["overlay"] = overlay_path

    spec_path = output_dir / "furniture_3d_spec.png"
    visualizer.render_spec_view(boxes, spec_path)
    outputs["spec"] = spec_path

    comparison_path = output_dir / "furniture_3d_comparison.png"
    visualizer.render_comparison(image_rgb, furniture_regions, boxes, comparison_path)
    outputs["comparison"] = comparison_path

    data_path = output_dir / "furniture_3d_data.json"
    visualizer.export_json(boxes, data_path)
    outputs["data"] = data_path
    return outputs
