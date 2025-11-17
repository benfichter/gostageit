from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

try:  # Optional dependency
    import clip  # type: ignore
except ImportError:  # pragma: no cover - optional
    clip = None

try:  # Optional dependency
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # type: ignore
except ImportError:  # pragma: no cover - optional
    SamAutomaticMaskGenerator = None
    sam_model_registry = None


def default_logger(message: str) -> None:
    print(message)


@dataclass
class FurnitureDetectorConfig:
    min_area_pixels: int = 350
    min_span_m: float = 0.18
    max_height_ratio: float = 0.92
    sam_iou_threshold: float = 0.25
    use_sam: bool = False
    sam_checkpoint: Optional[str] = None
    sam_model_type: str = "vit_h"
    use_clip: bool = False
    clip_labels: Sequence[str] = (
        "sofa",
        "sectional sofa",
        "loveseat",
        "accent chair",
        "dining chair",
        "coffee table",
        "side table",
        "console table",
        "bed",
        "dresser",
        "nightstand",
        "floor lamp",
        "table lamp",
        "bookshelf",
        "television",
        "bench",
        "stool",
        "plant",
    )
    clip_device: str = "cpu"
    sam_max_masks: int = 60
    # internal
    _sam_generator: Optional[SamAutomaticMaskGenerator] = field(init=False, default=None, repr=False)
    _clip_model: Optional[torch.nn.Module] = field(init=False, default=None, repr=False)
    _clip_preprocess: Optional[Callable] = field(init=False, default=None, repr=False)


class FurnitureDetector:
    def __init__(self, config: FurnitureDetectorConfig, log_fn: Callable[[str], None] = default_logger):
        self.config = config
        self.log = log_fn
        self._init_sam()
        self._init_clip()

    def _init_sam(self) -> None:
        if not self.config.use_sam:
            self.log("SAM refinement disabled (FURNITURE_USE_SAM=false).")
            return
        if SamAutomaticMaskGenerator is None or sam_model_registry is None:
            self.log("segment-anything not installed; skipping SAM refinement.")
            self.config.use_sam = False
            return
        if not self.config.sam_checkpoint or not Path(self.config.sam_checkpoint).exists():
            self.log("SAM checkpoint not found; skipping SAM refinement.")
            self.config.use_sam = False
            return
        self.log("Loading SAM checkpoint...")
        sam = sam_model_registry[self.config.sam_model_type](checkpoint=self.config.sam_checkpoint)
        sam.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.config._sam_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            box_nms_thresh=0.6,
            output_mode="binary_mask",
        )
        self.log("SAM loaded successfully.")

    def _init_clip(self) -> None:
        if not self.config.use_clip:
            self.log("CLIP classification disabled (FURNITURE_USE_CLIP=false).")
            return
        if clip is None:
            self.log("OpenAI CLIP not installed; skipping classification.")
            self.config.use_clip = False
            return
        self.log("Loading CLIP model...")
        model, preprocess = clip.load("ViT-B/32", device=self.config.clip_device)
        self.config._clip_model = model.eval()
        self.config._clip_preprocess = preprocess
        with torch.no_grad():
            text_tokens = clip.tokenize(list(self.config.clip_labels)).to(self.config.clip_device)
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        self._clip_text_features = text_features
        self.log("CLIP ready.")

    def detect(
        self,
        image_rgb: np.ndarray,
        points_calibrated: np.ndarray,
        mask: np.ndarray,
        normals: Optional[np.ndarray],
        room_height: float,
    ) -> List[Dict]:
        regions = self._geometric_candidates(points_calibrated, mask, normals, room_height)
        if not regions:
            return []
        if self.config.use_sam and self.config._sam_generator is not None:
            regions = self._refine_with_sam(image_rgb, points_calibrated, mask, regions)
        if self.config.use_clip and self.config._clip_model is not None:
            self._classify_with_clip(image_rgb, regions)

        # Drop heavy mask arrays before returning
        for region in regions:
            region.pop("pixel_mask", None)
        return regions

    def _geometric_candidates(
        self,
        points_calibrated: np.ndarray,
        mask: np.ndarray,
        normals: Optional[np.ndarray],
        room_height: float,
    ) -> List[Dict]:
        log = self.log
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
        min_span = self.config.min_span_m

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < self.config.min_area_pixels:
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

            if dims["height"] > room_height * self.config.max_height_ratio:
                continue
            if dims["width"] < min_span and dims["depth"] < min_span:
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
                    "pixel_mask": component_mask,
                    "confidence": 0.45,
                }
            )
            kept += 1

        log(f"Detected {kept} furniture region(s) via geometry")
        return regions

    def _refine_with_sam(
        self,
        image_rgb: np.ndarray,
        points_calibrated: np.ndarray,
        valid_mask: np.ndarray,
        regions: List[Dict],
    ) -> List[Dict]:
        generator = self.config._sam_generator
        if generator is None:
            return regions
        self.log("Running SAM refinement...")
        masks = generator.generate(image_rgb)
        masks = sorted(masks, key=lambda m: m.get("area", 0), reverse=True)
        kept_masks = masks[: self.config.sam_max_masks]
        H, W, _ = image_rgb.shape
        valid_mask_bool = valid_mask.astype(bool)

        for region in regions:
            region_mask = region.get("pixel_mask")
            if region_mask is None:
                continue
            best_iou = 0.0
            best_mask = None
            for sam_mask in kept_masks:
                mask_arr = sam_mask["segmentation"]
                if mask_arr.shape != region_mask.shape:
                    continue
                intersection = np.logical_and(region_mask, mask_arr).sum()
                union = np.logical_or(region_mask, mask_arr).sum()
                if union == 0:
                    continue
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_mask = mask_arr
            if best_mask is None or best_iou < self.config.sam_iou_threshold:
                continue

            combined_mask = best_mask & valid_mask_bool
            if not combined_mask.any():
                continue
            dims, center = self._dimensions_from_mask(points_calibrated, combined_mask)
            if dims is None:
                continue
            ys, xs = np.where(combined_mask)
            x1 = int(np.min(xs))
            y1 = int(np.min(ys))
            x2 = int(np.max(xs))
            y2 = int(np.max(ys))
            region.update(
                {
                    "pixel_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "dimensions_m": dims,
                    "center": center,
                    "pixel_mask": combined_mask,
                    "confidence": max(region.get("confidence", 0.5), best_iou),
                }
            )

        return regions

    def _dimensions_from_mask(
        self, points_calibrated: np.ndarray, mask: np.ndarray
    ) -> Optional[tuple[Dict[str, float], Dict[str, float]]]:
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
        return dims, center

    def _classify_with_clip(self, image_rgb: np.ndarray, regions: List[Dict]) -> None:
        model = self.config._clip_model
        preprocess = self.config._clip_preprocess
        if model is None or preprocess is None:
            return
        device = self.config.clip_device
        for region in regions:
            bbox = region["pixel_bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            image = Image.fromarray(crop)
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                probs = (image_features @ self._clip_text_features.T).softmax(dim=-1)
            best_idx = int(torch.argmax(probs))
            confidence = float(probs[0, best_idx].item())
            region["type"] = self.config.clip_labels[best_idx]
            region["confidence"] = max(region.get("confidence", 0.0), confidence)


__all__ = ["FurnitureDetector", "FurnitureDetectorConfig"]
