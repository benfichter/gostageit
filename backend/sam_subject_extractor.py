from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

LogFn = Callable[[str], None]


class SubjectExtractorUnavailable(RuntimeError):
    """Raised when Segment Anything resources are not available."""


@dataclass
class SubjectExtractionResult:
    mask_path: Path
    overlay_path: Path
    cutout_path: Path
    bbox: Dict[str, int]
    area: int


class SamSubjectExtractor:
    """
    Minimal wrapper around Segment Anything for Apple-style “lift subject” cutouts.
    Loads a SAM checkpoint once and reuses it to extract the dominant subject mask.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        model_type: str = "vit_h",
        device: Optional[torch.device] = None,
        log_fn: Optional[LogFn] = None,
        min_area_ratio: float = 0.02,
        max_area_ratio: float = 0.95,
    ) -> None:
        self.log = log_fn or (lambda msg: None)
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise SubjectExtractorUnavailable(
                f"SAM checkpoint not found at {self.checkpoint_path}"
            )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_type = model_type
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:
            raise SubjectExtractorUnavailable(
                "segment-anything package is not installed. "
                "Install it via `pip install segment-anything @ git+https://github.com/facebookresearch/segment-anything.git`."
            ) from exc

        self.log(
            f"Loading SAM ({model_type}) from {self.checkpoint_path} on {self.device}..."
        )
        sam = sam_model_registry[model_type](checkpoint=str(self.checkpoint_path))
        sam.to(device=self.device)
        sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        self.log("SAM ready for subject extraction.")

    def _select_subject_mask(
        self,
        masks: list[Dict],
        image_shape: tuple[int, int],
    ) -> Optional[Dict]:
        if not masks:
            return None
        image_area = image_shape[0] * image_shape[1]
        min_area = image_area * self.min_area_ratio
        max_area = image_area * self.max_area_ratio

        candidates = [
            m for m in masks if min_area <= m["area"] <= max_area and m["bbox"][2] > 5
        ]
        if not candidates:
            candidates = masks

        cx = image_shape[1] / 2
        cy = image_shape[0] / 2

        def score(mask: Dict) -> float:
            x, y, w, h = mask["bbox"]
            center_x = x + w / 2
            center_y = y + h / 2
            dist = np.hypot(center_x - cx, center_y - cy)
            area_score = mask["area"] / image_area
            iou_score = mask.get("predicted_iou", 0.0)
            stability = mask.get("stability_score", 0.0)
            # Prefer large, centered, high-quality masks.
            return area_score * 2.0 + iou_score + stability - 0.001 * dist

        return max(candidates, key=score)

    def extract_primary_subject(
        self, image_rgb: np.ndarray
    ) -> Optional[Dict[str, np.ndarray]]:
        masks = self.mask_generator.generate(image_rgb)
        subject = self._select_subject_mask(masks, image_rgb.shape[:2])
        if subject is None:
            return None
        segmentation = subject["segmentation"].astype(bool)
        bbox = subject["bbox"]
        return {"mask": segmentation, "bbox": bbox, "area": int(subject["area"])}

    def save_subject_assets(
        self,
        image_rgb: np.ndarray,
        output_dir: Path,
        stem: str,
    ) -> Optional[SubjectExtractionResult]:
        subject = self.extract_primary_subject(image_rgb)
        if subject is None:
            self.log("SAM subject extraction produced no masks.")
            return None

        output_dir.mkdir(parents=True, exist_ok=True)

        mask = subject["mask"]
        mask_path = output_dir / f"{stem}_subject_mask.png"
        overlay_path = output_dir / f"{stem}_subject_overlay.png"
        cutout_path = output_dir / f"{stem}_subject.png"

        self._write_mask(mask, mask_path)
        self._write_overlay(image_rgb, mask, overlay_path)
        self._write_cutout(image_rgb, mask, cutout_path)

        bbox = subject["bbox"]
        bbox_dict = {
            "x": int(bbox[0]),
            "y": int(bbox[1]),
            "width": int(bbox[2]),
            "height": int(bbox[3]),
        }
        return SubjectExtractionResult(
            mask_path=mask_path,
            overlay_path=overlay_path,
            cutout_path=cutout_path,
            bbox=bbox_dict,
            area=subject["area"],
        )

    def _write_mask(self, mask: np.ndarray, path: Path) -> None:
        mask_u8 = (mask.astype(np.uint8) * 255)
        cv2.imwrite(str(path), mask_u8)

    def _write_overlay(
        self, image_rgb: np.ndarray, mask: np.ndarray, path: Path
    ) -> None:
        overlay = image_rgb.copy()
        overlay[~mask] = (overlay[~mask] * 0.25).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
        cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    def _write_cutout(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        path: Path,
    ) -> None:
        alpha = (mask.astype(np.uint8) * 255)[..., None]
        rgba = np.concatenate([image_rgb, alpha], axis=2)
        Image.fromarray(rgba).save(path)

    def generate_masks(self, image_rgb: np.ndarray) -> List[Dict]:
        """
        Generate all SAM masks for downstream processing (e.g., furniture detection).
        """
        return self.mask_generator.generate(image_rgb)


__all__ = [
    "SamSubjectExtractor",
    "SubjectExtractorUnavailable",
    "SubjectExtractionResult",
]
