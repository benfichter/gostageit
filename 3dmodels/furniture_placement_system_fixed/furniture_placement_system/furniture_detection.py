"""
Simplified Furniture Detection Module
Placeholder that works without SAM for standalone testing
"""

from typing import Dict, List, Optional, Callable
import numpy as np


class FurnitureDetectorUnavailable(RuntimeError):
    """Raised when SAM is not configured for furniture detection."""


class FurnitureDetector:
    """
    Simplified furniture detector that returns empty results when SAM is not available
    """
    
    def __init__(
        self,
        sam_extractor=None,
        log_fn: Optional[Callable[[str], None]] = None,
        min_area_ratio: float = 0.003,
        max_area_ratio: float = 0.4,
        min_points: int = 200,
        max_regions: int = 30,
        max_iou: float = 0.7,
    ) -> None:
        if sam_extractor is None:
            raise FurnitureDetectorUnavailable(
                "SAM extractor is required for furniture detection. "
                "Set SAM_CHECKPOINT_PATH environment variable and install segment-anything."
            )
        self.sam = sam_extractor
        self.log = log_fn or print
    
    def detect(
        self,
        image_rgb: np.ndarray,
        points_calibrated: np.ndarray,
        depth_mask: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Detect furniture in the image"""
        self.log("Furniture detection requires SAM - returning empty list")
        return []


def detect_geometric_regions(
    points_calibrated: np.ndarray,
    mask: np.ndarray,
    normals: Optional[np.ndarray],
    room_height: float,
    log_fn: Optional[Callable[[str], None]] = None,
    min_area_pixels: int = 400,
) -> List[Dict]:
    """Fallback geometric detection without SAM"""
    log = log_fn or print
    log("Using simplified geometric detection (SAM not available)")
    return []
