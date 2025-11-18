"""
Simplified SAM Subject Extractor Placeholder
Works without SAM for standalone testing
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
import numpy as np


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
    Placeholder for SAM subject extraction.
    Returns None when SAM is not available.
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        model_type: str = "vit_h",
        device: Optional[str] = None,
        log_fn: Optional[LogFn] = None,
        min_area_ratio: float = 0.02,
        max_area_ratio: float = 0.95,
    ) -> None:
        self.log = log_fn or (lambda msg: None)
        self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path.exists():
            raise SubjectExtractorUnavailable(
                f"SAM checkpoint not found at {self.checkpoint_path}. "
                "Download from: https://github.com/facebookresearch/segment-anything"
            )
        
        # Would initialize SAM here if available
        self.log(f"SAM placeholder initialized (actual SAM not loaded)")
    
    def extract_primary_subject(self, image_rgb: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Extract primary subject - returns None in placeholder"""
        return None
    
    def save_subject_assets(
        self,
        image_rgb: np.ndarray,
        output_dir: Path,
        stem: str,
    ) -> Optional[SubjectExtractionResult]:
        """Save subject assets - returns None in placeholder"""
        return None
    
    def generate_masks(self, image_rgb: np.ndarray) -> List[Dict]:
        """Generate masks - returns empty list in placeholder"""
        return []
