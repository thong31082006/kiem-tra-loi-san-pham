from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class InspectionResult:
    status: str
    defects: List[str]
    metrics: Dict[str, float | str]
    result_image: np.ndarray
    mask_image: np.ndarray
    edge_image: np.ndarray
    diff_image: np.ndarray
    defect_scores: Dict[str, float] = field(default_factory=dict)
