from dataclasses import dataclass
from pathlib import Path

@dataclass
class EvalCFG:
    """
    Configuration
    Used in ML_test
    """
    THRESH_OK: float = 0.57 #0.65
    CKPT_PATH: Path = Path(r"ML/best3.pt")
    PLOT_CURVES: bool = True
