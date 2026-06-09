from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class CFG:
    """
    Configuration
    Stores training/evaluation paths and basic parameters.
    Outputs:
    - Configuration object with fields:
    - ROOT, TRAIN_DIR, VAL_DIR, TEST_DIR: dataset paths
    - CKPT_PATH: path to checkpoint to load
    - SAVE_CKPT: path to save best checkpoint
    - batch_size, num_workers, img_size, seed: training settings
    """
    ROOT: Path = Path(r"Q:\VisualStudio\ML_Model\InputData3")
    TRAIN_DIR: Path = Path(r"Q:\VisualStudio\ML_Model\InputData3\train")
    VAL_DIR: Path = Path(r"Q:\VisualStudio\ML_Model\InputData3\val")
    TEST_DIR: Path = Path(r"Q:\VisualStudio\ML_Model\InputData3\test")

    CKPT_PATH: str = r"Q:\VisualStudio\ML_Model\ML\best2_ft.pt"

    SAVE_CKPT: str = r"Q:\VisualStudio\ML_Model\ML\best3.pt"

    batch_size: int = 64
    num_workers: int = 4
    img_size: int = 128
    seed: int = 42

class CNN(nn.Module):
    """
    Model
    Simple CNN for binary image classification (outputs a single logit).
    Outputs:
    - nn.Module instance producing logits for BCEWithLogitsLoss.
    """
    def __init__(self):
        """
        Initialize
        Builds the convolutional backbone and classification head.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn4 = nn.BatchNorm2d(64)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 64)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward
        Runs a forward pass and returns raw logits.
        Parameters:
        - x (torch.Tensor): Input batch tensor of shape [B, 3, H, W]
        Outputs:
        - torch.Tensor: Logits tensor of shape [B, 1]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.gap(x)               
        x = x.view(x.size(0), 64)     
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)               
        return x