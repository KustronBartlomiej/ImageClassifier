from ML_objects import CNN
from Predict_objects import PredictPhoto

from pathlib import Path
from dataclasses import dataclass
import csv
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


@dataclass
class CFG:
    """
    Configuration for batch prediction.
    CKPT_PATH:      Path to trained checkpoint weights.
    IMG_DIR:        Directory containing images to predict.
    CSV_PATH:       Output CSV file path.
    IMG_SIZE:       Model input size in pixels.
    MEAN:           Normalization mean (R, G, B).
    STD:            Normalization standard deviation (R, G, B).
    THRESH_OK:      Decision threshold for P(OK).
    TTA_VIEWS:      Number of test-time augmentation views.
    IMG_EXTS:       Accepted image file extensions.
    EXPECTED_LABEL: Known ground-truth label for all images in IMG_DIR.
                    Set to "OK" or "NOK" to enable the correct column; "N/A" disables it.
    device:         Torch device used for inference.
    """
    CKPT_PATH: Path = Path(r"ML/best4.pt")
    IMG_DIR:   Path = Path(r"data/Split/val/NOK")
    CSV_PATH:  Path = Path(r"results/predictions.csv")

    IMG_SIZE: int = 128
    MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
    STD:  tuple[float, float, float] = (0.5, 0.5, 0.5)

    THRESH_OK: float = 0.57
    TTA_VIEWS: int = 6

    IMG_EXTS:       tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    EXPECTED_LABEL: str             = "NOK"  # "OK", "NOK", or "N/A"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictMany:
    """
    Batch prediction
    Runs TTA inference on all images in a directory and saves results to a CSV file.
    Parameters:
    - cfg (CFG): Batch prediction configuration.
    - model (nn.Module): Torch model used for inference.
    """
    def __init__(self, cfg: CFG, model: nn.Module):
        """Load checkpoint once and prepare model for inference."""
        self.config = cfg
        self.predictor = PredictPhoto(cfg, model)

        state = torch.load(cfg.CKPT_PATH, map_location=cfg.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.predictor.model.load_state_dict(state, strict=False)
        self.predictor.model.to(cfg.device)
        self.predictor.model.eval()
        print("[BATCH] Checkpoint loaded from:", cfg.CKPT_PATH)

    def list_images(self) -> list[Path]:
        """
        List images
        Returns a sorted list of image paths from IMG_DIR.
        Outputs:
        - list[Path]: Image paths with accepted extensions.
        """
        p = Path(self.config.IMG_DIR)
        return sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in self.config.IMG_EXTS
        )

    def predict_all(self) -> list[dict]:
        """
        Predict all
        Runs TTA inference on every image in IMG_DIR.
        Outputs:
        - list[dict]: One dict per image with prediction results.
        """
        images = self.list_images()
        if not images:
            print(f"[WARN] No images found in: {self.config.IMG_DIR}")
            return []

        tfm = self.predictor.make_tfm()
        expected = self.config.EXPECTED_LABEL
        results = []

        for img_path in tqdm(images, desc="[BATCH] Predicting", unit="img"):
            pil = Image.open(img_path).convert("RGB")
            p_ok_tta, _ = self.predictor.tta_p_ok(pil, tfm, n=self.config.TTA_VIEWS)
            p_nok_tta = 1.0 - p_ok_tta
            decision  = "OK" if p_ok_tta >= self.config.THRESH_OK else "NOK"
            correct   = (decision == expected) if expected != "N/A" else "N/A"

            results.append({
                "filename": img_path.name,
                "path":     str(img_path),
                "expected": expected,
                "p_ok":     round(p_ok_tta,  4),
                "p_nok":    round(p_nok_tta, 4),
                "thresh_ok": self.config.THRESH_OK,
                "decision": decision,
                "correct":  correct,
            })

        return results

    def save_csv(self, results: list[dict]) -> None:
        """
        Save CSV
        Writes prediction results to the CSV file at CSV_PATH.
        Parameters:
        - results (list[dict]): Prediction results from predict_all().
        """
        out = Path(self.config.CSV_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["filename", "path", "expected", "p_ok", "p_nok", "thresh_ok", "decision", "correct"]
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"[DONE] {len(results)} images → {out}")

    def run(self) -> None:
        """
        Run
        Executes batch prediction and saves results to CSV.
        """
        results = self.predict_all()
        if results:
            self.save_csv(results)


if __name__ == "__main__":
    cfg = CFG()
    model = CNN()

    predictor = PredictMany(cfg, model)
    predictor.run()
