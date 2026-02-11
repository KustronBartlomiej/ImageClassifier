from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from Config import CFG, CNN
from TestConfig import EvalCFG
from ML_val import Evaluator   


class TestEvaluator(Evaluator):
    """
    Testing
    Final test evaluation that uses test_loader and a fixed threshold.
    Parameters:
    - train_cfg (CFG): Training config (paths, checkpoint path, etc.).
    - eval_cfg (EvalCFG): Evaluation config with THRESH_OK and settings.
    - model (CNN | nn.Module): Model used for inference.
    Outputs:
    - TestEvaluator: Evaluator subclass configured for test reporting.
    """

    def collect_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inference
        Collects labels and P(OK) scores from the test_loader.
        Outputs:
        - Tuple[np.ndarray, np.ndarray]:
        - y_true: Ground-truth labels from the test set.
        - y_score: Predicted probabilities P(OK) for each sample.
        """
        y_true_list: list[np.ndarray] = []
        y_score_list: list[np.ndarray] = []

        data_loader = self.loader.test_loader  

        import torch

        with torch.inference_mode():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float()  

                logits = self.model(images).view(-1)     # [B]
                probs_ok = torch.sigmoid(logits)         # P(OK)

                y_true_list.append(labels.cpu().numpy())
                y_score_list.append(probs_ok.cpu().numpy())

        y_true = np.concatenate(y_true_list)
        y_score = np.concatenate(y_score_list)

        print(f"[TEST] collected {len(y_true)} probes from TESTU.")
        return y_true, y_score

    def run(self):
        """
        Report
        Computes metrics and generates plots without threshold tuning.
        Outputs:
        - None (prints metrics, plots confusion matrix and curves).
        """
        y_true, y_score = self.collect_predictions()
        self.compute_and_print_metrics(y_true, y_score)
        self.plot_confusion_matrix(y_true, y_score)
        self.plot_curves(y_true, y_score)


if __name__ == "__main__":
    train_cfg = CFG()
    eval_cfg = EvalCFG()  

    train_cfg.CKPT_PATH = str(eval_cfg.CKPT_PATH)

    model = CNN()

    tester = TestEvaluator(train_cfg, eval_cfg, model)
    tester.run()
