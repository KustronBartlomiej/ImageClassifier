from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,

    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
    log_loss,
)


import matplotlib.pyplot as plt

from Config import CFG, CNN
from TestConfig import EvalCFG
from ML_objects import Loader


class Evaluator:
    """
    Validation
    Evaluates a trained OK/NOK binary classifier on the validation set and reports metrics and plots.
    Parameters:
    - train_cfg (CFG): Training config used to build loaders and transforms.
    - eval_cfg (EvalCFG): Evaluation config with checkpoint path, threshold and plot settings.
    - model (nn.Module): Model instance used for inference.
    Outputs:
    - Evaluator: Configured evaluator ready to run evaluation.
    """

    def __init__(self, train_cfg: CFG, eval_cfg: EvalCFG, model: nn.Module):
        """
        Setup
        Builds loaders, loads checkpoint weights and prepares the model for evaluation.
        Parameters:
        - train_cfg (CFG): Training config used to build loaders and transforms.
        - eval_cfg (EvalCFG): Evaluation config with checkpoint path, threshold and plot settings.
        - model (nn.Module): Model instance used for inference.
        Outputs:
        - None
        """
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.model = model

        # Loader for train/val/test (from ML_objects)
        self.loader = Loader(self.model, self.train_cfg)
        self.loader.set_device()
        self.loader.set_transforms_and_datasets()
        self.loader.build_loaders_and_optim()

        self.device = self.loader.device

        # Load model weights from the checkpoint defined in EvalCFG
        state = torch.load(self.eval_cfg.CKPT_PATH, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print("[EVAL] loaded weights from:", self.eval_cfg.CKPT_PATH)
        print("       missing keys:", missing)
        print("       unexpected  :", unexpected)

        self.model.to(self.device)
        self.model.eval()

        self.class_to_idx = self.loader.test_ds.class_to_idx
        print("[EVAL] class_to_idx:", self.class_to_idx)

    def collect_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inference
        Collects ground-truth labels and P(OK) scores from the validation loader.
        Outputs:
        - Tuple[np.ndarray, np.ndarray]:
        - y_true: ImageFolder class indices (alphabetical: NOK=0, OK=1).
        - y_score: Predicted probabilities P(OK) for each sample.
        """
        y_true_list: list[np.ndarray] = []
        y_score_list: list[np.ndarray] = []

        data_loader = self.loader.val_loader

        with torch.inference_mode():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                logits = self.model(images).view(-1)
                probs_ok = torch.sigmoid(logits)

                y_true_list.append(labels.cpu().numpy())
                y_score_list.append(probs_ok.cpu().numpy())

        y_true = np.concatenate(y_true_list)
        y_score = np.concatenate(y_score_list)

        print(f"[EVAL] collected {len(y_true)} samples from validation/test set.")
        return y_true, y_score

    def _make_labels(self, y_true: np.ndarray):
        """
        Label mapping
        Resolves OK/NOK indices from ImageFolder.class_to_idx and builds a binary target vector (positive=OK).
        Parameters:
        - y_true (np.ndarray): Ground-truth labels as dataset indices.
        Outputs:
        - label_ok (int): Dataset index for class OK.
        - label_nok (int): Dataset index for class NOK.
        - y_true_ok (np.ndarray): Binary vector where 1=OK and 0=NOK.
        """
        label_ok = self.class_to_idx["OK"]
        label_nok = self.class_to_idx["NOK"]

        y_true_ok = (y_true == label_ok).astype(int)
        return label_ok, label_nok, y_true_ok

    def compute_and_print_metrics(self, y_true: np.ndarray, y_score: np.ndarray):
        """
        Metrics
        Computes and prints metrics for a fixed threshold EvalCFG.THRESH_OK where y_score is P(OK).
        Parameters:
        - y_true (np.ndarray): Ground-truth labels as dataset indices.
        - y_score (np.ndarray): Predicted probabilities P(OK).
        Outputs:
        - dict: Dictionary with key metrics, confusion matrix and the used threshold.
        """
        thr = float(self.eval_cfg.THRESH_OK)
        label_ok, label_nok, y_true_ok = self._make_labels(y_true)

        # Binary decision (1=OK, 0=NOK)
        y_pred_ok = (y_score >= thr).astype(int)

        # Map binary predictions back to dataset class indices
        y_pred = np.where(y_pred_ok == 1, label_ok, label_nok).astype(int)

        # Order [OK, NOK] in the confusion matrix
        labels_order = [label_ok, label_nok]
        cm = confusion_matrix(y_true, y_pred, labels=labels_order)

        # Confusion matrix layout for [OK, NOK] (rows=true, cols=pred)
        # TP_OK = cm[0,0], FN_OK = cm[0,1], FP_OK = cm[1,0] (NOK->OK), TN_OK = cm[1,1]
        TP = int(cm[0, 0])
        FN = int(cm[0, 1])
        FP = int(cm[1, 0])
        TN = int(cm[1, 1])

        acc = accuracy_score(y_true_ok, y_pred_ok)
        bacc = balanced_accuracy_score(y_true_ok, y_pred_ok)

        # Metrics important for OK/NOK:
        # recall_nok = NOK detection rate = TN / (TN + FP)
        recall_ok = TP / max(1, TP + FN)
        precision_ok = TP / max(1, TP + FP)
        recall_nok = TN / max(1, TN + FP)
        precision_nok = TN / max(1, TN + FN)

        f1_ok = 2 * precision_ok * recall_ok / max(1e-12, (precision_ok + recall_ok))
        f1_nok = 2 * precision_nok * recall_nok / max(1e-12, (precision_nok + recall_nok))

        # Critical error rates
        # NOK->OK rate = FP / (FP + TN)
        nok_as_ok_rate = FP / max(1, FP + TN)
        ok_as_nok_rate = FN / max(1, TP + FN)

        # Threshold-independent / probabilistic metrics
        auc_roc = roc_auc_score(y_true_ok, y_score)
        auc_pr = average_precision_score(y_true_ok, y_score)

        mcc = matthews_corrcoef(y_true_ok, y_pred_ok)
        kappa = cohen_kappa_score(y_true_ok, y_pred_ok)

        # Numerical stability for log_loss
        eps = 1e-7
        p = np.clip(y_score, eps, 1 - eps)
        ll = log_loss(y_true_ok, p, labels=[0, 1])
        brier = brier_score_loss(y_true_ok, p)

        print("\n=== CONFUSION MATRIX (rows=true, cols=pred), order: [OK, NOK] ===")
        print(cm)
        print(f"TP_OK={TP}  FN_OK={FN}  FP_OK(NOK->OK)={FP}  TN_OK={TN}")

        print("\n=== METRICS (threshold P(OK) = {:.3f}) ===".format(thr))
        print(f"Accuracy             : {acc:.4f}")
        print(f"Balanced Accuracy    : {bacc:.4f}")
        print(f"Precision OK         : {precision_ok:.4f}")
        print(f"Recall OK            : {recall_ok:.4f}")
        print(f"F1 OK                : {f1_ok:.4f}")
        print(f"Precision NOK        : {precision_nok:.4f}")
        print(f"Recall NOK           : {recall_nok:.4f}")
        print(f"F1 NOK               : {f1_nok:.4f}")
        print(f"NOK->OK rate (FP/(FP+TN)) : {nok_as_ok_rate:.4f}")
        print(f"OK->NOK rate (FN/(TP+FN)) : {ok_as_nok_rate:.4f}")
        print(f"ROC AUC (OK)         : {auc_roc:.4f}")
        print(f"PR AUC  (OK)         : {auc_pr:.4f}")
        print(f"MCC                  : {mcc:.4f}")
        print(f"Cohen's kappa        : {kappa:.4f}")
        print(f"Log loss             : {ll:.4f}")
        print(f"Brier score          : {brier:.4f}")

        print("\n=== CLASSIFICATION REPORT (sklearn) ===")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=[label_ok, label_nok],
                target_names=["OK", "NOK"],
                digits=4,
            )
        )

        return {
            "thr": thr,
            "cm": cm,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "precision_ok": precision_ok,
            "recall_ok": recall_ok,
            "f1_ok": f1_ok,
            "precision_nok": precision_nok,
            "recall_nok": recall_nok,
            "f1_nok": f1_nok,
            "nok_as_ok_rate": nok_as_ok_rate,
            "ok_as_nok_rate": ok_as_nok_rate,
            "roc_auc": auc_roc,
            "pr_auc": auc_pr,
            "mcc": mcc,
            "kappa": kappa,
            "log_loss": ll,
            "brier": brier,
        }

    def plot_confusion_matrix(self, y_true: np.ndarray, y_score: np.ndarray):
        """
        Plot
        Plots the confusion matrix for the current threshold.
        Parameters:
        - y_true (np.ndarray): Ground-truth labels as dataset indices.
        - y_score (np.ndarray): Predicted probabilities P(OK).
        Outputs:
        - None (shows a matplotlib figure).
        """
        thr = float(self.eval_cfg.THRESH_OK)
        label_ok, label_nok, _ = self._make_labels(y_true)

        y_pred_ok = (y_score >= thr).astype(int)
        y_pred = np.where(y_pred_ok == 1, label_ok, label_nok).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[label_ok, label_nok])

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["OK", "NOK"],
        )
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion matrix (threshold P(OK) = {thr:.2f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        plt.show()

    def plot_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray):
        """
        Plot
        Plots the ROC curve for OK as the positive class.
        Parameters:
        - y_true (np.ndarray): Ground-truth labels as dataset indices.
        - y_score (np.ndarray): Predicted probabilities P(OK).
        Outputs:
        - None (shows a matplotlib figure).
        """
        _, _, y_true_ok = self._make_labels(y_true)
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_true_ok, y_score, ax=ax, name="OK vs NOK")
        ax.set_title("ROC curve")
        ax.grid(True)
        fig.tight_layout()
        plt.show()

    def plot_pr_curve(self, y_true: np.ndarray, y_score: np.ndarray):
        """
        Plot
        Plots the Precision-Recall curve for OK as the positive class.
        Parameters:
        - y_true (np.ndarray): Ground-truth labels as dataset indices.
        - y_score (np.ndarray): Predicted probabilities P(OK).
        Outputs:
        - None (shows a matplotlib figure).
        """
        _, _, y_true_ok = self._make_labels(y_true)
        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_predictions(y_true_ok, y_score, ax=ax, name="OK vs NOK")
        ax.set_title("Precision-Recall curve")
        ax.grid(True)
        fig.tight_layout()
        plt.show()

    def plot_score_histogram(self, y_true: np.ndarray, y_score: np.ndarray, bins: int = 30):
        """
        Plot
        Plots the distribution of P(OK) separately for true OK and true NOK samples.
        Parameters:
        - y_true (np.ndarray): Ground-truth labels as dataset indices.
        - y_score (np.ndarray): Predicted probabilities P(OK).
        - bins (int): Number of histogram bins.
        Outputs:
        - None (shows a matplotlib figure).
        """
        label_ok, label_nok, _ = self._make_labels(y_true)

        s_ok = y_score[y_true == label_ok]
        s_nok = y_score[y_true == label_nok]

        fig, ax = plt.subplots()
        ax.hist(s_ok, bins=bins, alpha=0.6, label="true OK")
        ax.hist(s_nok, bins=bins, alpha=0.6, label="true NOK")
        ax.set_title("Histogram of P(OK) by true class")
        ax.set_xlabel("P(OK)")
        ax.set_ylabel("Number of samples")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        plt.show()

    def plot_threshold_tuning(self, y_true: np.ndarray, y_score: np.ndarray):
        """
        Threshold tuning
        Plots precision/recall/F1 as a function of the decision threshold.
        Parameters:
        - y_true (np.ndarray): Ground-truth labels as dataset indices.
        - y_score (np.ndarray): Predicted probabilities P(OK).
        Outputs:
        - None (shows a matplotlib figure).
        """
        if not getattr(self.eval_cfg, "PLOT_CURVES", True):
            return

        thr_used = float(self.eval_cfg.THRESH_OK)
        _, _, y_true_ok = self._make_labels(y_true)

        precisions, recalls, thresh = precision_recall_curve(y_true_ok, y_score)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

        best_idx = int(np.argmax(f1_scores))
        best_threshold = float(thresh[best_idx]) if best_idx < len(thresh) else thr_used

        print(
            f"Best threshold (max F1 on this set): {best_threshold:.3f}  "
            f"(F1={f1_scores[best_idx]:.4f})"
        )
        print(f"Used EvalCFG.THRESH_OK threshold    : {thr_used:.3f}")

        fig, ax = plt.subplots()
        ax.plot(thresh, f1_scores[:-1], label="F1-score")
        ax.plot(thresh, precisions[:-1], "--", label="Precision")
        ax.plot(thresh, recalls[:-1], "--", label="Recall")

        ax.axvline(best_threshold, linestyle=":", label=f"Best F1 thr = {best_threshold:.3f}")
        ax.axvline(thr_used, linestyle="--", label=f"Used thr = {thr_used:.3f}")

        ax.set_xlabel("Decision threshold P(OK)")
        ax.set_ylabel("Metric value")
        ax.set_title("Threshold tuning")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        plt.show()

    def plot_curves(self, y_true: np.ndarray, y_score: np.ndarray):
        """
        Plots
        Generates ROC, PR and score histogram plots if EvalCFG.PLOT_CURVES is enabled.
        Parameters:
        - y_true (np.ndarray): Ground-truth labels as dataset indices.
        - y_score (np.ndarray): Predicted probabilities P(OK).
        Outputs:
        - None (shows matplotlib figures).
        """
        if not self.eval_cfg.PLOT_CURVES:
            return

        self.plot_roc_curve(y_true, y_score)
        self.plot_pr_curve(y_true, y_score)
        self.plot_score_histogram(y_true, y_score)

    def run(self):
        """
        Run
        Runs the full validation pipeline: predictions, metrics, confusion matrix, curves and threshold tuning.
        Outputs:
        - None (prints metrics and shows plots).
        """
        y_true, y_score = self.collect_predictions()
        self.compute_and_print_metrics(y_true, y_score)
        self.plot_confusion_matrix(y_true, y_score)
        self.plot_curves(y_true, y_score)
        self.plot_threshold_tuning(y_true, y_score)


if __name__ == "__main__":
    train_cfg = CFG()
    eval_cfg = EvalCFG()

    train_cfg.CKPT_PATH = str(eval_cfg.CKPT_PATH)

    model = CNN()
    evaluator = Evaluator(train_cfg, eval_cfg, model)
    evaluator.run()
