from ML_objects import CNN
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as tvF
from dataclasses import dataclass
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt  


@dataclass
class CFG:
    """
    Configuration class for prediction and visualization

    CKPT_PATH: Path to a trained checkpoint with weights (e.g., best3.pt is the newest one)
    IMG_SIZE: Image size in pixels used as model input
    MEAN: Mean for input normalization (R, G, B)
    STD: Standard Deviation for input normalization (R, G, B)

    THRESH_OK: Decision Threshold for P(OK), 0.65 as default and 0.57 after tuning
    TTA_VIEWS: number of test-time views for evaluation

    device: Torch device used for inference (if available)
    """
    CKPT_PATH: Path = Path(r"ML/best3.pt")
    IMG_SIZE: int = 128
    MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
    STD: tuple[float, float, float] = (0.5, 0.5, 0.5)

    THRESH_OK: float = 0.57  #0.65
    TTA_VIEWS: int = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OKTarget:
    """
    Grad-CAM target for decision OK
    Takes:
    model_output: Model logits tensor of shape [B] or [B,1]
    Returns:
    Flattened tensor [B]
    """
    def __call__(self, model_output: torch.Tensor):
        return model_output.view(-1)


class NOKTarget:
    """
    Grad-CAM target for decision NOK
    Takes:
    model_output: Model logits tensor of shape [B] or [B,1]
    Returns:
    Flattened tensor [B]
    """
    def __call__(self, model_output: torch.Tensor):
        return -model_output.view(-1)


class PredictPhoto:
    """
    Single image prediction without Grad-CAM

    self.config: CFG object
    self.model: Torch model used for logits computation
    """
    def __init__(self, cfg: CFG, model: nn.Module):
        self.config = cfg
        self.model = model.to(cfg.device)

    def make_tfm(self):
        """
        builds preprocessing transformation
        """
        return transforms.Compose([
            transforms.Resize((self.config.IMG_SIZE, self.config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(self.config.MEAN, self.config.STD),
        ])

    def tta_p_ok(self, pil_img: Image.Image, tfm, n: int | None = None) -> tuple[float, float]:
        """
        Compute P(OK) using simple test-time augmentation (TTA).
        Parameters:
        pil_img: Input image as PIL.Image (RGB).
        tfm: Preprocessing transform (typically from make_tfm()).
        n: Number of views to use; defaults to cfg.TTA_VIEWS.
        Returns:
        (p_ok_tta, p_ok_base):
            p_ok_tta: Median P(OK) across TTA views.
            p_ok_base: P(OK) for the original (non-augmented) view.
        """
        if n is None:
            n = self.config.TTA_VIEWS

        views = [
            pil_img,                                        
            tvF.hflip(pil_img),                             # flip
            tvF.rotate(pil_img, 3),                         # rot +3°
            tvF.rotate(pil_img, -3),                        # rot -3°
            pil_img.resize(
                (self.config.IMG_SIZE + 2, self.config.IMG_SIZE + 2)
            ).resize((self.config.IMG_SIZE, self.config.IMG_SIZE)),  # micro-blur
        ]
        w, h = pil_img.size
        dw, dh = int(0.02 * w), int(0.02 * h)
        views.append(
            tvF.crop(
                pil_img,
                dh,
                dw,
                max(1, h - 2 * dh),
                max(1, w - 2 * dw),
            )
        )

        probs: list[float] = []
        with torch.inference_mode():
            for im in views[:max(1, min(n, len(views)))]:
                x = tfm(im).unsqueeze(0).to(self.config.device)
                logit = self.model(x)
                p_ok = torch.sigmoid(logit).item()
                probs.append(p_ok)

        p_ok_base = float(probs[0])
        p_ok_med = float(torch.tensor(probs).median().item())
        return p_ok_med, p_ok_base

    def describe_model(self) -> str:
        """
        Creates a short, readable model summary.
        Returns:
        Multi-line string describing conv layers and parameter counts.
        """
        lines: list[str] = []
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        for name in ["conv1", "conv2", "conv3", "conv4"]:
            if hasattr(self.model, name):
                m = getattr(self.model, name)
                if isinstance(m, nn.Conv2d):
                    lines.append(
                        f"  {name}: in={m.in_channels} -> out={m.out_channels}, "
                        f"k={tuple(m.kernel_size)}, s={tuple(m.stride)}, "
                        f"p={tuple(m.padding)}, d={tuple(m.dilation)}"
                    )

        lines.append("  GAP: AdaptiveAvgPool2d(1x1)")
        lines.append(f"  fc1: 64 -> 64, dropout p={self.model.drop.p}")
        lines.append("  fc2: 64 -> 1 (logit OK)")
        lines.append(f"  #params: {total:,} (trainable: {trainable:,})")

        return "Model params:\n" + "\n".join(lines)

    def predict_one(self):
        """Run an interactive prediction for a single image.
        Opens a file dialog, loads checkpoint weights, runs TTA inference,
        prints final label and probabilities to the console.
        Returns:
            None
        """
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(
            title="Choose image",
            filetypes=(
                ("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ),
        )
        root.destroy()
        if not image_path:
            print("No file chosen.")
            return

        state = torch.load(self.config.CKPT_PATH, map_location=self.config.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.config.device)
        self.model.eval()

        pil = Image.open(image_path).convert("RGB")
        tfm = self.make_tfm()

        p_ok_tta, p_ok_base = self.tta_p_ok(pil, tfm, n=self.config.TTA_VIEWS)

        p_nok_tta = 1.0 - p_ok_tta
        p_nok_base = 1.0 - p_ok_base

        label_str = "OK" if p_ok_tta >= self.config.THRESH_OK else "NOK"

        print("\n=== result ===")
        print(f"Decision : {label_str}  @thr_OK={self.config.THRESH_OK:.2f}")
        print(f"Path : {image_path}")
        print(f"P(OK)   : {p_ok_tta:.4f}  (base={p_ok_base:.4f})")
        print(f"P(NOK)  : {p_nok_tta:.4f} (base={p_nok_base:.4f})")
        print(self.describe_model())


class GradCam:
    """Grad-CAM visualization for a single image.
    Loads model weights, runs inference, and produces a heatmap overlay
    to inspect which regions influenced the decision.
    Parameters:
        cfg: Inference/visualization configuration.
        model: Torch model used for inference and Grad-CAM.
    """
    def __init__(self, cfg: CFG, model: nn.Module):
        self.config = cfg
        self.model = model.to(self.config.device)
        state = torch.load(self.config.CKPT_PATH, map_location=self.config.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.state = state
        self.img_path: str | None = None

    def select_image(self) -> str:
        """Open a file dialog and select an image for Grad-CAM.
        Returns:
            Selected image path as a string.
        Raises:
            SystemExit: If no file is selected.
        """
        root = tk.Tk()
        root.withdraw()
        img_path = filedialog.askopenfilename(
            title="Choose image Grad-CAM",
            filetypes=(
                ("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ),
        )
        root.destroy()
        if not img_path:
            raise SystemExit("No image chosen.")
        self.img_path = img_path
        return img_path

    def make_tfm(self):
        """Build the preprocessing transform for Grad-CAM inference."""
        return transforms.Compose([
            transforms.Resize((self.config.IMG_SIZE, self.config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(self.config.MEAN, self.config.STD),
        ])

    def prepare_model_and_input(self):
        """Load weights, set eval mode, and prepare model input.
        Returns:
            (inp, rgb):
                inp: Input tensor of shape [1, 3, IMG_SIZE, IMG_SIZE] on device.
                rgb: Float numpy array of shape [IMG_SIZE, IMG_SIZE, 3] in [0, 1].
        Raises:
            RuntimeError: If no image path was selected (self.img_path is None).
        """
        self.model.load_state_dict(self.state, strict=False)
        self.model.to(self.config.device)
        self.model.eval()

        if self.img_path is None:
            raise RuntimeError("missing path, call select_image().")

        pil = Image.open(self.img_path).convert("RGB")
        tfm = self.make_tfm()
        inp = tfm(pil).unsqueeze(0).to(self.config.device)  # [1,3,H,W]

        rgb = np.asarray(
            pil.resize((self.config.IMG_SIZE, self.config.IMG_SIZE)),
            dtype=np.float32,
        ) / 255.0  # [H,W,3] [0,1]

        return inp, rgb

    def compute_cam(self, inp, rgb):
        """Compute prediction and Grad-CAM heatmap.
        Parameters:
            inp: Input tensor [1, 3, H, W] on device.
            rgb: Numpy image [H, W, 3] in [0, 1] for visualization.
        Returns:
            (vis, grayscale, p_ok, p_nok, label_str):
                vis: RGB uint8 image with CAM overlay.
                grayscale: CAM heatmap [H, W] in [0, 1].
                p_ok: Probability of OK (sigmoid(logit)).
                p_nok: Probability of NOK (1 - p_ok).
                label_str: "OK" or "NOK" based on cfg.THRESH_OK.
        """
        with torch.inference_mode():
            logit = self.model(inp)  # [1,1]
            p_ok = torch.sigmoid(logit).item()
        p_nok = 1.0 - p_ok
        label_str = "OK" if p_ok >= self.config.THRESH_OK else "NOK"

        target_layer = self.model.conv4 if hasattr(self.model, "conv4") else self.model.conv3
        cam = GradCAM(model=self.model, target_layers=[target_layer])

        if label_str == "OK":
            target = OKTarget()
        else:
            target = NOKTarget()

        grayscale = cam(
            input_tensor=inp,
            targets=[target],
            eigen_smooth=True,
        )[0]  # [H,W] [0,1]
        vis = show_cam_on_image(rgb, grayscale, use_rgb=True, image_weight=0.5)

        return vis, grayscale, p_ok, p_nok, label_str

    def visualize(self, vis, grayscale, p_ok, p_nok, label_str):
        """Display the Grad-CAM overlay and heatmap using matplotlib.
        Parameters:
            vis: Overlay image (RGB) to show.
            grayscale: Heatmap array [H, W] in [0, 1].
            p_ok: Predicted P(OK).
            p_nok: Predicted P(NOK).
            label_str: Final label string ("OK" / "NOK").
        Returns:
            None
        Raises:
            RuntimeError: If self.img_path is None.
        """
        if self.img_path is None:
            raise RuntimeError("missing path.")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        axes[0].imshow(vis)
        axes[0].axis("off")
        axes[0].set_title("Image + Grad-CAM")

        im = axes[1].imshow(grayscale, cmap="jet", vmin=0.0, vmax=1.0)
        axes[1].axis("off")
        axes[1].set_title("Map (Grad-CAM)")

        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Influence")
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(["Ignored", "Medium", "Decisive"])

        title = (
            f"{self.img_path}\n"
            f"Label: {label_str} | "
            f"P(OK)={p_ok:.3f}  P(NOK)={p_nok:.3f}  "
            f"@thr_OK={self.config.THRESH_OK:.2f}  "
        )
        fig.suptitle(title, fontsize=9)
        plt.tight_layout()
        plt.show()

        print("\n=== INFO (Grad-CAM) ===")
        print(f"Path : {self.img_path}")
        print(f"Label   : {label_str}")
        print(f"P(OK)   : {p_ok:.4f}")
        print(f"P(NOK)  : {p_nok:.4f}")
        print(f"thr_OK  : {self.config.THRESH_OK:.2f}")

    def run(self):
        self.select_image()
        inp, rgb = self.prepare_model_and_input()
        vis, grayscale, p_ok, p_nok, label_str = self.compute_cam(inp, rgb)
        self.visualize(vis, grayscale, p_ok, p_nok, label_str)


if __name__ == "__main__":
    """Full Grad-CAM workflow: select image -> prepare -> compute -> visualize.
    May also call PredictPhoto to show result without Grad-CAM
    """
    cfg = CFG()
    model = CNN()

    grad_cam = GradCam(cfg, model)
    grad_cam.run()
