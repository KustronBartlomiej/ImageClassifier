from Config import CFG, CNN

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

import matplotlib.pyplot as plt


class PerimeterErasing(nn.Module):
    """
    Augmentation
    Randomly zeros a rectangular band region on one image side.
    Parameters:
    - p (float): Probability of applying the augmentation.
    - band (float): Fraction of min(H, W) used to define max erase size.
    Outputs:
    - None (callable nn.Module)
    """
    def __init__(self, p: float = 0.15, band: float = 0.12):
        super().__init__()
        self.p = p
        self.band = band

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Transform
        Applies perimeter erasing to a single image tensor with probability p.
        Parameters:
        - img (torch.Tensor): Image tensor of shape [C, H, W].
        Outputs:
        - torch.Tensor: Image tensor of shape [C, H, W] (possibly modified).
        """
        if random.random() > self.p:
            return img
        # img: tensor [C,H,W]
        C, H, W = img.shape
        b = max(2, int(min(H, W) * self.band))

        side = random.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            h = random.randint(int(b * 0.4), b)
            w = random.randint(int(W * 0.3), int(W * 0.9))
            x1 = random.randint(0, W - w)
            y1 = 0 if side == "top" else H - h
        else:
            w = random.randint(int(b * 0.4), b)
            h = random.randint(int(H * 0.3), int(H * 0.9))
            y1 = random.randint(0, H - h)
            x1 = 0 if side == "left" else W - w

        img[:, y1:y1 + h, x1:x1 + w] = 0.0
        return img


class Loader:
    """
    Loader
    Prepares device, datasets, dataloaders, and training components.
    Parameters:
    - model (nn.Module): Model to train/evaluate.
    - cfg (CFG): Project configuration with paths and hyperparameters.
    Outputs:
    - Loader: Initialized object with datasets, loaders and optim components.
    """
    def __init__(self, model: nn.Module, cfg: CFG):
        """Set attributes to None; call prepare() to initialize them."""
        self.model = model
        self.config = cfg

        self.device: torch.device | None = None
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.img_size = self.config.img_size

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.epochs = None

    def set_device(self):
        """
        Device
        Selects CUDA if available, otherwise CPU.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            props = torch.cuda.get_device_properties(0)
            print(f"Using CUDA: {props.name}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

    def set_transforms_and_datasets(self):
        """
        Datasets
        Builds torchvision transforms and ImageFolder datasets for train/val/test.
        """
        train_tfms = transforms.Compose([
            transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.92, 1.0), ratio=(0.98, 1.02)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            PerimeterErasing(p=0.15, band=0.12),
            transforms.Normalize(self.mean, self.std),
        ])

        val_test_tfms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        self.train_ds = datasets.ImageFolder(self.config.TRAIN_DIR, transform=train_tfms)
        self.val_ds = datasets.ImageFolder(self.config.VAL_DIR, transform=val_test_tfms)
        self.test_ds = datasets.ImageFolder(self.config.TEST_DIR, transform=val_test_tfms)

        self.class_to_idx = self.train_ds.class_to_idx
        self.train_targets = self.train_ds.targets
        np.bincount(self.train_targets, minlength=len(self.class_to_idx))

    def build_loaders_and_optim(self):
        """
        Setup
        Creates dataloaders, loads checkpoint weights if present, and builds
        criterion, optimizer, and scheduler.
        """
        class_counts = np.bincount(self.train_targets, minlength=len(self.class_to_idx))
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = [class_weights[label] for label in self.train_targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        pin_mem = False

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=pin_mem,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=pin_mem,
        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=pin_mem,
        )

        # model + weights
        self.model.to(self.device)

        try:
            state = torch.load(self.config.CKPT_PATH, map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print("Checkpoint loaded. Missing:", missing, "| Unexpected:", unexpected)
        except FileNotFoundError:
            print(f"Checkpoint not found: {self.config.CKPT_PATH} – starting with random weights.")

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=7e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=3e-6,
        )
        self.epochs = 20

    def prepare(self):
        """
        Prepare
        Runs device selection, dataset setup, and loader/optimizer creation.
        Outputs:
        - Loader: Returns self for chaining.
        """
        self.set_device()
        self.set_transforms_and_datasets()
        self.build_loaders_and_optim()
        return self  


class Trainer:
    """
    Training
    Runs the training loop, validation, checkpointing, and history tracking.
    Parameters:
    - cfg (CFG): Configuration with checkpoint path and save path.
    - model (nn.Module): Model to train.
    - loader (Loader): Prepared Loader with data and optim components.
    Outputs:
    - Trainer: Initialized trainer object.
    """
    def __init__(self, cfg, model: nn.Module, loader: Loader):
        self.config = cfg
        self.model = model
        self.device = loader.device

        self.train_loader = loader.train_loader
        self.val_loader = loader.val_loader
        self.test_loader = loader.test_loader

        self.criterion = loader.criterion
        self.optimizer = loader.optimizer
        self.scheduler = loader.scheduler
        self.epochs = loader.epochs

        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []

        self.best_val_loss = float("inf")
        self.patience = 5
        self.stale_epochs = 0
        self.smooth_eps = 0.05

    def train_model(self):
        """
        Train
        Trains for multiple epochs with validation, LR scheduling and early stopping.
        """
        for epoch in range(self.epochs):
            self.model.train()
            train_loss_sum = 0.0
            train_num = 0
            train_correct = 0

            for images, labels in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(self.device, non_blocking=True)  # [B,1]

                self.optimizer.zero_grad()

                logits = self.model(images)  # [B,1]

                targets_smooth = labels * (1.0 - self.smooth_eps) + 0.5 * self.smooth_eps

                loss = self.criterion(logits, targets_smooth)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                train_loss_sum += loss.item() * labels.size(0)
                train_num += labels.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                train_correct += (preds == labels).sum().item()

            avg_train_loss = train_loss_sum / train_num
            train_acc = train_correct / train_num

            self.model.eval()
            val_loss_sum = 0.0
            val_num = 0
            val_correct = 0

            with torch.inference_mode():
                for images, labels in self.val_loader:
                    images = images.to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)

                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                    val_loss_sum += loss.item() * labels.size(0)
                    val_num += labels.size(0)

                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()
                    val_correct += (preds == labels).sum().item()

            avg_val_loss = val_loss_sum / val_num
            val_acc = val_correct / val_num

            # best weights 
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.config.SAVE_CKPT)
                self.stale_epochs = 0
            else:
                self.stale_epochs += 1
                if self.stale_epochs >= self.patience:
                    print("Early stopping.")
                    break

            # LR update
            self.scheduler.step(avg_val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # saving
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(
                f"Epoch [{epoch+1}/{self.epochs}] "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

    def plot_history(self):
        """
        Plot
        Plots train/val loss and accuracy history using matplotlib.
        Outputs:
        - None (shows figures)
        """
        # Loss
        plt.figure()
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="train loss")
        plt.plot(range(1, len(self.val_losses) + 1),   self.val_losses,   label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Accuracy
        plt.figure()
        plt.plot(range(1, len(self.train_accs) + 1), self.train_accs, label="train acc")
        plt.plot(range(1, len(self.val_accs) + 1),   self.val_accs,   label="val acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Train / Val Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    cfg = CFG()
    model = CNN()

    loader = Loader(model, cfg).prepare()

    trainer = Trainer(cfg, model, loader)
    trainer.train_model()

    trainer.plot_history()
