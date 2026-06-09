# OK/NOK Image Classifier (PyTorch)

Binary image classification pipeline: **OK** vs **NOK** (defective part detection).

The pipeline covers the full workflow: dataset splitting, offline augmentation,
model training, validation, test evaluation, and single-image prediction
with optional Grad-CAM visualization.

---

## Requirements

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key packages: `torch==2.10.0`, `torchvision==0.25.0`, `scikit-learn`,
`opencv-python`, `matplotlib`, `grad-cam`.

---

## Project structure

```
.
├── ML/
│   ├── Config.py          # training configuration (CFG) and CNN model
│   ├── ML_objects.py      # Loader + Trainer (training pipeline)
│   ├── ML_val.py          # validation evaluator
│   ├── ML_test.py         # test evaluator
│   ├── TestConfig.py      # evaluation config (EvalCFG)
│   ├── Predict_objects.py # single-image inference and Grad-CAM
│   └── best3.pt           # trained checkpoint (default)
├── Augment/
│   ├── AugmentConfig.py   # offline augmentation config and operations
│   └── main.py            # runs offline augmentation
├── Split/
│   └── split.py           # group-aware dataset splitter
├── data/                  # raw images
├── results/               # example result images
└── requirements.txt
```

All scripts within a folder use relative imports and must be run
from their respective subdirectory (e.g. `cd ML && python ML_objects.py`).

---

## Dataset

### Folder format

`torchvision.datasets.ImageFolder` is used, so images must be organized as:

```
InputData3/
  train/
    OK/
    NOK/
  val/
    OK/
    NOK/
  test/
    OK/
    NOK/
```

### Splitting

`Split/split.py` builds train/val/test splits with **group-aware shuffling**
to prevent leakage between original NOK images and their augmented variants:

- OK images: each file is its own group.
- NOK images: group id = filename prefix before the first `_`
  (original and all its augmented variants share a group and always end up in the same split).

Default split ratios:

| Set   | Ratio |
|-------|-------|
| train | 70%   |
| val   | 15%   |
| test  | 15%   |

The split produces a `manifest.csv` with `path`, `label_name`, `label_idx`,
`group_id` and `split` columns for every image.

```bash
cd Split
python split.py
```

> Paths in `Split/split.py → CFG` are Windows absolute paths (`Q:\...`).
> Adjust them to your local machine before running.

---

## Model

Defined in `ML/Config.py`. Takes a **128×128 RGB** image and outputs a single logit.

### Architecture

| Layer | Type                        | Channels  | Kernel | Stride | Dilation |
|-------|-----------------------------|-----------|--------|--------|----------|
| conv1 | Conv2d + BatchNorm + ReLU   | 3 → 16    | 5×5    | 2      | 1        |
| conv2 | Conv2d + BatchNorm + ReLU   | 16 → 32   | 3×3    | 2      | 1        |
| conv3 | Conv2d + BatchNorm + ReLU   | 32 → 64   | 3×3    | 2      | 1        |
| conv4 | Conv2d + BatchNorm + ReLU   | 64 → 64   | 3×3    | 1      | **2**    |
| GAP   | AdaptiveAvgPool2d(1×1)      | 64 → 64   | —      | —      | —        |
| fc1   | Linear + Dropout(0.5)       | 64 → 64   | —      | —      | —        |
| fc2   | Linear                      | 64 → 1    | —      | —      | —        |

`conv4` uses **dilation=2** to widen the receptive field without increasing parameters.

### Decision rule

```
P(OK) = sigmoid(logit)
label = "OK"  if P(OK) >= THRESH_OK
        "NOK" otherwise
```

Default `THRESH_OK = 0.57` (tuned on the validation set; original default was 0.65).

---

## Training

Run from the `ML/` directory:

```bash
cd ML
python ML_objects.py
```

### Hyperparameters

| Parameter       | Value                                                      |
|-----------------|------------------------------------------------------------|
| Input size      | 128 × 128                                                  |
| Batch size      | 64                                                         |
| Optimizer       | Adam — lr=1e-4, weight_decay=7e-4                         |
| LR scheduler    | ReduceLROnPlateau — factor=0.5, patience=5, min_lr=3e-6   |
| Max epochs      | 20                                                         |
| Early stopping  | patience=5 (monitors val loss)                            |
| Loss            | BCEWithLogitsLoss + label smoothing ε=0.05                |
| Grad clipping   | max_norm=1.0                                               |
| Class balance   | WeightedRandomSampler                                      |

The best checkpoint (lowest val loss) is saved to `SAVE_CKPT` from `ML/Config.py`.

### Training transforms

Applied to the training set only:

| Transform            | Parameters                                                   |
|----------------------|--------------------------------------------------------------|
| RandomResizedCrop    | 128×128, scale=(0.92, 1.0), ratio=(0.98, 1.02)              |
| RandomHorizontalFlip | p=0.5                                                        |
| RandomAffine         | degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02)       |
| ColorJitter          | brightness=0.05, contrast=0.05                              |
| GaussianBlur         | kernel=3, sigma=(0.1, 1.0)                                  |
| ToTensor             | —                                                            |
| PerimeterErasing     | p=0.15, band=0.12 — custom: zeros a rectangular strip along one randomly chosen edge |
| Normalize            | mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)                  |

### Validation / test transforms

| Transform  | Parameters                                  |
|------------|---------------------------------------------|
| Resize     | 128×128                                     |
| CenterCrop | 128                                         |
| ToTensor   | —                                           |
| Normalize  | mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)  |

> Paths in `ML/Config.py → CFG` are Windows absolute paths. Adjust them before running.

---

## Evaluation

### Validation

```bash
cd ML
python ML_val.py
```

Loads the checkpoint from `ML/TestConfig.py → EvalCFG.CKPT_PATH`.
Computes accuracy, balanced accuracy, precision, recall, F1, ROC AUC, PR AUC,
MCC, Cohen's κ, Brier score, and per-class error rates.
Plots: confusion matrix, ROC curve, PR curve, score histogram,
and a threshold-tuning curve.

### Test

```bash
cd ML
python ML_test.py
```

Same metrics on the test set using the fixed threshold from `EvalCFG.THRESH_OK`.

---

## Single-image prediction

`ML/Predict_objects.py` provides two modes: **Grad-CAM visualization** and
**simple TTA prediction**. Both load the checkpoint from `CFG.CKPT_PATH`
(default: `ML/best3.pt`, relative path — run from the `ML/` directory).

### Grad-CAM (default)

```bash
cd ML
python Predict_objects.py
```

A file dialog opens. After selecting an image:

- the model runs inference
- a matplotlib window displays the Grad-CAM heatmap overlaid on the image
  and the raw activation map (jet colormap: 0 = ignored → 1 = decisive)
- the decision, P(OK) and P(NOK) are printed to the console

### Simple prediction with TTA (PredictPhoto)

**Option A — from Python:**

```python
from Predict_objects import CFG, PredictPhoto
from ML_objects import CNN

cfg = CFG()
model = CNN()
PredictPhoto(cfg, model).predict_one()
```

**Option B — replace the `__main__` block in `Predict_objects.py`:**

```python
if __name__ == "__main__":
    cfg = CFG()
    model = CNN()
    PredictPhoto(cfg, model).predict_one()
```

A file dialog opens, the model runs TTA inference, and the result is printed:

```
=== result ===
Decision : OK  @thr_OK=0.57
P(OK)    : 0.8431  (base=0.8102)
P(NOK)   : 0.1569  (base=0.1898)
```

`P(OK)` is the TTA median; `base` is the raw value for the unmodified image.

### Test-time augmentation (TTA)

`PredictPhoto` evaluates **6 views** per image and reports the **median P(OK)**:

| View       | Description                           |
|------------|---------------------------------------|
| original   | unmodified image                      |
| hflip      | horizontal flip                       |
| rot +3°    | rotation +3°                          |
| rot −3°    | rotation −3°                          |
| micro-blur | resize +2px then back to 128×128      |
| crop       | 2% border removed (each side)         |

The number of views is controlled by `CFG.TTA_VIEWS` (default: 6).

---

## Offline augmentation

`Augment/AugmentConfig.py` defines operations to generate additional NOK images
from originals. Run this before splitting the dataset.

```bash
cd Augment
python main.py
```

Each operation saves a new file with a tag suffix. Available operations:

- **Rotations:** `rot+3`, `rot-3` (reflected borders)
- **Shifts + zoom:** `shift_left`, `shift_right`, `shift_up`, `shift_down` (×0.90),
  `shift_big_left`, `shift_big_right` (×0.88), `zoom_center_0.90`, `zoom_center_0.80`
- **Crops:** `crop_top`, `crop_bottom` (×0.85), `crop_left`, `crop_right` (×0.82),
  `crop_top_strong`, `crop_bottom_strong` (×0.75)
- **Brightness:** `bright+10`, `bright-10`, `bright+5`, `bright-5`
- **Contrast:** `contrast+10`, `contrast-10`
- **Noise — global:** `bg+07`, `bg+12`
- **Noise — edge strip:** `bgp+08`, `bgp+15`
- **Flip:** `hflip`
- **Blur:** `blur`
- **Combos:** `hflip_rot+3`, `hflip_rot-3`

> Paths in `Augment/AugmentConfig.py → AugmentConfig` are Windows absolute paths.
> Adjust them before running.

---

## Notes

- Most config paths are Windows absolute paths (`Q:\...`).
  Adjust them to your local machine before running any script.
- All scripts use relative imports — run them from their subdirectory
  (`ML/`, `Augment/`, `Split/`).
- The inference checkpoint path is relative (`ML/best3.pt`);
  run inference scripts from the `ML/` directory.
- Grad-CAM requires `grad-cam` (included in `requirements.txt`).
