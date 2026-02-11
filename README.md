# OK/NOK Image Classifier (PyTorch)

This repository contains a small PyTorch pipeline for **binary image classification**:
**OK** vs **NOK** (defect / not OK).
It includes dataset splitting (group-aware), offline augmentation generation, model training, evaluation (val + test),
and single-image prediction with optional Grad-CAM visualization.

---

## Model overview

The model is a lightweight CNN that outputs **one logit** (later converted with `sigmoid` to **P(OK)**):

- 4× convolution blocks with BatchNorm + ReLU
- `conv4` uses **dilation=2**
- Global Average Pooling (AdaptiveAvgPool2d 1×1)
- Fully-connected head: `64 -> 64 -> 1` with **Dropout(p=0.5)**
- Training loss: `BCEWithLogitsLoss`

Typical decision rule:

- compute `P(OK) = sigmoid(logit)`
- predict **OK** if `P(OK) >= THRESH_OK`, otherwise **NOK**

---

## Project structure (Python files)

### `Config.py`
- Training configuration (`CFG`) with dataset paths, batch size, image size, checkpoint paths.
- Defines the CNN model (`CNN`).

### `ML_objects.py`
- **Training pipeline**:
  - `Loader`: sets device, builds torchvision transforms, loads `ImageFolder` datasets and dataloaders.
  - Uses `WeightedRandomSampler` to reduce class imbalance impact.
  - `Trainer`: training loop, early stopping, LR scheduling, history plots.

Run:
```bash
python ML_objects.py
```

### `ML_val.py`
- **Validation evaluator**:
  - Loads checkpoint from `EvalCFG.CKPT_PATH`.
  - Computes metrics (accuracy, balanced accuracy, ROC AUC, PR AUC, MCC, etc.).
  - Plots confusion matrix and curves (ROC / PR / histogram), plus threshold tuning plot.

Run:
```bash
python ML_val.py
```

### `ML_test.py`
- **Test evaluator** (final reporting on `test_loader`), using a fixed threshold `THRESH_OK`.

Run:
```bash
python ML_test.py
```

### `TestConfig.py`
- Evaluation config (`EvalCFG`) used by `ML_val.py` and `ML_test.py`:
  - `CKPT_PATH` (checkpoint to evaluate)
  - `THRESH_OK` (decision threshold)
  - `PLOT_CURVES`

### `AugmentConfig.py`
- **Offline augmentation generator**:
  - `AugmentConfig`: source/target paths and target image count.
  - `AugmentMethods`: augmentation operations used to create new NOK variants.

### `main.py`
- Runs **offline augmentation** from `AugmentConfig.py`:
  - Reads original NOK images from `SRC_DIR`
  - Saves augmented variants to `DEST_DIR`
  - Stops when `GOOD_COUNT` total images is reached

Run:
```bash
python main.py
```

### `split.py`
- **Group-aware dataset split** (prevents leakage between originals and their augmentations):
  - OK images: each file is its own group
  - NOK images: group id is the filename prefix before the first `_`
- Creates `train/`, `val/`, `test/` folders under `OUT_ROOT` and writes `manifest.csv`.

Run:
```bash
python split.py
```

### `Predict_objects.py`
- Inference utilities:
  - `PredictPhoto`: single-image prediction with simple **TTA** (test-time augmentation).
  - `GradCam`: Grad-CAM visualization to inspect which regions influenced the decision.

Run (default `__main__` runs Grad-CAM flow):
```bash
python Predict_objects.py
```

---

## Dataset format

Training/evaluation uses `torchvision.datasets.ImageFolder`, so folders must look like:

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

---

## Augmentations

### Offline augmentations (saved to disk) — `AugmentConfig.py`

Operations (each creates a new file with a suffix tag):

- Rotations with reflected borders: `rot+3`, `rot-3`
- Shifts + zoom crop:
  - `shift_left`, `shift_right`, `shift_up`, `shift_down` (scale 0.90)
  - `shift_big_left`, `shift_big_right` (scale 0.88)
  - `zoom_center_0.90`, `zoom_center_0.80`
- Crops (stronger zoom/shift):
  - `crop_top`, `crop_bottom` (scale 0.85)
  - `crop_left`, `crop_right` (scale 0.82)
  - `crop_top_strong`, `crop_bottom_strong` (scale 0.75)
- Brightness/contrast:
  - `bright+10`, `bright-10`, `bright+5`, `bright-5`
  - `contrast+10`, `contrast-10`
- Background noise:
  - global noise: `bg+07`, `bg+12`
  - perimeter-only noise: `bgp+08`, `bgp+15`
- Horizontal flip: `hflip`
- Gaussian blur: `blur`
- Combos:
  - `hflip_rot+3`, `hflip_rot-3`

### Online augmentations (during training) — `ML_objects.py`

Applied only to the training set:

- `RandomResizedCrop(scale=(0.92, 1.0), ratio=(0.98, 1.02))`
- `RandomHorizontalFlip(p=0.5)`
- `RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02))`
- `ColorJitter(brightness=0.05, contrast=0.05)`
- `GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))`
- `PerimeterErasing(p=0.15, band=0.12)` (custom: zeros a random border rectangle)
- `Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))`

Validation/test transforms are deterministic: Resize + CenterCrop + Normalize.

### Test-time augmentation (TTA) for prediction — `Predict_objects.py`

For a single image, `PredictPhoto` can compute a **median P(OK)** across views:

- original
- horizontal flip
- rotate +3°
- rotate -3°
- “micro-blur” (resize up then back down)
- small crop (~2% border)

---

## Predict one image (PredictPhoto)

### Option A: Use it from Python
```python
from Predict_objects import CFG, PredictPhoto
from ML_objects import CNN

cfg = CFG()
model = CNN()

pred = PredictPhoto(cfg, model)
pred.predict_one()  # opens a file dialog and prints the result
```

### Option B: Call from `__main__` (quick edit)
In `Predict_objects.py`, replace the bottom block with:

```python
if __name__ == "__main__":
    cfg = CFG()
    model = CNN()
    PredictPhoto(cfg, model).predict_one()
```

**Output:** prints the final label (OK/NOK), `P(OK)` (TTA median + base), and a short model summary.

---

## Notes / common gotchas

- Many paths in configs are Windows absolute paths (e.g. `Q:\\...`).
  Adjust them to your local machine before running training/splitting/augmentation.
- For inference (`Predict_objects.py`) the default checkpoint path is relative: `ML/best3.pt`.
- Grad-CAM requires `pytorch-grad-cam`.

---

## Minimal dependencies

From imports used in this repo:

- `torch`, `torchvision`
- `numpy`, `Pillow`
- `opencv-python` (offline rotate with reflected borders)
- `scikit-learn` (metrics + group split)
- `matplotlib`
- `tqdm`
- `pytorch-grad-cam` (optional, only for Grad-CAM)
