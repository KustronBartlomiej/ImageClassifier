from pathlib import Path
from dataclasses import dataclass

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import cv2


@dataclass
class AugmentConfig:
    """
    Augmentation config
    Stores input/output paths and basic settings for offline image augmentation.
    Parameters:
    - SRC_DIR (Path): Directory with source images (original NOK).
    - DEST_DIR (Path): Directory where augmented variants are saved.
    - GOOD_COUNT (int): Target number of images for the NOK class.
    - IMG_EXTS (tuple[str, ...]): Allowed image extensions.
    Outputs:
    - AugmentConfig: Configuration object used by augmentation utilities.
    """
    SRC_DIR: Path = Path(r"Q:\VisualStudio\ML_Model\data\zdjecia_kopia\zle")
    DEST_DIR: Path = Path(r"Q:\VisualStudio\ML_Model\data\augmented3")
    GOOD_COUNT: int = 9809
    IMG_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class AugmentMethods:
    """
    Augmentation methods
    Provides a set of offline augmentation operations and helpers for saving variants.
    Parameters:
    - config (AugmentConfig): Configuration with folders and settings.
    Outputs:
    - AugmentMethods: Object with augmentation operations ready to use.
    """

    def __init__(self, config: AugmentConfig):
        """
        Init
        Stores config used for file IO and filtering.
        Parameters:
        - config (AugmentConfig): Augmentation configuration.
        Outputs:
        - None
        """
        self.config = config

    def rotate_reflect_cv(self, img_pil: Image.Image, angle: float) -> Image.Image:
        """
        Rotate with reflected borders
        Rotates an image using OpenCV and fills borders by reflection.
        Parameters:
        - img_pil (Image.Image): Input PIL image (RGB).
        - angle (float): Rotation angle in degrees.
        Outputs:
        - Image.Image: Rotated PIL image (RGB).
        """
        arr = np.array(img_pil)[:, :, ::-1].copy()  # PIL RGB -> OpenCV BGR
        h, w = arr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(
            arr,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        return Image.fromarray(rot[:, :, ::-1])  # BGR -> RGB

    def shift_crop(self, img_pil: Image.Image, dx: int, dy: int, scale: float = 1.0) -> Image.Image:
        """
        Zoom and shift crop
        Crops a scaled window, shifts it by (dx, dy) and resizes back to original size.
        Parameters:
        - img_pil (Image.Image): Input image.
        - dx (int): Horizontal shift in pixels (positive = right).
        - dy (int): Vertical shift in pixels (positive = down).
        - scale (float): Crop scale relative to original (e.g. 0.9 means zoom-in).
        Outputs:
        - Image.Image: Transformed image with original size.
        """
        w, h = img_pil.size
        new_w, new_h = int(w * scale), int(h * scale)

        left = min(max(0, int((w - new_w) / 2 + dx)), w - new_w)
        top = min(max(0, int((h - new_h) / 2 + dy)), h - new_h)

        cropped = img_pil.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.LANCZOS)

    def change_background(self, img_pil: Image.Image, intensity: float = 0.1) -> Image.Image:
        """
        Add global noise
        Adds random noise to all pixels to slightly dirty the background.
        Parameters:
        - img_pil (Image.Image): Input image.
        - intensity (float): Noise range in normalized space [0,1].
        Outputs:
        - Image.Image: Image with added noise.
        """
        arr = np.array(img_pil).astype(np.float32)
        noise = np.random.uniform(-intensity, intensity, arr.shape)
        arr = np.clip(arr / 255.0 + noise, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))

    def change_background_perimeter(
        self,
        img_pil: Image.Image,
        intensity: float = 0.12,
        band: float = 0.15
    ) -> Image.Image:
        """
        Add perimeter noise
        Adds stronger noise only near the image borders (perimeter band).
        Parameters:
        - img_pil (Image.Image): Input image.
        - intensity (float): Noise range in normalized space [0,1].
        - band (float): Band width as a fraction of the shorter image side.
        Outputs:
        - Image.Image: Image with noisy perimeter and clean center.
        """
        arr = np.array(img_pil).astype(np.float32) / 255.0  # [H,W,3]
        h, w, _ = arr.shape

        b = max(2, int(min(h, w) * band))
        mask = np.zeros((h, w), dtype=np.float32)
        mask[:b, :] = 1.0
        mask[-b:, :] = 1.0
        mask[:, :b] = 1.0
        mask[:, -b:] = 1.0

        mask3 = mask[..., None]
        noise = np.random.uniform(-intensity, intensity, arr.shape).astype(np.float32)

        arr = np.clip(arr + noise * mask3, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))

    def adjust_brightness_contrast(self, img_pil: Image.Image, b: float, c: float) -> Image.Image:
        """
        Brightness and contrast
        Adjusts brightness and contrast using PIL enhancers.
        Parameters:
        - img_pil (Image.Image): Input image.
        - b (float): Brightness multiplier (1.0 = no change).
        - c (float): Contrast multiplier (1.0 = no change).
        Outputs:
        - Image.Image: Adjusted image.
        """
        im1 = ImageEnhance.Brightness(img_pil).enhance(b)
        im2 = ImageEnhance.Contrast(im1).enhance(c)
        return im2

    def get_ops(self):
        """
        Build operation list
        Returns a list of (tag, function) pairs with about 30 augmentation variants.
        Outputs:
        - list[tuple[str, callable]]: Augmentation operations in a fixed order.
        """
        R = []

        base = [
            ("rot+3",  lambda im: self.rotate_reflect_cv(im, 3)),
            ("rot-3",  lambda im: self.rotate_reflect_cv(im, -3)),
            ("shift_left",   lambda im: self.shift_crop(im, -20, 0, 0.90)),
            ("shift_right",  lambda im: self.shift_crop(im,  20, 0, 0.90)),
            ("shift_up",     lambda im: self.shift_crop(im,  0, -20, 0.90)),
            ("shift_down",   lambda im: self.shift_crop(im,  0,  20, 0.90)),
            ("zoom_center_0.90", lambda im: self.shift_crop(im, 0, 0, 0.90)),
            ("zoom_center_0.80", lambda im: self.shift_crop(im, 0, 0, 0.80)),
        ]

        base += [
            ("bright+10",   lambda im: self.adjust_brightness_contrast(im, 1.10, 1.00)),
            ("bright-10",   lambda im: self.adjust_brightness_contrast(im, 0.90, 1.00)),
            ("contrast+10", lambda im: self.adjust_brightness_contrast(im, 1.00, 1.10)),
            ("contrast-10", lambda im: self.adjust_brightness_contrast(im, 1.00, 0.90)),
            ("bgp+08", lambda im: self.change_background_perimeter(im, intensity=0.08, band=0.15)),
            ("bgp+15", lambda im: self.change_background_perimeter(im, intensity=0.15, band=0.18)),
            ("hflip",  lambda im: im.transpose(Image.FLIP_LEFT_RIGHT)),
            ("blur",   lambda im: im.filter(ImageFilter.GaussianBlur(radius=1.0))),
        ]
        R += base

        head_scale_ops = [
            ("crop_top",           lambda im: self.shift_crop(im,  0, -30, 0.85)),
            ("crop_bottom",        lambda im: self.shift_crop(im,  0,  30, 0.85)),
            ("crop_left",          lambda im: self.shift_crop(im, -30,  0, 0.82)),
            ("crop_right",         lambda im: self.shift_crop(im,  30,  0, 0.82)),
            ("crop_top_strong",    lambda im: self.shift_crop(im,  0, -45, 0.75)),
            ("crop_bottom_strong", lambda im: self.shift_crop(im,  0,  45, 0.75)),
        ]

        bg_ops = [
            ("bg+07",   lambda im: self.change_background(im, 0.07)),
            ("bg+12",   lambda im: self.change_background(im, 0.12)),
            ("bright+5", lambda im: self.adjust_brightness_contrast(im, 1.05, 1.00)),
            ("bright-5", lambda im: self.adjust_brightness_contrast(im, 0.95, 1.00)),
        ]

        combo_ops = [
            ("hflip_rot+3",     lambda im: self.rotate_reflect_cv(im.transpose(Image.FLIP_LEFT_RIGHT), 3)),
            ("hflip_rot-3",     lambda im: self.rotate_reflect_cv(im.transpose(Image.FLIP_LEFT_RIGHT), -3)),
            ("shift_big_left",  lambda im: self.shift_crop(im, -30, 0, 0.88)),
            ("shift_big_right", lambda im: self.shift_crop(im,  30, 0, 0.88)),
        ]

        R += head_scale_ops
        R += bg_ops
        R += combo_ops

        return R

    def augment_one(self, src_file: Path, how_many: int = 30) -> int:
        """
        Augment a single file
        Creates up to `how_many` variants for one file without overwriting existing outputs.
        Parameters:
        - src_file (Path): Source image file path.
        - how_many (int): Maximum number of variants to generate from the ops list.
        Outputs:
        - int: Number of images actually saved.
        """
        img = ImageOps.exif_transpose(Image.open(src_file)).convert("RGB")

        ops = self.get_ops()
        to_save = min(how_many, len(ops))
        saved = 0
        base = src_file.stem

        for tag, op in ops[:to_save]:
            out = self.config.DEST_DIR / f"{base}_{tag}.jpg"
            if out.exists():
                print("[AUG] skip (exists):", out)
                continue
            op(img).save(out, quality=95)
            print("[AUG] saved:", out)
            saved += 1
        return saved

    def list_images(self, path: Path):
        """
        List images
        Returns a sorted list of image files from a directory (recursive) or a single file if provided.
        Parameters:
        - path (Path): Directory or file path.
        Outputs:
        - list[Path]: List of image paths matching IMG_EXTS.
        """
        if path.is_file():
            return [path] if path.suffix.lower() in self.config.IMG_EXTS else []
        return sorted(
            p for p in path.rglob("*")
            if p.suffix.lower() in self.config.IMG_EXTS
        )

    def count_variants_for_file(self, stem: str) -> int:
        """
        Count saved variants
        Counts already generated variants in DEST_DIR for a given original stem.
        Parameters:
        - stem (str): Base filename (without extension).
        Outputs:
        - int: Total number of matching variant files in DEST_DIR.
        """
        total = 0
        for ext in self.config.IMG_EXTS:
            total += len(list(self.config.DEST_DIR.glob(f"{stem}_*{ext}")))
        return total
