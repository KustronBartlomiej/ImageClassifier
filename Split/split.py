from pathlib import Path
from typing import List, Tuple, Dict
import shutil
import csv
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class CFG:
    """
    Split config
    Stores input folders, output root, split sizes and random seeds for dataset splitting.
    Parameters:
    - OK_DIR (Path): Folder with OK images.
    - NOK_ORIG_DIR (Path): Folder with original NOK images.
    - NOK_AUG_DIR (Path): Folder with augmented NOK images.
    - OUT_ROOT (Path): Output root where train/val/test folders are created.
    - VAL_SIZE (float): Validation split size.
    - TEST_SIZE (float): Test split size.
    - SEED1 (int): Seed for the first group split.
    - SEED2 (int): Seed for the second group split.
    - IMG_EXTS (tuple[str, ...]): Allowed image extensions.
    Outputs:
    - CFG: Configuration object for SplitSet.
    """
    OK_DIR: Path = Path(r"Q:\VisualStudio\ML_Model\data\zdjecia_kopia\dobre")
    NOK_ORIG_DIR: Path = Path(r"Q:\VisualStudio\ML_Model\data\zdjecia_kopia\zle")
    NOK_AUG_DIR: Path = Path(r"Q:\VisualStudio\ML_Model\data\augmented3")

    OUT_ROOT: Path = Path(r"Q:\VisualStudio\ML_Model\InputData3")
    VAL_SIZE: float = 0.15
    TEST_SIZE: float = 0.15
    SEED1: int = 42
    SEED2: int = 123

    IMG_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class SplitSet:
    """
    Group-based dataset splitter
    Builds an index of OK/NOK images and creates train/val/test splits without group leakage.
    Parameters:
    - cfg (CFG): Split configuration.
    Outputs:
    - SplitSet: Split helper object.
    """

    def __init__(self, cfg: CFG):
        """
        Init
        Stores configuration for listing, grouping, splitting and copying images.
        Parameters:
        - cfg (CFG): Split configuration.
        Outputs:
        - None
        """
        self.config = cfg

    def list_images(self, path: Path) -> List[Path]:
        """
        List images
        Returns a sorted list of images from a directory (recursive). If a file is provided, returns [file].
        Parameters:
        - path (Path): Directory or file path.
        Outputs:
        - List[Path]: List of image paths with allowed extensions.
        """
        if not path.exists():
            return []
        if path.is_file():
            return [path] if path.suffix.lower() in self.config.IMG_EXTS else []
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in self.config.IMG_EXTS)

    def ensure_dir(self, p: Path) -> Path:
        """
        Ensure directory
        Creates a directory (with parents). Raises if a file exists at the same path.
        Parameters:
        - p (Path): Directory path to create.
        Outputs:
        - Path: The created/existing directory path.
        """
        if p.exists() and not p.is_dir():
            raise NotADirectoryError(f"A file exists where a directory is expected: {p}")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def safe_copy(self, src: Path, dst: Path) -> Path:
        """
        Safe copy
        Copies a file without overwriting. If destination exists, appends a numbered __dupN suffix.
        Parameters:
        - src (Path): Source file.
        - dst (Path): Destination file path.
        Outputs:
        - Path: The actual destination path that was written.
        """
        if not dst.exists():
            shutil.copy2(src, dst)
            return dst
        stem, suf = dst.stem, dst.suffix
        k = 1
        while True:
            cand = dst.with_name(f"{stem}__dup{k}{suf}")
            if not cand.exists():
                shutil.copy2(src, cand)
                return cand
            k += 1

    def nok_group_id(self, path: Path) -> str:
        """
        NOK grouping
        Returns the family id for NOK: everything before the first '_' in the filename stem.
        Parameters:
        - path (Path): Image path.
        Outputs:
        - str: Group id for NOK images.
        """
        return path.stem.split("_", 1)[0]

    def ok_group_id(self, path: Path) -> str:
        """
        OK grouping
        Returns the group id for OK: each image is its own group (stem).
        Parameters:
        - path (Path): Image path.
        Outputs:
        - str: Group id for OK images.
        """
        return path.stem

    def build_index(self) -> Tuple[List[Path], List[int], List[str]]:
        """
        Build index
        Builds (paths, labels, groups) for all images.
        labels: 0=OK, 1=NOK
        groups: for OK = stem, for NOK = prefix before '_'
        Outputs:
        - Tuple[List[Path], List[int], List[str]]: paths, labels, groups lists with equal length.
        """
        paths: List[Path] = []
        labels: List[int] = []
        groups: List[str] = []

        for p in self.list_images(self.config.OK_DIR):
            paths.append(p)
            labels.append(0)
            groups.append(self.ok_group_id(p))

        for p in self.list_images(self.config.NOK_ORIG_DIR):
            paths.append(p)
            labels.append(1)
            groups.append(self.nok_group_id(p))

        for p in self.list_images(self.config.NOK_AUG_DIR):
            paths.append(p)
            labels.append(1)
            groups.append(self.nok_group_id(p))

        assert len(paths) == len(labels) == len(groups), "Lists have different lengths"
        return paths, labels, groups

    def make_splits(
        self,
        paths: List[Path],
        labels: List[int],
        groups: List[str],
        val_size: float,
        test_size: float,
        seed1: int,
        seed2: int,
    ) -> Dict[str, List[int]]:
        """
        Make splits
        Creates train/val/test splits by GROUPS to avoid leakage.
        Parameters:
        - paths (List[Path]): All image paths.
        - labels (List[int]): Labels aligned with paths (0=OK, 1=NOK).
        - groups (List[str]): Group ids aligned with paths.
        - val_size (float): Validation size fraction.
        - test_size (float): Test size fraction.
        - seed1 (int): Seed for the first split (train vs temp).
        - seed2 (int): Seed for the second split (val vs test inside temp).
        Outputs:
        - Dict[str, List[int]]: Index dictionary with keys: 'train', 'val', 'test'.
        """
        gss1 = GroupShuffleSplit(
            n_splits=1,
            test_size=val_size + test_size,
            random_state=seed1,
        )
        train_idx, temp_idx = next(gss1.split(paths, labels, groups))

        temp_paths = [paths[i] for i in temp_idx]
        temp_labels = [labels[i] for i in temp_idx]
        temp_groups = [groups[i] for i in temp_idx]

        rel_test = test_size / (val_size + test_size)
        gss2 = GroupShuffleSplit(
            n_splits=1,
            test_size=rel_test,
            random_state=seed2,
        )
        val_sub, test_sub = next(gss2.split(temp_paths, temp_labels, temp_groups))

        val_idx = [temp_idx[i] for i in val_sub]
        test_idx = [temp_idx[i] for i in test_sub]

        return {
            "train": list(train_idx),
            "val": val_idx,
            "test": test_idx,
        }

    def copy_split(
        self,
        paths: List[Path],
        labels: List[int],
        split_idxs: List[int],
        out_root: Path,
    ) -> tuple[int, int]:
        """
        Copy split
        Copies selected indices into out_root/{OK,NOK}. Names are preserved, collisions get __dupN suffix.
        Parameters:
        - paths (List[Path]): All image paths.
        - labels (List[int]): Labels aligned with paths.
        - split_idxs (List[int]): Indices to copy.
        - out_root (Path): Output root for this split (train/val/test).
        Outputs:
        - tuple[int, int]: (copied_ok, copied_nok)
        """
        dst_ok = self.ensure_dir(out_root / "OK")
        dst_nok = self.ensure_dir(out_root / "NOK")

        ok_files = [paths[i] for i in split_idxs if labels[i] == 0]
        nok_files = [paths[i] for i in split_idxs if labels[i] == 1]

        copied_ok = 0
        for p in tqdm(ok_files, desc=f"COPY {out_root.name}/OK", unit="img"):
            self.safe_copy(p, dst_ok / p.name)
            copied_ok += 1

        copied_nok = 0
        for p in tqdm(nok_files, desc=f"COPY {out_root.name}/NOK", unit="img"):
            self.safe_copy(p, dst_nok / p.name)
            copied_nok += 1

        return copied_ok, copied_nok

    def write_manifest(
        self,
        paths: List[Path],
        labels: List[int],
        groups: List[str],
        splits: Dict[str, List[int]],
        out_csv: Path,
    ) -> None:
        """
        Write manifest
        Writes a CSV file: path,label_name,label_idx,group_id,split.
        Parameters:
        - paths (List[Path]): All image paths.
        - labels (List[int]): Labels aligned with paths.
        - groups (List[str]): Group ids aligned with paths.
        - splits (Dict[str, List[int]]): Split indices.
        - out_csv (Path): Output CSV path.
        Outputs:
        - None
        """
        label_name = {0: "OK", 1: "NOK"}
        rows = []

        for split_name, idxs in splits.items():
            for i in idxs:
                rows.append(
                    {
                        "path": str(paths[i]),
                        "label_name": label_name[labels[i]],
                        "label_idx": labels[i],
                        "group_id": groups[i],
                        "split": split_name,
                    }
                )

        self.ensure_dir(out_csv.parent)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["path", "label_name", "label_idx", "group_id", "split"],
            )
            w.writeheader()
            w.writerows(rows)

    def summarize(
        self,
        labels: List[int],
        groups: List[str],
        splits: Dict[str, List[int]],
    ) -> None:
        """
        Summary and checks
        Prints split sizes and verifies that groups are disjoint across train/val/test.
        Parameters:
        - labels (List[int]): Labels aligned with the global index.
        - groups (List[str]): Group ids aligned with the global index.
        - splits (Dict[str, List[int]]): Split indices.
        Outputs:
        - None
        """

        def count(idxs: List[int]) -> tuple[int, int, int]:
            """
            Count helper
            Counts OK, NOK and total items for a given index list.
            Parameters:
            - idxs (List[int]): Indices belonging to a split.
            Outputs:
            - tuple[int, int, int]: (ok_count, nok_count, total_count)
            """
            ok = sum(1 for i in idxs if labels[i] == 0)
            nok = sum(1 for i in idxs if labels[i] == 1)
            return ok, nok, len(idxs)

        for name in ["train", "val", "test"]:
            ok, nok, n = count(splits[name])
            print(f"{name:5}: {n:5d} (OK {ok:5d} | NOK {nok:5d})")

        g_train = {groups[i] for i in splits["train"]}
        g_val = {groups[i] for i in splits["val"]}
        g_test = {groups[i] for i in splits["test"]}

        assert g_train.isdisjoint(g_val)
        assert g_train.isdisjoint(g_test)
        assert g_val.isdisjoint(g_test)
        print("[CHECK] Groups are disjoint across splits")


if __name__ == "__main__":
    cfg = CFG()
    split = SplitSet(cfg)

    n_ok = len(split.list_images(cfg.OK_DIR))
    n_nok_orig = len(split.list_images(cfg.NOK_ORIG_DIR))
    n_nok_aug = len(split.list_images(cfg.NOK_AUG_DIR))

    print("[STAT] OK:", n_ok)
    print("[STAT] NOK originals:", n_nok_orig)
    print("[STAT] NOK augmented:", n_nok_aug)
    print("[STAT] TOTAL:", n_ok + n_nok_orig + n_nok_aug)

    paths, labels, groups = split.build_index()
    splits = split.make_splits(
        paths,
        labels,
        groups,
        val_size=cfg.VAL_SIZE,
        test_size=cfg.TEST_SIZE,
        seed1=cfg.SEED1,
        seed2=cfg.SEED2,
    )
    split.summarize(labels, groups, splits)

    for split_name in ["train", "val", "test"]:
        out_dir = split.ensure_dir(cfg.OUT_ROOT / split_name)
        c_ok, c_nok = split.copy_split(paths, labels, splits[split_name], out_dir)
        print(f"[{split_name}] copied: OK {c_ok} | NOK {c_nok}")

    split.write_manifest(paths, labels, groups, splits, cfg.OUT_ROOT / "manifest.csv")
    print(f"[DONE] Dataset ready in: {cfg.OUT_ROOT}")
