from AugmentConfig import AugmentConfig, AugmentMethods


def main():
    """
    Offline augmentation runner
    Generates augmented images from SRC_DIR into DEST_DIR until GOOD_COUNT is reached.
    Parameters:
    - None
    Outputs:
    - None
    """
    cfg = AugmentConfig()
    augmenter = AugmentMethods(cfg)

    assert cfg.SRC_DIR.exists() and cfg.SRC_DIR.is_dir(), f"Missing source folder: {cfg.SRC_DIR}"
    cfg.DEST_DIR.mkdir(parents=True, exist_ok=True)
    assert cfg.SRC_DIR.resolve() != cfg.DEST_DIR.resolve(), "SRC_DIR and DEST_DIR cannot be the same folder."

    src_files = augmenter.list_images(cfg.SRC_DIR)
    n_src = len(src_files)
    n_aug = len(augmenter.list_images(cfg.DEST_DIR))
    total_now = n_src + n_aug
    remaining = max(0, cfg.GOOD_COUNT - total_now)

    print(f"[STAT] originals: {n_src}, augmented: {n_aug}, total: {total_now}")
    print(f"[STAT] missing: {remaining}")

    created = 0

    if remaining == 0:
        print("[OK] complete")
    else:
        max_ops = len(augmenter.get_ops())

        while created < remaining:
            made_in_round = 0
            for f in src_files:
                if created >= remaining:
                    break

                have = augmenter.count_variants_for_file(f.stem)
                if have < max_ops:
                    c = augmenter.augment_one(f, how_many=have + 1)
                    if c > 0:
                        created += c
                        made_in_round += 1
                        if created % 5 == 0 or created >= remaining:
                            print(f"[PROGRESS] added {created}/{remaining}")

            if made_in_round == 0:
                print("No more new variants possible (all files reached max_ops).")
                break

    n_aug2 = len(augmenter.list_images(cfg.DEST_DIR))
    total2 = n_src + n_aug2
    print(f"[DONE] added: {created}. Augmented now: {n_aug2}, total: {total2}")
    print(f"[CHECK] missing: {max(0, cfg.GOOD_COUNT - total2)}")


if __name__ == "__main__":
    main()
