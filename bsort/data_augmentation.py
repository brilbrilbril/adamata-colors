import os
import random
from pathlib import Path
from typing import Dict, Any, Optional

import albumentations as A
import click
import cv2
import numpy as np
from tqdm import tqdm

# from dotenv import load_dotenv
from bsort.helper import read_yolo_label, write_yolo_label

from .helper import load_config


def run_augmentation(
    config_path: str,
    split: str = "train",
    force: bool = False
) -> None:
    """Run data augmentation on training images.

    This function applies various augmentation techniques to images and their
    corresponding YOLO format labels, creating multiple augmented versions of
    each image to expand the training dataset.

    Args:
        config_path: Path to the configuration file containing augmentation settings.
        split: Dataset split to augment (e.g., 'train', 'val', 'test').
            Defaults to 'train'.
        force: If True, removes existing augmented images and re-augments.
            If False, skips augmentation if output directory already contains files.
            Defaults to False.

    Raises:
        click.Abort: If no images are found in the input directory or if
            required files are missing.
        FileNotFoundError: If the configuration file or required directories
            cannot be found.
        Exception: If augmentation process encounters an unexpected error.

    Returns:
        None

    Examples:
        >>> run_augmentation('config.yaml', split='train', force=False)
        >>> run_augmentation('config.yaml', split='val', force=True)

    Note:
        - The function expects images in JPG format and labels in YOLO format.
        - Augmented images are saved with suffix '_aug{i}.jpg' where i is the
          augmentation index.
        - The function uses the albumentations library for transformations.
    """
    click.echo(click.style("\n=== Data Augmentation ===", fg="cyan", bold=True))

    try:
        settings: Dict[str, Any] = load_config(config_path)
        click.echo(f"Loaded config: {config_path}")

        base_path: Path = Path(settings.get("base_path", "."))
        input_dir: Path = base_path / settings.get("input_dir", "relabel")
        augment_dir: Path = base_path / settings.get("augment_dir", "relabel_aug")

        aug_settings: Dict[str, Any] = settings.get("augmentation", {})
        n_aug: int = aug_settings.get("aug_per_image", 5)

        img_dir: Path = input_dir / split / "images"
        lbl_dir: Path = input_dir / split / "labels"
        out_img_dir: Path = augment_dir / split / "images"
        out_lbl_dir: Path = augment_dir / split / "labels"

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        existing_count: int = len(list(out_img_dir.glob("*")))
        if existing_count > 0 and not force:
            click.echo(
                click.style(
                    f"Augmented images already exist ({existing_count} files). Use --force to re-augment.",
                    fg="yellow",
                )
            )
            return

        if force and existing_count > 0:
            click.echo(f"Removing existing augmented images...")
            for f in out_img_dir.glob("*"):
                f.unlink()
            for f in out_lbl_dir.glob("*"):
                f.unlink()

        # augmentation config
        transform: A.Compose = A.Compose(
            [
                A.HorizontalFlip(p=aug_settings.get("horizontal_flip", 0)),
                A.VerticalFlip(p=aug_settings.get("vertical_flip", 0)),
                A.Rotate(limit=aug_settings.get("rotation_limit", 0), p=0.5),
                A.RandomBrightnessContrast(p=aug_settings.get("brightness_contrast", 0)),
                A.Blur(blur_limit=3, p=aug_settings.get("blur", 0)),
                A.ColorJitter(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=aug_settings.get("shift_limit", 0),
                    scale_limit=aug_settings.get("scale_limit", 0),
                    rotate_limit=15,
                    p=0.5,
                ),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

        images: list[Path] = sorted(list(img_dir.glob("*.jpg")))

        if not images:
            click.echo(click.style(f"✗ No images found in {img_dir}", fg="red"))
            raise click.Abort()

        click.echo(f"\nProcessing {len(images)} images...")
        click.echo(f"Augmentations per image: {n_aug}")
        click.echo(f"Total output images: {len(images) * n_aug}\n")

        for img_path in tqdm(images, desc=f"Augmenting {split}"):
            label_path: Path = lbl_dir / img_path.with_suffix(".txt").name

            image: Optional[np.ndarray] = cv2.imread(str(img_path))
            if image is None:
                click.echo(f"Warning: Could not read {img_path}")
                continue

            bboxes, classes = read_yolo_label(str(label_path))

            for i in range(n_aug):
                try:
                    aug: Dict[str, Any] = transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=classes
                    )
                    aug_img: np.ndarray = aug["image"]
                    aug_bboxes: list = aug["bboxes"]
                    aug_classes: list = aug["class_labels"]

                    out_name: str = img_path.stem + f"_aug{i}.jpg"

                    cv2.imwrite(str(out_img_dir / out_name), aug_img)
                    write_yolo_label(
                        str(out_lbl_dir / out_name.replace(".jpg", ".txt")),
                        aug_bboxes,
                        aug_classes
                    )
                except Exception as e:
                    click.echo(
                        f"Warning: Augmentation failed for {img_path.name} "
                        f"(aug {i}): {e}"
                    )
                    continue

        click.echo(click.style("Augmentation completed!", fg="green", bold=True))
        click.echo(f"Output directory: {augment_dir / split}")

    except FileNotFoundError as e:
        click.echo(click.style(f"File not found: {e}", fg="red"))
        raise click.Abort()
    except Exception as e:
        click.echo(click.style(f"Augmentation failed: {str(e)}", fg="red"))
        raise click.Abort()