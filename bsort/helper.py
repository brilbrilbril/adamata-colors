import os
from pathlib import Path

import click
import yaml


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def check_augmentation_exists(base_path, split="train"):
    """Check if augmentation directory exists and has content."""
    aug_path = Path(base_path) / "relabel_aug" / split / "images"
    return aug_path.exists() and len(list(aug_path.glob("*"))) > 0


def read_yolo_label(label_path):
    bboxes = []
    classes = []
    if not os.path.exists(label_path):
        return bboxes, classes

    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, xc, yc, w, h = line.strip().split()
            bboxes.append([float(xc), float(yc), float(w), float(h)])
            classes.append(int(cls))
    return bboxes, classes


def write_yolo_label(label_path, bboxes, classes):
    with open(label_path, "w") as f:
        for cls, bb in zip(classes, bboxes):
            f.write(f"{cls} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")


def check_augmentation_exists(base_path, split="train"):
    """Check if augmentation directory exists and has content."""
    aug_path = Path(base_path) / "relabel_aug" / split / "images"
    return aug_path.exists() and len(list(aug_path.glob("*"))) > 0


def create_dynamic_yolo_config(config, output_path="config_dynamic.yaml"):
    """Create YOLO config file based on settings."""
    base_path = Path(config.get("base_path", "."))

    # Check if augmentation data exists
    use_augmentation = check_augmentation_exists(base_path)

    # If augmentation data exists, use it
    if use_augmentation:
        train_path = config.get("augmented_train_path", "relabel_aug/train/images")
        click.echo(click.style("✓ Using augmented training data", fg="green"))
    else:
        train_path = config.get("raw_train_path", "relabel/train/images")
        click.echo(click.style("✓ Using raw training data (no augmentation found)", fg="yellow"))

    yolo_config = {
        "path": str(base_path),
        "train": train_path,
        "val": config.get("val_path", "relabel/val/images"),
        "names": config.get("names", {}),
    }
    
    output_file = base_path / output_path
    with open(output_file, "w") as f:
        yaml.dump(yolo_config, f, default_flow_style=False, sort_keys=False)

    click.echo(f"✓ YOLO config created: {output_file}")
    return str(output_file)
