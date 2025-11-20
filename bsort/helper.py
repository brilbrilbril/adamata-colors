import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the parsed configuration data.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML file is malformed.

    Examples:
        >>> config = load_config('config.yaml')
        >>> print(config['base_path'])
    """
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


def check_augmentation_exists(base_path: str, split: str = "train") -> bool:
    """Check if augmentation directory exists and has content.

    Args:
        base_path: Base directory path containing the augmentation folder.
        split: Dataset split to check (e.g., 'train', 'val', 'test').
            Defaults to 'train'.

    Returns:
        True if the augmentation directory exists and contains files,
        False otherwise.

    Examples:
        >>> if check_augmentation_exists('/path/to/data', 'train'):
        ...     print("Augmented data available")
    """
    aug_path: Path = Path(base_path) / "relabel_aug" / split / "images"
    return aug_path.exists() and len(list(aug_path.glob("*"))) > 0


def read_yolo_label(label_path: str) -> Tuple[List[List[float]], List[int]]:
    """Read YOLO format label file and extract bounding boxes and classes.

    Parses a YOLO format label file where each line contains:
    class_id center_x center_y width height (all normalized to [0, 1]).

    Args:
        label_path: Path to the YOLO format label text file.

    Returns:
        A tuple containing:
            - List of bounding boxes, where each bbox is [x_center, y_center, width, height]
            - List of class IDs corresponding to each bounding box

    Examples:
        >>> bboxes, classes = read_yolo_label('labels/image001.txt')
        >>> print(f"Found {len(bboxes)} objects")

    Note:
        If the label file doesn't exist, returns empty lists for both
        bounding boxes and classes.
    """
    bboxes: List[List[float]] = []
    classes: List[int] = []
    if not os.path.exists(label_path):
        return bboxes, classes

    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, xc, yc, w, h = line.strip().split()
            bboxes.append([float(xc), float(yc), float(w), float(h)])
            classes.append(int(cls))
    return bboxes, classes


def write_yolo_label(label_path: str, bboxes: List[List[float]], classes: List[int]) -> None:
    """Write bounding boxes and classes to a YOLO format label file.

    Writes annotations in YOLO format where each line contains:
    class_id center_x center_y width height (all values normalized).

    Args:
        label_path: Output path for the YOLO format label file.
        bboxes: List of bounding boxes, where each bbox is
            [x_center, y_center, width, height].
        classes: List of class IDs corresponding to each bounding box.

    Returns:
        None

    Raises:
        IOError: If the file cannot be written.

    Examples:
        >>> bboxes = [[0.5, 0.5, 0.3, 0.4]]
        >>> classes = [0]
        >>> write_yolo_label('output/label.txt', bboxes, classes)

    Note:
        Coordinates are written with 6 decimal places of precision.
    """
    with open(label_path, "w") as f:
        for cls, bb in zip(classes, bboxes):
            f.write(f"{cls} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")


def create_dynamic_yolo_config(
    config: Dict[str, Any], output_path: str = "config_dynamic.yaml"
) -> str:
    """Create YOLO configuration file based on settings.

    Generates a YOLO-compatible YAML configuration file, automatically
    selecting between raw and augmented training data based on availability.

    Args:
        config: Dictionary containing configuration settings with keys:
            - base_path: Base directory for the dataset
            - augmented_train_path: Path to augmented training images
            - raw_train_path: Path to raw training images
            - val_path: Path to validation images
            - names: Dictionary mapping class IDs to class names
        output_path: Filename for the output YAML config file.
            Defaults to 'config_dynamic.yaml'.

    Returns:
        String path to the created configuration file.

    Raises:
        IOError: If the output file cannot be written.

    Examples:
        >>> config = {
        ...     'base_path': '/data',
        ...     'names': {0: 'person', 1: 'car'}
        ... }
        >>> config_file = create_dynamic_yolo_config(config)
        >>> print(f"Config saved to: {config_file}")

    Note:
        The function automatically detects whether augmented data exists
        and configures the training path accordingly. It displays colored
        status messages using click.
    """
    base_path: Path = Path(config.get("base_path", "."))

    use_augmentation: bool = check_augmentation_exists(base_path)

    # if exists = use augmentation images as train
    if use_augmentation:
        train_path: str = config.get("augmented_train_path", "relabel_aug/train/images")
        click.echo(click.style("Using augmented training data", fg="green"))
    else:
        train_path: str = config.get("raw_train_path", "relabel/train/images")
        click.echo(click.style("Using raw training data (no augmentation found)", fg="yellow"))

    yolo_config: Dict[str, Any] = {
        "path": str(base_path),
        "train": train_path,
        "val": config.get("val_path", "relabel/val/images"),
        "names": config.get("names", {}),
    }

    output_file: Path = base_path / output_path
    with open(output_file, "w") as f:
        yaml.dump(yolo_config, f, default_flow_style=False, sort_keys=False)

    click.echo(f"✓ YOLO config created: {output_file}")
    return str(output_file)
