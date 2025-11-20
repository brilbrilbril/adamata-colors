import os
from typing import Any, Dict, Optional

import click
import wandb
from ultralytics import YOLO

from .helper import create_dynamic_yolo_config, load_config


def run_training(
    config_path: str,
    epochs: Optional[int] = None,
    device: Optional[str] = None,
    batch: Optional[int] = None,
    imgsz: Optional[int] = None,
) -> None:
    """Train a YOLO object detection model.

    Trains a YOLO model using the specified configuration and training parameters.
    Supports integration with Weights & Biases (WandB) for experiment tracking
    and visualization. The function automatically creates a dynamic YOLO config
    that selects between raw and augmented training data.

    Args:
        config_path: Path to the YAML configuration file containing training settings,
            model parameters, and data paths.
        epochs: Number of training epochs. If None, uses value from config.
            Defaults to None.
        device: Device to use for training (e.g., 'cpu', 'cuda', '0', '0,1').
            If None or empty string, YOLO will auto-select. Defaults to None.
        batch: Batch size for training. If None or -1, uses auto batch sizing.
            Defaults to None.
        imgsz: Input image size (height and width). Common values are 640, 1280.
            If None, uses value from config. Defaults to None.

    Returns:
        None

    Raises:
        click.Abort: If the config file is not found or training fails.
        FileNotFoundError: If the configuration file doesn't exist.
        Exception: If an unexpected error occurs during training setup or execution.

    Examples:
        >>> # Train with default config settings
        >>> run_training('config.yaml')

        >>> # Train with custom epochs and device
        >>> run_training('config.yaml', epochs=200, device='cuda:0')

        >>> # Train with custom batch size and image size
        >>> run_training('config.yaml', batch=16, imgsz=1280)

        >>> # Override all parameters
        >>> run_training('config.yaml', epochs=300, device='0,1', batch=32, imgsz=640)

    Note:
        - WandB integration requires WANDB_API_KEY environment variable to be set.
        - If WandB is enabled but API key is missing, training continues without logging.
        - Batch size of -1 or None triggers automatic batch size optimization.
        - The function creates a dynamic YOLO config file that automatically selects
          augmented data if available, otherwise falls back to raw training data.
        - Training results are saved in the project directory specified in config
          or defaults to 'runs/detect/train'.
    """
    try:
        settings: Dict[str, Any] = load_config(config_path)
        click.echo(f"Loaded config: {config_path}")

        yolo_config_path: str = create_dynamic_yolo_config(settings)

        train_settings: Dict[str, Any] = settings.get("training", {})
        model_name: str = train_settings.get("model", "yolov9t.pt")
        epochs_val: int = epochs or train_settings.get("epochs", 100)
        imgsz_val: int = imgsz or train_settings.get("imgsz", 640)
        batch_size: int = batch or train_settings.get("batch", -1)
        device_setting: str = device or train_settings.get("device", "")

        wandb_settings: Dict[str, Any] = settings.get("wandb", {})
        use_wandb: bool = wandb_settings.get("enabled", True)
        wandb_project: str = wandb_settings.get("project", "bsort-yolo")
        wandb_entity: Optional[str] = wandb_settings.get("entity", None)
        wandb_name: Optional[str] = wandb_settings.get("name", None)

        if use_wandb:
            wandb_api_key: Optional[str] = os.getenv("WANDB_API_KEY")
            if not wandb_api_key:
                click.echo(
                    click.style(
                        "WANDB_API_KEY not found. Please set it in environment variables.",
                        fg="yellow",
                    )
                )
                click.echo("Get your API key from: https://wandb.ai/authorize")
                use_wandb = False
            else:
                click.echo(click.style("✓ WandB enabled", fg="green"))

                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_name,
                    config={
                        "model": model_name,
                        "epochs": epochs_val,
                        "imgsz": imgsz_val,
                        "batch": batch_size,
                        "device": device_setting or "auto",
                    },
                    tags=["yolo", "object-detection"],
                )
                click.echo(f"WandB project: {wandb_project}")
                if wandb_entity:
                    click.echo(f"WandB entity: {wandb_entity}")

        click.echo(f"Model: {model_name}")
        click.echo(f"Epochs: {epochs_val}")
        click.echo(f"Image size: {imgsz_val}")
        click.echo(f'Batch size: {batch_size if batch_size > 0 else "auto"}')
        click.echo(f'Device: {device_setting or "auto"}')

        click.echo("initializing model...")
        model: YOLO = YOLO(model_name)

        train_params: Dict[str, Any] = {
            "data": yolo_config_path,
            "epochs": epochs_val,
            "imgsz": imgsz_val,
        }

        if batch_size > 0:
            train_params["batch"] = batch_size

        if device_setting:
            train_params["device"] = device_setting

        if "project" in train_settings:
            train_params["project"] = train_settings["project"]
        if "name" in train_settings:
            train_params["name"] = train_settings["name"]

        # Train model
        click.echo("\n" + "=" * 50)
        click.echo("Starting training...")
        click.echo("=" * 50 + "\n")

        model.train(**train_params)

        click.echo(click.style("\n✓ Training completed successfully!", fg="green", bold=True))

    except FileNotFoundError as e:
        click.echo(click.style(f"✗ Config file not found: {e}", fg="red"))
        raise click.Abort()
    except Exception as e:
        click.echo(click.style(f"✗ Training failed: {str(e)}", fg="red"))
        raise click.Abort()
