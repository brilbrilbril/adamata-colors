import os

import click
import wandb
from ultralytics import YOLO

from .helper import create_dynamic_yolo_config, load_config


def run_training(config_path, epochs=None, device=None, batch=None, imgsz=None):
    """Train YOLO model."""
    try:
        # Load settings
        settings = load_config(config_path)
        click.echo(f"Loaded config: {config_path}")

        # Create dynamic YOLO config
        yolo_config_path = create_dynamic_yolo_config(settings)

        # Get training parameters
        train_settings = settings.get("training", {})
        model_name = train_settings.get("model", "yolov9t.pt")
        epochs_val = epochs or train_settings.get("epochs", 100)
        imgsz_val = imgsz or train_settings.get("imgsz", 640)
        batch_size = batch or train_settings.get("batch", -1)
        device_setting = device or train_settings.get("device", "")

        # WandB settings
        wandb_settings = settings.get("wandb", {})
        use_wandb = wandb_settings.get("enabled", True)
        wandb_project = wandb_settings.get("project", "bsort-yolo")
        wandb_entity = wandb_settings.get("entity", None)
        wandb_name = wandb_settings.get("name", None)

        # Initialize WandB
        if use_wandb:
            # Check if API key is set
            wandb_api_key = os.getenv("WANDB_API_KEY")
            if not wandb_api_key:
                click.echo(
                    click.style(
                        "⚠ WANDB_API_KEY not found. Please set it in environment variables.",
                        fg="yellow",
                    )
                )
                click.echo("Get your API key from: https://wandb.ai/authorize")
                use_wandb = False
            else:
                click.echo(click.style("✓ WandB enabled", fg="green"))

                # Initialize WandB
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

        # Display training info
        click.echo(f"Model: {model_name}")
        click.echo(f"Epochs: {epochs_val}")
        click.echo(f"Image size: {imgsz_val}")
        click.echo(f'Batch size: {batch_size if batch_size > 0 else "auto"}')
        click.echo(f'Device: {device_setting or "auto"}')

        # Initialize model
        click.echo("\nInitializing model...")
        model = YOLO(model_name)

        # Prepare training parameters
        train_params = {
            "data": yolo_config_path,
            "epochs": epochs_val,
            "imgsz": imgsz_val,
        }

        if batch_size > 0:
            train_params["batch"] = batch_size

        if device_setting:
            train_params["device"] = device_setting

        # Add optional parameters from config
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
