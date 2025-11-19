import click

from .data_augmentation import run_augmentation
from .inference import run_inference
from .model_train import run_training


@click.group()
def cli():
    """bsort - CLI tool for YOLO training and inference."""
    pass


@cli.command()
@click.option("--config", "-c", required=True, help="Path to settings.yaml")
@click.option("--split", "-s", default="train", help="Dataset split to augment (default: train)")
@click.option("--force", is_flag=True, help="Force re-augmentation even if exists")
def augment(config, split, force):
    """Run data augmentation on training images."""
    run_augmentation(config, split, force)


@cli.command()
@click.option("--config", "-c", required=True, help="Path to settings.yaml")
@click.option("--epochs", "-e", default=None, type=int, help="Override epochs from config")
@click.option("--device", "-d", default=None, help="Device to use (e.g., 0, cpu)")
@click.option("--batch", "-b", default=None, type=int, help="Batch size")
@click.option("--imgsz", default=None, type=int, help="Image size")
def train(config, epochs, device, batch, imgsz):
    """Train YOLO model."""
    run_training(config, epochs, device, batch, imgsz)


@cli.command()
@click.option("--config", "-c", required=True, help="Path to settings.yaml")
@click.option("--image", "-i", default=None, help="Path to single image file")
@click.option("--dir", "-d", default=None, help="Directory containing images")
@click.option("--model", "-m", default=None, help="Override model path from config")
@click.option("--conf", default=None, type=float, help="Confidence threshold")
@click.option("--save", is_flag=True, help="Save results to file")
@click.option("--show", is_flag=True, help="Display results with matplotlib")
def infer(config, image, dir, model, conf, save, show):
    """Run inference on image(s)."""
    run_inference(config, image, dir, model, conf, save, show)


if __name__ == "__main__":
    cli()
