import glob
import os
from typing import Optional, List, Dict, Any

import click
import cv2
from tqdm import tqdm
from ultralytics import YOLO

from .helper import load_config


def run_inference(
    config_path: str,
    image: Optional[str] = None,
    dir: Optional[str] = None,
    model: Optional[str] = None,
    conf: Optional[float] = None,
    save: bool = False,
    show: bool = False
) -> None:
    """Run YOLO inference on single or multiple images.

    Performs object detection using a trained YOLO model on specified images.
    Results can be displayed interactively and/or saved to disk with bounding
    boxes and labels visualized.

    Args:
        config_path: Path to the YAML configuration file containing inference settings.
        image: Path to a single image file for inference. Mutually exclusive with dir.
            Defaults to None.
        dir: Path to a directory containing multiple images for batch inference.
            Mutually exclusive with image. Defaults to None.
        model: Path to the YOLO model weights file (.pt). If None, uses the path
            specified in config. Defaults to None.
        conf: Confidence threshold for detections (0.0 to 1.0). If None, uses
            the threshold specified in config. Defaults to None.
        save: If True, saves annotated images with detections to
            'runs/detect/predict/'. Defaults to False.
        show: If True, displays each annotated image interactively using matplotlib.
            Defaults to False.

    Returns:
        None

    Raises:
        click.Abort: If required files/directories are not found, no images are
            specified, or inference fails.
        FileNotFoundError: If the config file cannot be found.
        Exception: If an unexpected error occurs during inference.

    Examples:
        >>> # Run inference on a single image
        >>> run_inference('config.yaml', image='test.jpg', save=True)
        
        >>> # Run inference on a directory with custom confidence
        >>> run_inference('config.yaml', dir='images/', conf=0.5, show=True)
        
        >>> # Use default settings from config
        >>> run_inference('config.yaml')

    Note:
        - If neither image nor dir is specified, the function attempts to use
          the default image directory from the config file.
        - Only .jpg images are processed when using directory mode.
        - The --save flag creates output in 'runs/detect/predict/' directory.
        - The --show flag requires a display environment (won't work headless).
    """
    click.echo(click.style("\n=== Inference ===", fg="cyan", bold=True))

    try:
        settings: Dict[str, Any] = load_config(config_path)
        click.echo(f"Loaded config: {config_path}")

        infer_settings: Dict[str, Any] = settings.get("inference", {})
        model_path: str = model or infer_settings.get(
            "model", "runs/detect/train/weights/best.pt"
        )
        confidence: float = conf or infer_settings.get("conf", 0.25)

        image_paths: List[str] = []
        if image:
            if not os.path.exists(image):
                click.echo(click.style(f"Image not found: {image}", fg="red"))
                raise click.Abort()
            image_paths = [image]
        elif dir:
            if not os.path.exists(dir):
                click.echo(click.style(f"Directory not found: {dir}", fg="red"))
                raise click.Abort()
            image_paths = glob.glob(os.path.join(dir, "*.jpg"))
            if not image_paths:
                click.echo(click.style(f"No .jpg images found in {dir}", fg="red"))
                raise click.Abort()
        else:
            default_dir: str = infer_settings.get("image_dir", "unseen")
            if os.path.exists(default_dir):
                image_paths = glob.glob(os.path.join(default_dir, "*.jpg"))

            if not image_paths:
                click.echo(
                    click.style("No images specified. Use --image or --dir", fg="red")
                )
                raise click.Abort()

        if not os.path.exists(model_path):
            click.echo(click.style(f"Model not found: {model_path}", fg="red"))
            raise click.Abort()

        click.echo(f"Model: {model_path}")
        click.echo(f"Images: {len(image_paths)}")
        click.echo(f"Confidence threshold: {confidence}")

        click.echo("Loading model...")
        yolo_model: YOLO = YOLO(model_path)

        # show the image result if --show enabled
        if show:
            import matplotlib.pyplot as plt

        click.echo("Running inference...")

        for path in tqdm(image_paths, desc="Processing images"):
            result = yolo_model(path, conf=confidence)[0]

            boxes = result.boxes
            click.echo(f"\n{os.path.basename(path)}:")

            if len(boxes) > 0:
                for box in boxes:
                    cls: int = int(box.cls[0])
                    conf_score: float = float(box.conf[0])
                    class_name: str = result.names[cls]
                    click.echo(f"  • {class_name}: {conf_score:.2f}")
            else:
                click.echo("  No detections")

            # Save or show results
            if save:
                img = result.plot()
                output_dir: str = "runs/detect/predict"
                os.makedirs(output_dir, exist_ok=True)
                output_path: str = os.path.join(output_dir, os.path.basename(path))
                cv2.imwrite(output_path, img)

            if show:
                img = result.plot()
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(12, 8))
                plt.imshow(rgb)
                plt.title(os.path.basename(path))
                plt.axis("off")
                plt.show()

        if save:
            click.echo(
                click.style(f"Results saved to runs/detect/predict/", fg="green")
            )

        click.echo(click.style("Inference completed!", fg="green", bold=True))

    except FileNotFoundError as e:
        click.echo(click.style(f"File not found: {e}", fg="red"))
        raise click.Abort()
    except Exception as e:
        click.echo(click.style(f"Inference failed: {str(e)}", fg="red"))
        raise click.Abort()