import os
import cv2
import glob
import click
from tqdm import tqdm
from ultralytics import YOLO
from .helper import load_config


def run_inference(config_path, image=None, dir=None, model=None, conf=None, save=False, show=False):
    """Run inference on image(s)."""
    click.echo(click.style('\n=== Inference ===', fg='cyan', bold=True))
    
    try:
        # Load settings
        settings = load_config(config_path)
        click.echo(f'Loaded config: {config_path}')
        
        # Get inference parameters
        infer_settings = settings.get('inference', {})
        model_path = model or infer_settings.get('model', 'runs/detect/train/weights/best.pt')
        confidence = conf or infer_settings.get('conf', 0.25)
        
        # Get image paths
        image_paths = []
        if image:
            if not os.path.exists(image):
                click.echo(click.style(f'Image not found: {image}', fg='red'))
                raise click.Abort()
            image_paths = [image]
        elif dir:
            if not os.path.exists(dir):
                click.echo(click.style(f'Directory not found: {dir}', fg='red'))
                raise click.Abort()
            image_paths = glob.glob(os.path.join(dir, '*.jpg'))
            if not image_paths:
                click.echo(click.style(f'No .jpg images found in {dir}', fg='red'))
                raise click.Abort()
        else:
            # Use default from config
            default_dir = infer_settings.get('image_dir', 'unseen')
            if os.path.exists(default_dir):
                image_paths = glob.glob(os.path.join(default_dir, '*.jpg'))
            
            if not image_paths:
                click.echo(click.style('No images specified. Use --image or --dir', fg='red'))
                raise click.Abort()
        
        # Check if model exists
        if not os.path.exists(model_path):
            click.echo(click.style(f'Model not found: {model_path}', fg='red'))
            raise click.Abort()
        
        # Display inference info
        click.echo(f'Model: {model_path}')
        click.echo(f'Images: {len(image_paths)}')
        click.echo(f'Confidence threshold: {confidence}')
        
        # Load model
        click.echo('\nLoading model...')
        yolo_model = YOLO(model_path)
        
        # Show matplotlib if requested
        if show:
            import matplotlib.pyplot as plt
        
        # Run inference
        click.echo('Running inference...\n')
        
        for path in tqdm(image_paths, desc="Processing images"):
            result = yolo_model(path, conf=confidence)[0]
            
            # Display results
            boxes = result.boxes
            click.echo(f'\n{os.path.basename(path)}:')
            
            if len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    class_name = result.names[cls]
                    click.echo(f'  • {class_name}: {conf_score:.2f}')
            else:
                click.echo('  No detections')
            
            # Save or show results
            if save:
                img = result.plot()
                output_dir = 'runs/detect/predict'
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, os.path.basename(path))
                cv2.imwrite(output_path, img)
            
            if show:
                img = result.plot()
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(12, 8))
                plt.imshow(rgb)
                plt.title(os.path.basename(path))
                plt.axis('off')
                plt.show()
        
        if save:
            click.echo(click.style(f'\n✓ Results saved to runs/detect/predict/', fg='green'))
        
        click.echo(click.style('\n✓ Inference completed!', fg='green', bold=True))
        
    except FileNotFoundError as e:
        click.echo(click.style(f'✗ File not found: {e}', fg='red'))
        raise click.Abort()
    except Exception as e:
        click.echo(click.style(f'✗ Inference failed: {str(e)}', fg='red'))
        raise click.Abort()
