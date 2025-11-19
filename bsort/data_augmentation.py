import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import albumentations as A
from dotenv import load_dotenv
from bsort.helper import read_yolo_label, write_yolo_label
import click 
from .helper import load_config
from pathlib import Path

def run_augmentation(config_path, split='train', force=False):
    """Run data augmentation on training images."""
    click.echo(click.style('\n=== Data Augmentation ===', fg='cyan', bold=True))
    
    try:
        # Load settings
        settings = load_config(config_path)
        click.echo(f'Loaded config: {config_path}')
        
        # Get paths
        base_path = Path(settings.get('base_path', '.'))
        input_dir = base_path / settings.get('input_dir', 'relabel')
        augment_dir = base_path / settings.get('augment_dir', 'relabel_aug')
        
        # Get augmentation settings
        aug_settings = settings.get('augmentation', {})
        n_aug = aug_settings.get('aug_per_image', 5)
        
        # Setup directories
        img_dir = input_dir / split / "images"
        lbl_dir = input_dir / split / "labels"
        out_img_dir = augment_dir / split / "images"
        out_lbl_dir = augment_dir / split / "labels"
        
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if augmentation already exists
        existing_count = len(list(out_img_dir.glob('*')))
        if existing_count > 0 and not force:
            click.echo(click.style(
                f'✓ Augmented images already exist ({existing_count} files). Use --force to re-augment.',
                fg='yellow'
            ))
            return
        
        if force and existing_count > 0:
            click.echo(f'Removing existing augmented images...')
            for f in out_img_dir.glob('*'):
                f.unlink()
            for f in out_lbl_dir.glob('*'):
                f.unlink()
            
        
        # Setup augmentation pipeline
        transform = A.Compose(
            [
                A.HorizontalFlip(p=aug_settings.get('horizontal_flip', 0)),
                A.VerticalFlip(p=aug_settings.get('vertical_flip', 0)),
                A.Rotate(limit=aug_settings.get('rotation_limit', 0), p=0.5),
                A.RandomBrightnessContrast(p=aug_settings.get('brightness_contrast', 0)),
                A.Blur(blur_limit=3, p=aug_settings.get('blur', 0)),
                A.ColorJitter(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=aug_settings.get('shift_limit', 0), 
                    scale_limit=aug_settings.get('scale_limit', 0), 
                    rotate_limit=15, 
                    p=0.5
                ),
            ],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )
        
        # Get images
        images = sorted(list(img_dir.glob('*.jpg')))
        
        if not images:
            click.echo(click.style(f'✗ No images found in {img_dir}', fg='red'))
            raise click.Abort()
        
        click.echo(f'\nProcessing {len(images)} images...')
        click.echo(f'Augmentations per image: {n_aug}')
        click.echo(f'Total output images: {len(images) * n_aug}\n')
        
        # Process images
        for img_path in tqdm(images, desc=f"Augmenting {split}"):
            label_path = lbl_dir / img_path.with_suffix('.txt').name
            
            image = cv2.imread(str(img_path))
            if image is None:
                click.echo(f'Warning: Could not read {img_path}')
                continue
            
            bboxes, classes = read_yolo_label(str(label_path))
            
            for i in range(n_aug):
                try:
                    aug = transform(image=image, bboxes=bboxes, class_labels=classes)
                    aug_img = aug['image']
                    aug_bboxes = aug['bboxes']
                    aug_classes = aug['class_labels']
                    
                    out_name = img_path.stem + f"_aug{i}.jpg"
                    
                    cv2.imwrite(str(out_img_dir / out_name), aug_img)
                    write_yolo_label(
                        str(out_lbl_dir / out_name.replace('.jpg', '.txt')),
                        aug_bboxes, 
                        aug_classes
                    )
                except Exception as e:
                    click.echo(f'Warning: Augmentation failed for {img_path.name} (aug {i}): {e}')
                    continue
        
        click.echo(click.style('\nAugmentation completed!', fg='green', bold=True))
        click.echo(f'Output directory: {augment_dir / split}')
        
    except FileNotFoundError as e:
        click.echo(click.style(f'File not found: {e}', fg='red'))
        raise click.Abort()
    except Exception as e:
        click.echo(click.style(f'Augmentation failed: {str(e)}', fg='red'))
        raise click.Abort()