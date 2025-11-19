"""Unit tests for utils module."""
import os
import tempfile
import pytest
import yaml
from pathlib import Path
from bsort.helper import (
    load_config,
    read_yolo_label,
    write_yolo_label,
    check_augmentation_exists,
)


def test_load_config():
    """Test loading configuration from YAML file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            'base_path': '.',
            'input_dir': 'test_input',
            'names': {0: 'class1', 1: 'class2'}
        }
        yaml.dump(config, f)
        temp_file = f.name
    
    try:
        loaded_config = load_config(temp_file)
        assert loaded_config['base_path'] == '.'
        assert loaded_config['input_dir'] == 'test_input'
        assert loaded_config['names'][0] == 'class1'
    finally:
        os.unlink(temp_file)


def test_read_write_yolo_label():
    """Test reading and writing YOLO label files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("0 0.5 0.5 0.3 0.4\n")
        f.write("1 0.2 0.3 0.1 0.2\n")
        temp_file = f.name
    
    try:
        # Test reading
        bboxes, classes = read_yolo_label(temp_file)
        assert len(bboxes) == 2
        assert len(classes) == 2
        assert classes[0] == 0
        assert classes[1] == 1
        assert bboxes[0] == [0.5, 0.5, 0.3, 0.4]
        
        # Test writing
        output_file = temp_file + '.out'
        write_yolo_label(output_file, bboxes, classes)
        
        # Read back and verify
        new_bboxes, new_classes = read_yolo_label(output_file)
        assert bboxes == new_bboxes
        assert classes == new_classes
        
        os.unlink(output_file)
    finally:
        os.unlink(temp_file)


def test_read_yolo_label_nonexistent():
    """Test reading non-existent label file."""
    bboxes, classes = read_yolo_label('nonexistent_file.txt')
    assert bboxes == []
    assert classes == []


def test_check_augmentation_exists():
    """Test checking if augmentation directory exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test when directory doesn't exist
        assert check_augmentation_exists(tmpdir) is False
        
        # Create directory structure
        aug_path = Path(tmpdir) / "relabel_aug" / "train" / "images"
        aug_path.mkdir(parents=True, exist_ok=True)
        
        # Test when directory exists but is empty
        assert check_augmentation_exists(tmpdir) is False
        
        # Add a file
        (aug_path / "test.jpg").touch()
        
        # Test when directory exists and has content
        assert check_augmentation_exists(tmpdir) is True


def test_write_yolo_label_empty():
    """Test writing empty YOLO label file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file = f.name
    
    try:
        write_yolo_label(temp_file, [], [])
        bboxes, classes = read_yolo_label(temp_file)
        assert bboxes == []
        assert classes == []
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)