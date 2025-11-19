from setuptools import setup, find_packages
import os 

long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name='bsort',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click>=8.0.0',
        'pyyaml>=6.0',
        'ultralytics>=8.0.0',
        'opencv-python>=4.5.0',
        'albumentations>=1.3.0',
        'tqdm>=4.60.0',
        'matplotlib>=3.3.0',
    ],
    entry_points={
        'console_scripts': [
            'bsort=bsort.cli:cli',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='CLI tool for YOLO training and inference',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)