Adamata Take Home Test: Bottle Caps Color Detection

By: Brillyando Magathan Achmad

- Project Summary:
    Given images that consist of several bottle caps, the project's objectives are to detect and classify the colors into 3 classes: light blue (0), dark blue (1), and others (2).

    This project uses YOLOv9 Tiny version, since it's the most lightweight model, but still has decent performance among the newer one. https://docs.ultralytics.com/models/yolov9/ 

    Why do I choose YOLOv9 Tiny, instead of YOLO11n? Because the objective of this test is to integrate the system into edge device, such as raspberry pi 5. YOLO11n is exactly better in the performance, but it has 3M parameters, so it's very doubtful when it comes to inference - since this tests aim to run on 5-10ms speed.

- Dataset Overview:
    The dataset consists of 12 images along with its label. But, the labels itself do not represent the actual color (ground truth) yet. So I have to assign the label by myself. Here I experimented with 2 methods, which is rule-based automatic label and manual assignment.

    1. Rule-based
        The rule-based method is assign the color by the HSV threshold. For example, the blue color is likely in 88 <= H <= 110. Then the difference between light blue and dark blue is in the V, which is approx 95. Using this method will save my time to annotate the ground truth of the colors. But, it's so difficult to tweak the threshold since ground truths must be 100% correct.

    2. Manual assignment
        Since there's no explanation in the description that the bottle caps appear in an image have different color, so I decided to assign it the same because it looks so identical (at least for me). In other words, for an image, the bottle caps have the same color.

    To enrich the dataset, I do the augmentation using albumentations, thus the model training can leverage the augmented dataset. Of course if you want to train by yourself, you can decide to do augmentation or not (read more in the CLI commands below)

- Result:
      Using CPU only on my machine:
          Training and validation:
  <img width="1647" height="859" alt="Screenshot 2025-11-20 214734" src="https://github.com/user-attachments/assets/22aa67a1-b562-478e-a3be-79a174db7b48" />
  <img width="1686" height="525" alt="Screenshot 2025-11-20 214752" src="https://github.com/user-attachments/assets/14652035-4b21-4d76-98a9-4e0c4023b1fa" />
          Inference:
  <img width="1769" height="791" alt="Screenshot 2025-11-20 214918" src="https://github.com/user-attachments/assets/903f0187-72ef-4b2b-8a52-6b7b1d524e87" />

  You can see that the training time for 100 epochs was 1 hour with validation mAP50 reached 0.995. This indicates overfit due to lack of dataset.
  The inference reached speed time 43-48ms which is far from the objective.

      Using GPU:
          Training and validation:
  <img width="1736" height="668" alt="image" src="https://github.com/user-attachments/assets/28340450-5693-433b-840c-24593745b4dd" />
  <img width="1640" height="523" alt="Screenshot 2025-11-20 222226" src="https://github.com/user-attachments/assets/4963115c-b041-4ed4-90cf-b0ede1d090bd" />
          Inference:
  <img width="1767" height="919" alt="Screenshot 2025-11-20 222305" src="https://github.com/user-attachments/assets/00759119-39ca-44af-8722-08e0c7da1fff" />

  You can see that the training and validation mAP50 did not differ that much. But the inference time decreased until it reached 10-15ms per image inference. This was the most challenging one because the objective still did not achieve yet even after using GPU Cuda supported.

- Installation:

    **IMPORTANT: MAKE SURE YOUR PYTHON VERSION IS 3.10.x, otherwise it's not working (because I have no time to test using another version)**
    1. Clone this repository

    ```
    git clone https://github.com/brilbrilbril/adamata-colors.git
    ```

    2. Head to root project directory

    ```
    cd adamata-colors
    ```

    3. Install poetry. Please refer to its official documentation: https://python-poetry.org/docs/

    4. Make sure your python version is 3.10+, otherwise this project wont work.

    5. Install the dependencies

    ```
    poetry install
    ```

    6. Activate the virtual environment

    ```
    poetry env activate
    ``` 

    OR 

    ```
    poetry self add poetry-plugin-shell

    poetry shell
    ```

    7. Now you can augment the images, train model, or do inference.

    **NOTE:** If you want have CUDA GPU and want to utilize it for training, please uninstall and install the correct pytorch dependency for GPU utilization

    ```
    poetry run pip uninstall -y torch torchvision 
    ```

    ```
    poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    ```

    Then you can insert the ```--device 0``` in the command. 


- Running recommendation:
1. Run augmentation first
2. Set the wandb API KEY if you want to monitor
3. Run the training command.
4. Do inference with --show or --save enabled


- CLI Command Documentations:

**Base Command:** `bsort`

**Description:** CLI tool for YOLO training and inference.

### 1\. Data Augmentation: `augment`

This command runs data augmentation on your training images.

**Usage:**

```bash
bsort augment [OPTIONS]
```

| Option | Shorthand | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--config` | `-c` | **Yes** | N/A | Path to **`settings.yaml`** configuration file. |
| `--split` | `-s` | No | `train` | Dataset split to augment (e.g., `train`, `val`). |
| `--force` | (Flag) | No | N/A | **Force** re-augmentation even if augmented files already exist. |

**Example:**

```bash
# Run augmentation on the 'train' split using a configuration file
bsort augment --config path/to/settings.yaml

# Force re-augmentation on the 'val' split
bsort augment -c path/to/settings.yaml -s val --force
```

-----

### 2\. Model Training: `train`

This command trains a YOLO model based on the specified configuration and optional overrides.

**Usage:**

```bash
bsort train [OPTIONS]
```

| Option | Shorthand | Required | Default | Type | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--config` | `-c` | **Yes** | N/A | String | Path to **`settings.yaml`** configuration file. |
| `--epochs` | `-e` | No | Config value | Integer | Override the number of training epochs from the config. |
| `--device` | `-d` | No | Config value | String | Device to use for training (e.g., `0`, `1`, ` ` for auto ). |
| `--batch` | `-b` | No | Config value | Integer | Override the training batch size. |
| `--imgsz` | N/A | No | Config value | Integer | Override the input image size (e.g., `640`). |

**Example:**

```bash
# Basic training using config file
bsort train --config path/to/settings.yaml

# Override epochs and use CPU
bsort train -c path/to/settings.yaml -e 50 -d cpu
```

-----

### 3\. Inference: `infer`

This command runs object detection (inference) on single images or a directory of images.

**Usage:**

```bash
bsort infer [OPTIONS]
```

| Option | Shorthand | Required | Default | Type | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--config` | `-c` | **Yes** | N/A | String | Path to **`settings.yaml`** configuration file. |
| `--image` | `-i` | No | N/A | String | Path to a **single image** file for inference. |
| `--dir` | `-d` | No | N/A | String | **Directory** containing images to run inference on. |
| `--model` | `-m` | No | Config value | String | Override the path to the trained model weights. |
| `--conf` | N/A | No | Config value | Float | **Confidence threshold** for detection filtering (0.0 to 1.0). |
| `--save` | (Flag) | No | N/A | Boolean | **Save** the inference results (images/metadata) to a file. |
| `--show` | (Flag) | No | N/A | Boolean | **Display** the results using a tool like Matplotlib. |

**Note:** You must provide either the `--image` or the `--dir` option to specify the input data.

**Example:**

```bash
# Run inference on a single image and save the result
bsort infer -c path/to/settings.yaml -i path/to/image.jpg --save

# Run inference on a directory of images, display results, and use a custom confidence threshold
bsort infer -c path/to/settings.yaml -d path/to/images_folder --show --conf 0.4
```


**You can check the wandb.ai training monitor here:** https://wandb.ai/maghatan-a/bsort-yolo
