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



- Installation:
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