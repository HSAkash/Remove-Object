import os
import cv2
import yaml
import gdown
import numpy as np
from box import ConfigBox
from box.exceptions import BoxValueError


def read_yaml(file_path) -> ConfigBox:
    """
    Reads a YAML configuration file and returns its contents as a ConfigBox object.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        ConfigBox: Parsed YAML content accessible via dot notation.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML is malformed or cannot be converted to ConfigBox.
    """
    try:
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
            return ConfigBox(content)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"YAML file not found: {file_path}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {file_path}") from e
    except BoxValueError as e:
        raise ValueError(f"Error converting YAML to ConfigBox: {file_path}") from e

def download_model(url: str, model_path: str):
    """
    Downloads a model file from a given URL using gdown.

    Args:
        url (str): URL to the model file.
        model_path (str): Destination path to save the downloaded model.

    Notes:
        Creates directories if they do not exist.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    gdown.download(url, model_path, quiet=False, resume=True)

def norm_img(np_img: np.ndarray) -> np.ndarray:
    """
    Normalizes an image for model input:
    - Adds a channel dimension if grayscale.
    - Converts to CHW format.
    - Scales pixel values to [0, 1].

    Args:
        np_img (np.ndarray): Input image as a NumPy array.

    Returns:
        np.ndarray: Normalized image.
    """
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img

def resize_max_size(np_img: np.ndarray, size_limit: int, interpolation=cv2.INTER_CUBIC) -> np.ndarray:
    """
    Resizes an image so that its longest side is equal to or less than the given size limit.

    Args:
        np_img (np.ndarray): Input image.
        size_limit (int): Maximum size for the longer side.
        interpolation: OpenCV interpolation method.

    Returns:
        np.ndarray: Resized image.
    """
    h, w = np_img.shape[:2]
    if max(h, w) > size_limit:
        ratio = size_limit / max(h, w)
        new_w = int(w * ratio + 0.5)
        new_h = int(h * ratio + 0.5)
        return cv2.resize(np_img, dsize=(new_w, new_h), interpolation=interpolation)
    else:
        return np_img

def ceil_modulo(x: int, mod: int) -> int:
    """
    Rounds a number up to the nearest multiple of `mod`.

    Args:
        x (int): Input number.
        mod (int): Modulo base.

    Returns:
        int: Smallest integer ≥ x that is divisible by `mod`.

    Example:
        ceil_modulo(70, 16) -> 80

    Purpose:
        This function is commonly used in deep learning pipelines where image dimensions
        need to be divisible by a certain factor (e.g., 8, 16, 32), especially when:
        - Using convolutional neural networks (CNNs) with strided/downsampling layers.
        - Working with models that require consistent spatial shapes across layers.
    """
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod

def pad_img_to_modulo(img: np.ndarray, mod: int) -> np.ndarray:
    """
    Pads an image so its height and width become divisible by `mod`.

    Args:
        img (np.ndarray): Input image in CHW format (channels, height, width).
        mod (int): The modulo value to align the dimensions to.

    Returns:
        np.ndarray: Symmetrically padded image with dimensions divisible by `mod`.

    Example:
        Original size: (3, 253, 510)
        mod: 16
        Padded size: (3, 256, 512)

    Purpose:
        ✅ Why pad to a multiple (mod)?
        - Many deep learning models (e.g., U-Net, ResNet, ViT) require input sizes divisible
          by powers of 2 due to downsampling/upsampling stages.
        - Padding ensures feature maps align correctly during skip connections or reconstruction.
        - Helps with consistent batch processing and exporting models to ONNX or TensorRT.
        - Optimizes performance on GPU hardware where kernel execution prefers aligned dimensions.
    """
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )

if __name__ == "__main__":
    """
    Main execution block:
    - Downloads a pretrained model using gdown.
    - Saves the model in the `models` directory with a custom name.
    """
    from pyprojroot import here

    model_url = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
    model_path = here("models/obj_remover.pt").__str__()

    download_model(model_url, model_path)
