import os
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import logger


def load_img(path: str, max_dim: int = 512) -> tf.Tensor:
    """
    Loads an image, resizes it while preserving aspect ratio,
    converts it to RGB, and normalizes it to [0, 1].

    Output shape:
        [1, height, width, 3]
    """
    if not os.path.exists(path):
        logger.error(f"Image not found: {path}")
        raise FileNotFoundError(f"Image not found at {path}")

    try:
        img = Image.open(path).convert("RGB")

        w, h = img.size
        logger.info(f"Original image size: {w}x{h}")

        scale = max_dim / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"Resized image size: {new_w}x{new_h}")

        # Normalize pixels from [0, 255] to [0, 1]
        img_arr = np.array(img).astype(np.float32) / 255.0

        # Add batch dimension: [H, W, 3] -> [1, H, W, 3]
        img_tensor = tf.convert_to_tensor(img_arr)[tf.newaxis, ...]

        return img_tensor

    except Exception as e:
        logger.error(f"Error loading image {path}: {str(e)}")
        raise


def tensor_to_image(tensor: tf.Tensor) -> Image.Image:
    """
    Converts a tensor back to a PIL image.

    Supports tensors in:
        [0, 1] or [0, 255]
    """
    tensor = np.array(tensor)

    if tensor.ndim > 3:
        tensor = tensor[0]

    # If the tensor is normalized, convert back to [0, 255]
    if tensor.max() <= 1.0:
        tensor = tensor * 255.0

    tensor = np.clip(tensor, 0, 255).astype(np.uint8)

    return Image.fromarray(tensor)


def save_image(tensor: tf.Tensor, output_path: str) -> None:
    """
    Saves a tensor as an image file.
    """
    img = tensor_to_image(tensor)
    img.save(output_path)

    logger.info(f"Image saved successfully to {output_path}")