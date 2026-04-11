import os
import numpy as np
import tensorflow as tf
from PIL import Image
from src.config import logger

def load_img(path: str, max_dim: int = 512) -> tf.Tensor:
    """Loads an image, resizes it, and converts it to a float32 tensor."""
    if not os.path.exists(path):
        logger.error(f"Image not found: {path}")
        raise FileNotFoundError(f"Image not found at {path}")
        
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        img_arr = np.array(img).astype(np.float32)
        img_tensor = tf.convert_to_tensor(img_arr)[tf.newaxis, ...]
        return img_tensor
    except Exception as e:
        logger.error(f"Error loading image {path}: {str(e)}")
        raise

def tensor_to_image(tensor: tf.Tensor) -> Image.Image:
    """Converts a tensor back to a PIL Image."""
    tensor = tensor * 255 if tf.reduce_max(tensor) <= 1.0 else tensor
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

def save_image(tensor: tf.Tensor, output_path: str) -> None:
    """Saves a tensor as an image file."""
    img = tensor_to_image(tensor)
    img.save(output_path)
    logger.info(f"Image saved successfully to {output_path}")