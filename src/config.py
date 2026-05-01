import logging
from typing import List

# ============================================================
# Logging Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("NST_Project")


# ============================================================
# Default Hyperparameters
# ============================================================
# The project works internally with image pixels in the range [0, 1].
# Before entering VGG19, images are converted back to [0, 255]
# and passed through vgg19.preprocess_input.

DEFAULT_MAX_DIM: int = 512
DEFAULT_ITERATIONS: int = 600

# Balanced defaults after experiments.
# You can still override these values from the UI.
DEFAULT_STYLE_WEIGHT: float = 0.08
DEFAULT_CONTENT_WEIGHT: float = 4500.0
DEFAULT_TV_WEIGHT: float = 120.0
DEFAULT_LR: float = 0.004

# Save intermediate images every N iterations.
# Use 0 for faster demo runs.
DEFAULT_SAVE_EVERY: int = 0


# ============================================================
# VGG19 Layers
# ============================================================
# Updated configuration:
# - Style uses conv2 layers instead of conv1.
# - Content uses block5_conv1 instead of block4_conv2.
#
# Reason:
# conv2 style layers can give richer and more stable style representation.
# block5_conv1 content layer preserves higher-level content structure
# and may allow style to appear more strongly.

STYLE_LAYERS: List[str] = [
    "block1_conv2",
    "block2_conv2",
    "block3_conv2",
    "block4_conv2",
    "block5_conv2",
]

# Lower layers capture small textures and colors.
# Higher layers capture larger visual patterns.
# Giving lower weight to early layers helps reduce noisy tiny textures.
STYLE_LAYER_WEIGHTS: List[float] = [
    0.05,
    0.15,
    0.25,
    0.30,
    0.25,
]

CONTENT_LAYERS: List[str] = [
    "block5_conv1"
]

# Replacing MaxPooling with AveragePooling often produces smoother
# Neural Style Transfer results.
USE_AVG_POOLING: bool = True


# ============================================================
# Validation
# ============================================================

if len(STYLE_LAYER_WEIGHTS) != len(STYLE_LAYERS):
    raise ValueError(
        "STYLE_LAYER_WEIGHTS must have the same length as STYLE_LAYERS"
    )