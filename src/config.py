import logging
from typing import List

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NST_Project")

# Default Hyperparameters
DEFAULT_MAX_DIM: int = 512
DEFAULT_ITERATIONS: int = 200
DEFAULT_STYLE_WEIGHT: float = 1e-2
DEFAULT_CONTENT_WEIGHT: float = 1.0
DEFAULT_TV_WEIGHT: float = 1e-4
DEFAULT_LR: float = 2.0

# VGG19 Layers
STYLE_LAYERS: List[str] = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
CONTENT_LAYERS: List[str] = ["block4_conv2"]