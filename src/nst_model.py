import tensorflow as tf
from typing import Tuple, List
from src.config import STYLE_LAYERS, CONTENT_LAYERS, logger

def build_feature_extractor(style_layers: List[str], content_layers: List[str]) -> tf.keras.Model:
    """Builds a VGG19 model that outputs specific layer activations."""
    logger.info("Building VGG19 feature extractor...")
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    model = tf.keras.Model([vgg.input], outputs)
    return model

class StyleContentModel(tf.keras.models.Model):
    """Encapsulates the feature extraction process."""
    def __init__(self, style_layers: List[str] = STYLE_LAYERS, content_layers: List[str] = CONTENT_LAYERS):
        super(StyleContentModel, self).__init__()
        self.vgg = build_feature_extractor(style_layers, content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs: tf.Tensor) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        """Expects float input in [0, 255]. Returns (style_outputs, content_outputs)."""
        inputs = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(inputs)
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]
        return style_outputs, content_outputs