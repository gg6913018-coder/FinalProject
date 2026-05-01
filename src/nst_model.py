import tensorflow as tf
from typing import Tuple, List

from src.config import (
    STYLE_LAYERS,
    CONTENT_LAYERS,
    USE_AVG_POOLING,
    logger
)


def build_feature_extractor(
    style_layers: List[str],
    content_layers: List[str],
    use_avg_pooling: bool = USE_AVG_POOLING
) -> tf.keras.Model:
    """
    Builds a VGG19 feature extractor.

    If use_avg_pooling=True, MaxPooling2D layers are replaced with
    AveragePooling2D layers. This is a common NST improvement that can produce
    smoother and less noisy stylized images.
    """
    logger.info("Building VGG19 feature extractor...")

    vgg = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet"
    )

    vgg.trainable = False

    # Simple path: use the original VGG19 as-is.
    if not use_avg_pooling:
        outputs = [
            vgg.get_layer(name).output
            for name in (style_layers + content_layers)
        ]

        model = tf.keras.Model([vgg.input], outputs)
        model.trainable = False

        logger.info("Using original VGG19 MaxPooling layers.")
        return model

    # Advanced path: rebuild VGG19 feature extractor while replacing
    # MaxPooling2D with AveragePooling2D.
    logger.info("Using AveragePooling instead of MaxPooling.")

    inputs = tf.keras.Input(shape=(None, None, 3))
    x = inputs

    layer_outputs = {}

    for layer in vgg.layers[1:]:  # Skip original InputLayer
        if isinstance(layer, tf.keras.layers.Conv2D):
            new_layer = tf.keras.layers.Conv2D.from_config(
                layer.get_config()
            )

            x = new_layer(x)
            new_layer.set_weights(layer.get_weights())

        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            config = layer.get_config()

            new_layer = tf.keras.layers.AveragePooling2D(
                pool_size=config["pool_size"],
                strides=config["strides"],
                padding=config["padding"],
                name=layer.name
            )

            x = new_layer(x)

        else:
            new_layer = layer.__class__.from_config(layer.get_config())
            x = new_layer(x)

            if layer.get_weights():
                new_layer.set_weights(layer.get_weights())

        layer_outputs[layer.name] = x

    outputs = [
        layer_outputs[name]
        for name in (style_layers + content_layers)
    ]

    model = tf.keras.Model(inputs, outputs)
    model.trainable = False

    return model


class StyleContentModel(tf.keras.models.Model):
    """
    Extracts style and content representations from an image.

    The project stores images internally in [0, 1].
    Before sending the image to VGG19, the image is converted back to [0, 255]
    and then processed with vgg19.preprocess_input.
    """

    def __init__(
        self,
        style_layers: List[str] = STYLE_LAYERS,
        content_layers: List[str] = CONTENT_LAYERS,
        use_avg_pooling: bool = USE_AVG_POOLING
    ):
        super(StyleContentModel, self).__init__()

        self.vgg = build_feature_extractor(
            style_layers,
            content_layers,
            use_avg_pooling
        )

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)

        self.vgg.trainable = False

    def call(self, inputs: tf.Tensor) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        """
        Expects input image tensor in [0, 1].

        Returns:
            style_outputs, content_outputs
        """
        # VGG19 preprocess_input expects image values in the [0, 255] range.
        inputs = inputs * 255.0
        inputs = tf.keras.applications.vgg19.preprocess_input(inputs)

        outputs = self.vgg(inputs)

        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]

        return style_outputs, content_outputs