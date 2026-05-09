import tensorflow as tf
from typing import List, Optional


def gram_matrix(input_tensor: tf.Tensor) -> tf.Tensor:
    """
    Calculates the Gram matrix for a given feature map.

    Input shape:
        [batch, height, width, channels]

    Output shape:
        [batch, channels, channels]
    """
    result = tf.linalg.einsum( 
        "bijc,bijd->bcd", 
        input_tensor,
        input_tensor
    ) # Computes the inner product of the feature maps across spatial dimensions

    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

    return result / num_locations


def style_loss(
    style_outputs: List[tf.Tensor],
    style_targets: List[tf.Tensor],
    layer_weights: Optional[List[float]] = None
) -> tf.Tensor:
    """
    Calculates the total style loss across all chosen style layers.

    If layer_weights is provided, each style layer receives a different weight.
    This helps reduce noisy low-level texture and emphasize larger style patterns.
    """
    loss = tf.zeros(shape=(), dtype=tf.float32)

    if layer_weights is None:
        layer_weights = [1.0 / float(len(style_outputs))] * len(style_outputs)

    if len(layer_weights) != len(style_outputs):
        raise ValueError(
            "layer_weights length must match number of style outputs"
        )

    total_weight = float(sum(layer_weights))
    normalized_weights = [w / total_weight for w in layer_weights]

    for target, output, weight in zip(
        style_targets,
        style_outputs,
        normalized_weights
    ):
        current_gram = gram_matrix(output)

        loss += tf.cast(weight, tf.float32) * tf.reduce_mean(
            tf.square(current_gram - target)
        )

    return loss


def content_loss(
    content_outputs: List[tf.Tensor],
    content_targets: List[tf.Tensor]
) -> tf.Tensor:
    """
    Calculates the content loss.
    """
    loss = tf.zeros(shape=(), dtype=tf.float32)

    weight_per_layer = 1.0 / float(len(content_outputs)) # Equal weight for each content layer

    for target, output in zip(content_targets, content_outputs):
        loss += weight_per_layer * tf.reduce_mean( 
            tf.square(output - target) # Mean squared error between content features
        )

    return loss


def total_variation_loss(image: tf.Tensor) -> tf.Tensor:
    """
    Calculates total variation loss to reduce high-frequency noise.
    """
    return tf.reduce_mean(tf.image.total_variation(image)) 