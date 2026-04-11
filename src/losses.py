import tensorflow as tf
from typing import List

def gram_matrix(input_tensor: tf.Tensor) -> tf.Tensor:
    """Calculates the Gram matrix for a given feature map."""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def style_loss(style_outputs: List[tf.Tensor], style_targets: List[tf.Tensor]) -> tf.Tensor:
    """Calculates the total style loss across all chosen layers."""
    loss = tf.zeros(shape=())
    weight_per_layer = 1.0 / float(len(style_outputs))
    for target, output in zip(style_targets, style_outputs):
        loss += weight_per_layer * tf.reduce_mean(tf.square(gram_matrix(output) - target))
    return loss

def content_loss(content_outputs: List[tf.Tensor], content_targets: List[tf.Tensor]) -> tf.Tensor:
    """Calculates the content loss."""
    loss = tf.zeros(shape=())
    weight_per_layer = 1.0 / float(len(content_outputs))
    for target, output in zip(content_targets, content_outputs):
        loss += weight_per_layer * tf.reduce_mean(tf.square(output - target))
    return loss

def total_variation_loss(image: tf.Tensor) -> tf.Tensor:
    """Calculates the total variation loss to reduce high-frequency noise."""
    return tf.reduce_mean(tf.image.total_variation(image))