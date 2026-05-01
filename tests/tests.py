import tensorflow as tf

from src.losses import gram_matrix, style_loss
from src.image_utils import tensor_to_image


def test_gram_matrix_shape():
    """
    Gram matrix should keep the batch dimension.
    Input:  [1, 64, 64, 128]
    Output: [1, 128, 128]
    """
    dummy_feature_map = tf.random.normal((1, 64, 64, 128))
    gram = gram_matrix(dummy_feature_map)

    assert gram.shape == (1, 128, 128)


def test_style_loss_no_nan():
    """
    Style loss should return a valid non-negative scalar.
    """
    dummy_output = [tf.random.normal((1, 32, 32, 64))]
    dummy_target = [gram_matrix(dummy_output[0])]

    loss = style_loss(dummy_output, dummy_target)

    assert not tf.math.is_nan(loss)
    assert loss.numpy() >= 0


def test_tensor_conversion_from_normalized_tensor():
    """
    Tests conversion from a normalized [0, 1] tensor to a PIL image.
    """
    dummy_tensor = tf.constant([[[[1.0, 0.0, 0.0]]]])
    img = tensor_to_image(dummy_tensor)

    assert img.size == (1, 1)