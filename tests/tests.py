import pytest
import tensorflow as tf
import numpy as np
from src.losses import gram_matrix, style_loss
from src.image_utils import tensor_to_image

def test_gram_matrix_shape():
    # יצירת טנזור דמה בפורמט (Batch, Height, Width, Channels)
    dummy_feature_map = tf.random.normal((1, 64, 64, 128))
    gram = gram_matrix(dummy_feature_map)
    
    # מטריצת גראם חייבת להיות בגודל (Channels, Channels)
    assert gram.shape == (128, 128)

def test_style_loss_no_nan():
    dummy_output = [tf.random.normal((1, 32, 32, 64))]
    dummy_target = [tf.random.normal((64, 64))] # Gram shape
    
    loss = style_loss(dummy_output, dummy_target)
    assert not tf.math.is_nan(loss)
    assert loss.numpy() >= 0

def test_tensor_conversion():
    dummy_tensor = tf.constant([[[[255.0, 0.0, 0.0]]]]) # אדום
    img = tensor_to_image(dummy_tensor)
    assert img.size == (1, 1)