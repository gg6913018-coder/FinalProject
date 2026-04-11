import time
import os
import tensorflow as tf
from typing import List, Optional
from src.config import logger
from src.image_utils import load_img, save_image
from src.nst_model import StyleContentModel
from src.losses import gram_matrix, style_loss, content_loss, total_variation_loss

class StyleTransferPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.extractor = StyleContentModel()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.get('lr', 2.0), beta_1=0.99, epsilon=1e-1)
        
    def _blend_styles(self, style_paths: List[str], max_dim: int) -> List[tf.Tensor]:
        """Supports Multi-Style by averaging Gram matrices from multiple images."""
        all_style_targets = []
        for path in style_paths:
            style_img = load_img(path, max_dim)
            style_outputs, _ = self.extractor(style_img)
            all_style_targets.append([gram_matrix(out) for out in style_outputs])
        
        # Average the targets
        avg_targets = []
        for layer_idx in range(len(all_style_targets[0])):
            layer_avg = tf.reduce_mean([targets[layer_idx] for targets in all_style_targets], axis=0)
            avg_targets.append(layer_avg)
        return avg_targets

    def run(self, content_path: str, style_paths: List[str], output_path: str) -> dict:
        start_time = time.time()
        max_dim = self.config.get('max_dim', 512)
        
        content_img = load_img(content_path, max_dim)
        style_targets = self._blend_styles(style_paths, max_dim)
        _, content_targets = self.extractor(content_img)

        # Initialize image
        start_mode = self.config.get('start_from', 'content')
        if start_mode == 'noise':
            init_img = tf.random.uniform(content_img.shape, 0.0, 255.0)
        elif start_mode == 'mix':
            init_img = 0.7 * content_img + 0.3 * tf.random.uniform(content_img.shape, 0.0, 255.0)
        else:
            init_img = content_img
            
        generated_image = tf.Variable(init_img)
        
        weights = {
            'style': self.config.get('style_weight', 1e-2),
            'content': self.config.get('content_weight', 1.0),
            'tv': self.config.get('tv_weight', 1e-4)
        }

        @tf.function
        def train_step(image):
            with tf.GradientTape() as tape:
                style_outputs, content_outputs = self.extractor(image)
                s_loss = style_loss(style_outputs, style_targets) * weights['style']
                c_loss = content_loss(content_outputs, content_targets) * weights['content']
                tv_loss = total_variation_loss(image) * weights['tv']
                loss = s_loss + c_loss + tv_loss

            grads = tape.gradient(loss, image)
            self.optimizer.apply_gradients([(grads, image)])
            image.assign(tf.clip_by_value(image, 0.0, 255.0))
            return loss

        iterations = self.config.get('iterations', 200)
        logger.info(f"Starting optimization for {iterations} iterations...")
        
        for i in range(1, iterations + 1):
            loss = train_step(generated_image)
            if i % 50 == 0:
                logger.info(f"Step {i}/{iterations} - Loss: {loss:.2f}")

        save_image(generated_image, output_path)
        elapsed_time = time.time() - start_time
        logger.info(f"Finished in {elapsed_time:.2f} seconds.")
        
        return {
            "output_path": output_path,
            "elapsed_time": elapsed_time,
            "iterations": iterations
        }