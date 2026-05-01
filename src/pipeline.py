import csv
import os
import time
from typing import List

# Important: use non-GUI backend before importing pyplot.
# This prevents Tkinter crashes when Flask runs image generation.
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf

from src.config import (
    logger,
    DEFAULT_LR,
    DEFAULT_SAVE_EVERY,
    STYLE_LAYER_WEIGHTS
)
from src.image_utils import load_img, save_image
from src.nst_model import StyleContentModel
from src.losses import (
    gram_matrix,
    style_loss,
    content_loss,
    total_variation_loss
)


class StyleTransferPipeline:
    """
    Manages the full Neural Style Transfer process:
    loading images, extracting features, calculating losses,
    optimizing the generated image, and saving results.
    """

    def __init__(self, config: dict):
        self.config = config
        self.extractor = StyleContentModel()

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.get("lr", DEFAULT_LR),
            beta_1=0.99,
            epsilon=1e-1
        )

    def _blend_styles(self, style_paths: List[str], max_dim: int) -> List[tf.Tensor]:
        """
        Supports Multi-Style Transfer by averaging Gram matrices
        from multiple style images.
        """
        all_style_targets = []

        for path in style_paths:
            style_img = load_img(path, max_dim)
            style_outputs, _ = self.extractor(style_img)

            style_grams = [
                gram_matrix(output)
                for output in style_outputs
            ]

            all_style_targets.append(style_grams)

        avg_targets = []

        for layer_idx in range(len(all_style_targets[0])):
            layer_avg = tf.reduce_mean(
                [targets[layer_idx] for targets in all_style_targets],
                axis=0
            )

            avg_targets.append(layer_avg)

        return avg_targets

    def _save_loss_history(
        self,
        loss_history: List[dict],
        csv_path: str
    ) -> None:
        """
        Saves loss values to a CSV file for analysis and project documentation.
        """
        if not loss_history:
            return

        fieldnames = [
            "iteration",
            "total_loss",
            "style_loss",
            "content_loss",
            "tv_loss"
        ]

        with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(loss_history)

        logger.info(f"Loss history saved to {csv_path}")

    def _save_loss_plot(
        self,
        loss_history: List[dict],
        plot_path: str
    ) -> bool:
        """
        Saves a loss graph as an image.
        Returns True if the graph was created successfully.
        """
        if not loss_history:
            return False

        try:
            import matplotlib.pyplot as plt

            iterations = [row["iteration"] for row in loss_history]
            total_values = [row["total_loss"] for row in loss_history]
            style_values = [row["style_loss"] for row in loss_history]
            content_values = [row["content_loss"] for row in loss_history]
            tv_values = [row["tv_loss"] for row in loss_history]

            plt.figure(figsize=(10, 6))

            plt.plot(iterations, total_values, label="Total Loss")
            plt.plot(iterations, style_values, label="Style Loss")
            plt.plot(iterations, content_values, label="Content Loss")
            plt.plot(iterations, tv_values, label="TV Loss")

            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Neural Style Transfer Loss Over Iterations")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

            logger.info(f"Loss plot saved to {plot_path}")
            return True

        except Exception as e:
            logger.warning(f"Could not create loss plot: {str(e)}")
            return False

    def run(
        self,
        content_path: str,
        style_paths: List[str],
        output_path: str
    ) -> dict:
        """
        Runs the full NST process.
        """
        start_time = time.time()

        max_dim = self.config.get("max_dim", 512)
        iterations = self.config.get("iterations", 600)
        save_every = self.config.get("save_every", DEFAULT_SAVE_EVERY)

        logger.info("Loading content image...")
        content_img = load_img(content_path, max_dim)

        logger.info("Preparing style targets...")
        style_targets = self._blend_styles(style_paths, max_dim)

        logger.info("Preparing content targets...")
        _, content_targets = self.extractor(content_img)

        # Initialize generated image
        start_mode = self.config.get("start_from", "content")

        if start_mode == "noise":
            init_img = tf.random.uniform(content_img.shape, 0.0, 1.0)

        elif start_mode == "mix":
            noise = tf.random.uniform(content_img.shape, 0.0, 1.0)
            init_img = 0.7 * content_img + 0.3 * noise

        else:
            init_img = content_img

        generated_image = tf.Variable(init_img)

        weights = {
            "style": self.config.get("style_weight", 0.08),
            "content": self.config.get("content_weight", 4500.0),
            "tv": self.config.get("tv_weight", 120.0)
        }

        @tf.function
        def train_step(image):
            with tf.GradientTape() as tape:
                style_outputs, content_outputs = self.extractor(image)

                s_loss = (
                    style_loss(
                        style_outputs,
                        style_targets,
                        STYLE_LAYER_WEIGHTS
                    )
                    * weights["style"]
                )

                c_loss = (
                    content_loss(content_outputs, content_targets)
                    * weights["content"]
                )

                tv_loss = (
                    total_variation_loss(image)
                    * weights["tv"]
                )

                total_loss = s_loss + c_loss + tv_loss

            grads = tape.gradient(total_loss, image)
            self.optimizer.apply_gradients([(grads, image)])

            # Keep image pixels in valid normalized range.
            image.assign(tf.clip_by_value(image, 0.0, 1.0))

            return total_loss, s_loss, c_loss, tv_loss

        output_dir = os.path.dirname(output_path)
        output_name = os.path.splitext(os.path.basename(output_path))[0]

        loss_history = []
        intermediate_images = []

        logger.info(f"Starting optimization for {iterations} iterations...")

        for i in range(1, iterations + 1):
            total_loss_value, style_loss_value, content_loss_value, tv_loss_value = train_step(
                generated_image
            )

            row = {
                "iteration": i,
                "total_loss": float(total_loss_value.numpy()),
                "style_loss": float(style_loss_value.numpy()),
                "content_loss": float(content_loss_value.numpy()),
                "tv_loss": float(tv_loss_value.numpy())
            }

            loss_history.append(row)

            if i % 50 == 0 or i == 1:
                logger.info(
                    f"Step {i}/{iterations} | "
                    f"Total: {row['total_loss']:.4f} | "
                    f"Style: {row['style_loss']:.4f} | "
                    f"Content: {row['content_loss']:.4f} | "
                    f"TV: {row['tv_loss']:.4f}"
                )

            if save_every > 0 and i % save_every == 0:
                intermediate_filename = f"{output_name}_step_{i:04d}.png"
                intermediate_path = os.path.join(
                    output_dir,
                    intermediate_filename
                )

                save_image(generated_image, intermediate_path)
                intermediate_images.append(intermediate_filename)

        save_image(generated_image, output_path)

        csv_filename = f"{output_name}_loss_history.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        self._save_loss_history(loss_history, csv_path)

        plot_filename = f"{output_name}_loss_plot.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plot_created = self._save_loss_plot(loss_history, plot_path)

        elapsed_time = time.time() - start_time

        logger.info(f"Finished in {elapsed_time:.2f} seconds.")

        return {
            "output_path": output_path,
            "elapsed_time": elapsed_time,
            "iterations": iterations,
            "loss_csv": csv_filename,
            "loss_plot": plot_filename if plot_created else None,
            "intermediate_images": intermediate_images
        }