"""
Application pipeline entry point.

This file orchestrates:
- GAN segmentation
- (Autoencoder refinement – disabled for now)

During development, we can intercept and visualize intermediate outputs.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Add project root to PYTHONPATH
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from src.gan_module.inference import load_gan_generator, gan_predict_mask
from src.helpers.image_utils import (
    load_rgb_image,
    visualize_segmentation_result,
)

# ---------------------------------------------------------------------
# Load models once
# ---------------------------------------------------------------------
GAN_MODEL_PATH = "./models/gan_generator.keras"
gan = load_gan_generator(GAN_MODEL_PATH)

# ---------------------------------------------------------------------
def run_gan_only(
    image: tf.Tensor,
    save_path: str = None,
    visualize: bool = True,
) -> tf.Tensor:
    """
    Run GAN only and optionally visualize/save output.

    Args:
        image: RGB image tensor (H, W, 3)
        save_path: where to save the visualization PNG
        visualize: whether to show/save visualization

    Returns:
        rough_mask: (256,256,1) float tensor
    """
    rough_mask = gan_predict_mask(gan, image)

    if visualize:
        visualize_segmentation_result(
            image=image,
            mask=rough_mask,
            threshold=0.5,
            title="GAN Output",
            save_path=save_path,
        )

    return rough_mask


# ---------------------------------------------------------------------
if __name__ == "__main__":
    IMAGE_DIR = PROJECT_ROOT / "test_images"
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    print(f"Found {len(image_paths)} test images")

    for img_path in image_paths:
        print(f"\nProcessing {img_path.name}")

        image = load_rgb_image(str(img_path))

        # Build output filename: original + _gan_only.png
        save_path = (
            img_path.with_suffix("").as_posix() + "_gan_only.png"
        )

        _ = run_gan_only(
            image=image,
            save_path=save_path,
            visualize=True,
        )