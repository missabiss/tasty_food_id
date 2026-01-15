import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def load_rgb_image(path: str) -> tf.Tensor:
    """
    Load an RGB image from disk.
    Returns: (H, W, 3) uint8 tensor
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    return img

def preprocess_rgb_image(image: tf.Tensor, img_size: int = 256) -> tf.Tensor:
    """
    Standardize how we prepare input images for ALL models (GAN/AE/CNN).

    Args:
        image: RGB image tensor, shape (H, W, 3), dtype uint8 or float.
        img_size: final size (img_size, img_size)

    Returns:
        image: float32 tensor in [0, 1], shape (img_size, img_size, 3)
    """
    image = tf.image.resize(image, (img_size, img_size))
    image = tf.cast(image, tf.float32)

    # If image is uint8 originally, values are 0..255. Normalize.
    # If image is already float in 0..1, this will still behave fine if max <= 1.
    image = tf.cond(
        tf.reduce_max(image) > 1.5,
        lambda: image / 255.0,
        lambda: image
    )
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_mask(mask: tf.Tensor, img_size: int = 256) -> tf.Tensor:
    """
    Standardize ground-truth masks.

    Args:
        mask: mask tensor, could be (H, W), (H, W, 1), uint8/float, binary or multi-class.
    Returns:
        mask: float32 tensor in [0,1], shape (img_size, img_size, 1)
    """
    if tf.rank(mask) == 2:
        mask = mask[..., tf.newaxis]

    mask = tf.image.resize(mask, (img_size, img_size), method="nearest")
    mask = tf.cast(mask, tf.float32)

    # Convert "anything nonzero" -> 1.0 (binary food vs background)
    mask = tf.where(mask > 0.0, 1.0, 0.0)
    return mask


def visualize_segmentation_result(
    image: tf.Tensor,
    mask: tf.Tensor,
    threshold: float = 0.5,
    title: str = None,
    save_path: str = None,
):
    """
    Visualize input image, soft mask, and binary mask side-by-side.
    Optionally save the figure to disk.
    """
    image_np = image.numpy()
    mask_np = mask.numpy()[..., 0]
    binary = (mask_np > threshold).astype(np.float32)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap="gray")
    plt.title("GAN Mask (Soft)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(binary, cmap="gray")
    plt.title("GAN Mask (Binary)")
    plt.axis("off")

    if title:
        plt.suptitle(title)

    plt.tight_layout()

    # SAVE THE COMPOSITE FIGURE
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()
    plt.close()