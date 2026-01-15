import tensorflow as tf
from src.helpers.image_utils import preprocess_rgb_image

def load_gan_generator(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)

def gan_predict_mask(generator: tf.keras.Model, image: tf.Tensor, img_size: int = 256) -> tf.Tensor:
    """
    Core integration function for the rest of the project.

    Args:
        generator: loaded GAN generator model
        image: RGB tensor (H,W,3), uint8 or float
    Returns:
        mask: float32 tensor (img_size,img_size,1) in [0,1]
    """
    x = preprocess_rgb_image(image, img_size=img_size)
    x = x[tf.newaxis, ...]                 # add batch dim
    mask = generator(x, training=False)[0] # remove batch dim
    return tf.clip_by_value(mask, 0.0, 1.0)