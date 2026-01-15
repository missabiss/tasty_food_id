from typing import Callable, Optional, Tuple
import tensorflow as tf
import tensorflow_datasets as tfds
from .config import AEConfig

GanMaskFn = Optional[Callable[[tf.Tensor], tf.Tensor]]

def create_dummy_mask(image: tf.Tensor, img_size: int) -> tf.Tensor:
    '''
    create a dummy mask for the image
    normally this would be a GAN mask
    this is a placeholder for the GAN mask function created by the GAN team member
    '''
    image = tf.image.resize(image, (img_size, img_size))
    image = tf.image.rgb_to_grayscale(image)
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    # this dummy mask will create a mask guessing where food is based on the greyscale value
    mask = tf.where(image >0.5, 1.0, 0.0)

    return mask


def add_noise_to_mask(mask: tf.Tensor) -> tf.Tensor:
    """
    Adds salt and pepper noise to the input mask.
    mask: input mask tensor [H, W, 1]
    returns: noisy mask tensor [H, W, 1]
    """
    noise = tf.random.uniform(tf.shape(mask), minval=0.0, maxval=1.0)
    # this is creating white and black pixels where the noise values are at their extremes, creating a salt and pepper effect.
    s = tf.cast(noise > 0.7, tf.float32)
    p = tf.cast(noise < 0.3, tf.float32)

    noisy_mask = mask + .35*s - .35*p
    noisy_mask = tf.clip_by_value(noisy_mask, 0.0, 1.0)

    return noisy_mask

def make_preprocess_mask_function(
        gan_mask_function: Optional[GanMaskFn] = None,
        add_noise: bool = True
) -> Callable[[dict], Tuple[tf.Tensor, tf.Tensor]]:
    """
    Creates a preprocessing function for generating input masks for the autoencoder.
    """
    def _preprocess_mask(sample: dict) -> Tuple[tf.Tensor, tf.Tensor]:
        image = sample["image"]
        clean_mask = gan_mask_function(image) if gan_mask_function else create_dummy_mask(image)
        clean_mask = tf.cast(clean_mask, tf.float32)

        # ensure channel dimensions if not existent
        if len(clean_mask.shape) == 2:
            clean_mask = clean_mask[..., tf.newaxis]
        clean_mask = tf.clip_by_value(clean_mask, 0.0, 1.0)

        noisy_mask = add_noise_to_mask(clean_mask) if add_noise else clean_mask

        return noisy_mask, clean_mask
    
    return _preprocess_mask

def load_food_101_splits() -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the Food-101 dataset and splits it into training and validation sets.
    """
    train_ds, val_ds = tfds.load(
        "food101",
        split=["train[:90%]", "train[90%:]"],
        as_supervised=False
    )
    return train_ds, val_ds

def build_food101_pipeline(
        config: AEConfig,
        gan_mask_function: Optional[GanMaskFn] = None,
        add_noise: bool = True,
        train_ds: Optional[tf.data.Dataset] = None,
        val_ds: Optional[tf.data.Dataset] = None
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    
    """
    if train_ds is None or val_ds is None:
        train_ds, val_ds = load_food_101_splits()

    preprocess_fn = make_preprocess_mask_function(
        gan_mask_function=gan_mask_function,
        add_noise=add_noise
    )
    autotune = tf.data.AUTOTUNE

    train_pipeline = (
        train_ds
        .map(preprocess_fn, num_parallel_calls=autotune)
        .shuffle(config.buffer_size)
        .batch(config.batch_size)
        .prefetch(autotune)
    )

    val_pipeline = (
        val_ds
        .map(preprocess_fn, num_parallel_calls=autotune)
        .batch(config.batch_size)
        .prefetch(autotune)
    )

    return train_pipeline, val_pipeline