import tensorflow as tf
from tensorflow.keras import layers, Model
from .config import AEConfig
from typing import Optional, Tuple

def build_autoencoder(
    config: AEConfig,
    input_shape: Optional[Tuple[int, int, int]] = None
) -> tf.keras.Model:
    """
    Builds a convolutional autoencoder model for mask refinement.
    config: AEConfig object containing configuration parameters
    input_shape: shape of the input masks (H, W, C)
    returns: compiled autoencoder model    
    """
    if input_shape is None:
        input_shape = (config.img_size, config.img_size, 1)

    inputs = layers.Input(shape=input_shape, name="mask_in")

    # Encoder
    x = layers.Conv2D(32, 3, activation="relu", padding="same", name="enc_conv1")(inputs)
    x = layers.MaxPooling2D(2, padding="same", name="enc_pool1")(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same", name="enc_conv2")(x)
    x = layers.MaxPooling2D(2, padding="same", name="enc_pool2")(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same", name="enc_conv3")(x)
    x = layers.MaxPooling2D(2, padding="same", name="enc_pool3")(x)

    bottleneck = layers.Conv2D(256, 3, activation="relu", padding="same", name="bottleneck")(x)

    # Decoder
    x = layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same", name="dec_deconv1")(bottleneck)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same", name="dec_deconv2")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same", name="dec_deconv3")(x)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="mask_out")(x)
    return Model(inputs, outputs, name="mask_autoencoder")

