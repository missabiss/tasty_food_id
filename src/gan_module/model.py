import tensorflow as tf
from tensorflow.keras import layers, Model

def load_gan_segmenter(path="models/segmenter.keras"):
    pass

def generate_mask(model, images):
    """
    images: [B, H, W, 3] float32 [0,1]
    returns: rough masks [B, H, W, 1] in [0,1]
    """
    pass

def build_generator(img_size: int = 256) -> Model:
    """
    U-Net style generator: RGB image -> 1-channel mask (sigmoid in [0,1])

    Output shape: (img_size, img_size, 1)
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))

    # --- Encoder blocks ---
    def down_block(x, filters):
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        return x

    # --- Decoder blocks ---
    def up_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Concatenate()([x, skip])
        return x

    d1 = down_block(inputs, 64)     # 128x128
    d2 = down_block(d1, 128)        # 64x64
    d3 = down_block(d2, 256)        # 32x32
    d4 = down_block(d3, 512)        # 16x16
    d5 = down_block(d4, 512)        # 8x8

    # bottleneck
    b = layers.Conv2D(512, 4, strides=2, padding="same")(d5)   # 4x4
    b = layers.ReLU()(b)

    u1 = up_block(b, d5, 512)       # 8x8
    u2 = up_block(u1, d4, 512)      # 16x16
    u3 = up_block(u2, d3, 256)      # 32x32
    u4 = up_block(u3, d2, 128)      # 64x64
    u5 = up_block(u4, d1, 64)       # 128x128

    # final upsample to 256x256
    out = layers.Conv2DTranspose(1, 4, strides=2, padding="same")(u5)
    out = layers.Activation("sigmoid")(out)

    return Model(inputs, out, name="gan_generator")


def build_discriminator(img_size: int = 256) -> Model:
    """
    PatchGAN discriminator: looks at (image, mask) and predicts real/fake patches.

    Inputs:
      - image: (img_size, img_size, 3)
      - mask:  (img_size, img_size, 1)
    Output:
      - patch logits map (smaller spatial grid)
    """
    img_in = layers.Input(shape=(img_size, img_size, 3))
    mask_in = layers.Input(shape=(img_size, img_size, 1))
    x = layers.Concatenate()([img_in, mask_in])  # channels: 4

    def disc_block(x, filters, stride=2):
        x = layers.Conv2D(filters, 4, strides=stride, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
        return x

    x = disc_block(x, 64)
    x = disc_block(x, 128)
    x = layers.BatchNormalization()(x)
    x = disc_block(x, 256)
    x = layers.BatchNormalization()(x)

    # final conv -> patch logits
    x = layers.Conv2D(1, 4, strides=1, padding="same")(x)

    return Model([img_in, mask_in], x, name="gan_discriminator")