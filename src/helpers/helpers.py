# this file is for helpers to load data consistently across modules.  
# e.g., image loaders, text loaders, dataset loaders etc.

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np


IMAGE_SIZE = (256, 256)

# Load Food101 dataset
def load_food101():
    train, train_val, test = tfds.load(
        "food101",
        split=["train[:90%]", "train[90%:]", "validation"],
        as_supervised=False
    )

    return train, train_val, test

# Preprocess images: resize and normalize
def preprocess_image(image, label=None):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    if label is not None:
        return image, label
    return image

# Load a single image from file path
def load_single_image(image_path):
    image = keras_image.load_img(image_path)
    arr = keras_image.img_to_array(image)
    arr = arr /255.0
    return np.expand_dims(arr, axis=0)
