import tensorflow as tf

def build_cnn_model():
    pass

def load_cnn_model(path="models/best_cnn.keras"):
    return tf.keras.models.load_model(path)

def predict_dish(model, images):
    """
    images: batch of preprocessed food101 images
    returns: class indices or probabilities
    """
    return model.predict(images)