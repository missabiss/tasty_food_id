import tensorflow as tf
from tensorflow.keras import callbacks
from typing import Optional

from .metrics import JaccardIndexMetric
from .config import AEConfig
from .model import build_autoencoder
from .data import build_food101_pipeline, GanMaskFn

def compile_ae_model(model: tf.keras.Model, config: AEConfig) -> tf.keras.Model:
    """
    Compiles the autoencoder model with optimizer, loss function, and metrics.
    model: autoencoder model to compile
    config: AEConfig object containing configuration parameters
    returns: compiled autoencoder model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[JaccardIndexMetric(threshold=config.jaccard_threshold), "accuracy"]
    )
    return model

def train_mask_ae_model(
    config: AEConfig,
    gan_mask_function: Optional[GanMaskFn] = None,
    add_noise: bool = True,
    train_ds: Optional[tf.data.Dataset] = None,
    val_ds: Optional[tf.data.Dataset] = None
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    """
    # ensure directories for checkpoints and logs exist
    checkpoint_path, log_dir = config.check_directories()

    # create pipelines
    train_pipeline, val_pipeline = build_food101_pipeline(config=config,
                                                          gan_mask_function=gan_mask_function,
                                                          add_noise=add_noise,
                                                          train_ds=train_ds,
                                                          val_ds=val_ds)
    
    # build and compile the autoencoder model
    ae_model = build_autoencoder(config=config)
    compiled_model = compile_ae_model(ae_model, config=config)


    checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config.early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )

    tensorboard = callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False
    )

    history = compiled_model.fit(
        train_pipeline,
        validation_data=val_pipeline,
        epochs=config.max_epochs,
        callbacks=[checkpoint, early_stop, tensorboard]
    )

    return compiled_model, history

def load_trained_ae_model(
    model_path: str,
    threshold: float = 0.5
) -> tf.keras.Model:
    """
    Loads a trained autoencoder model from the specified path.
    model_path: path to the saved model
    threshold: threshold for the Jaccard index metric
    returns: loaded autoencoder model
    """
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"JaccardIndexMetric": JaccardIndexMetric(threshold=threshold)}
    )