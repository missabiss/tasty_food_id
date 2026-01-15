from pathlib import Path
from datetime import datetime

class AEConfig:
    """
    Configuration class for the autoencoder model and training parameters.
    """
    def __init__(self,
                 img_size: int = 128,
                 batch_size: int = 32,
                 shuffle_buffer_size: int = 1000,
                 max_epochs: int = 30,
                 learning_rate: float = 1e-3,
                 early_stopping_patience: int = 5,
                 jaccard_threshold: float = 0.5,
                 model_checkpoint_path: str = "models/autoencoder/mask_ae.keras",
                 logs_path_root: str = "logs/autoencoder/"):
        self.img_size = img_size
        self.batch_size = batch_size
        self.buffer_size = shuffle_buffer_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.jaccard_threshold = jaccard_threshold
        self.model_checkpoint_path = model_checkpoint_path
        self.logs_path_root = logs_path_root

    def check_directories(self):
        ch_path = Path(self.model_checkpoint_path)
        ch_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = Path(self.logs_path_root) / datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path.mkdir(parents=True, exist_ok=True)
        return str(ch_path), str(log_path)