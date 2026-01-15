import tensorflow as tf

def jaccard_index(y_true, y_pred, threshold: float=0.5):
    """
    Computes the Jaccard Index, also known as Intersection over Union (IoU).
    y_true: ground truth masks [B, H, W, 1]
    y_pred: predicted masks [B, H, W, 1]
    returns: Jaccard Index score
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred_binary)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_binary) - intersection

    return intersection / (union + 1e-7)

class JaccardIndexMetric(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='jaccard_index', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        jaccard_idx = jaccard_index(y_true, y_pred, self.threshold)
        self.total.assign_add(jaccard_idx)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / (self.count + 0.000001)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)