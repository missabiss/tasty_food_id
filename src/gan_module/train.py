"""
Training for the GAN was performed in Google Colab due to dataset size
and compute requirements. This file is retained for reference.
"""

import tensorflow as tf
from dataclasses import dataclass
from typing import Dict

from src.gan.model import build_generator, build_discriminator



@dataclass
class GanConfig:
    img_size: int = 256
    lambda_l1: float = 100.0  # pix2pix typical weighting
    gen_lr: float = 2e-4
    disc_lr: float = 2e-4


class Pix2PixTrainer:
    """
    Trainer implementing pix2pix-style losses:
      - Generator loss = adversarial + lambda * L1(mask, pred_mask)
      - Discriminator loss = real/fake BCE
    """
    def __init__(self, config: GanConfig):
        self.cfg = config
        self.gen = build_generator(img_size=config.img_size)
        self.disc = build_discriminator(img_size=config.img_size)

        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.gen_opt = tf.keras.optimizers.Adam(config.gen_lr, beta_1=0.5)
        self.disc_opt = tf.keras.optimizers.Adam(config.disc_lr, beta_1=0.5)

    def generator_loss(self, disc_fake_logits, y_true_mask, y_pred_mask):
        adv = self.bce(tf.ones_like(disc_fake_logits), disc_fake_logits)
        l1 = tf.reduce_mean(tf.abs(y_true_mask - y_pred_mask))
        return adv + self.cfg.lambda_l1 * l1, adv, l1

    def discriminator_loss(self, disc_real_logits, disc_fake_logits):
        real_loss = self.bce(tf.ones_like(disc_real_logits), disc_real_logits)
        fake_loss = self.bce(tf.zeros_like(disc_fake_logits), disc_fake_logits)
        return real_loss + fake_loss

    @tf.function
    def train_step(self, x_img, y_mask) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            pred_mask = self.gen(x_img, training=True)

            disc_real = self.disc([x_img, y_mask], training=True)
            disc_fake = self.disc([x_img, pred_mask], training=True)

            gen_total, gen_adv, gen_l1 = self.generator_loss(disc_fake, y_mask, pred_mask)
            disc_total = self.discriminator_loss(disc_real, disc_fake)

        gen_grads = gen_tape.gradient(gen_total, self.gen.trainable_variables)
        disc_grads = disc_tape.gradient(disc_total, self.disc.trainable_variables)

        self.gen_opt.apply_gradients(zip(gen_grads, self.gen.trainable_variables))
        self.disc_opt.apply_gradients(zip(disc_grads, self.disc.trainable_variables))

        return {
            "gen_total": gen_total,
            "gen_adv": gen_adv,
            "gen_l1": gen_l1,
            "disc_total": disc_total,
        }

    def fit(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset = None, epochs: int = 10):
        """
        train_ds yields (image, mask) batches:
          image: (B, 256, 256, 3) float32 [0,1]
          mask:  (B, 256, 256, 1) float32 {0,1}
        """
        for epoch in range(1, epochs + 1):
            for step, (x_img, y_mask) in enumerate(train_ds):
                metrics = self.train_step(x_img, y_mask)

                if step % 100 == 0:
                    tf.print(
                        "epoch", epoch, "step", step,
                        "gen_total", metrics["gen_total"],
                        "disc_total", metrics["disc_total"],
                        "gen_l1", metrics["gen_l1"],
                    )

    def save_generator(self, path: str):
        self.gen.save(path)