from matplotlib import pyplot as plt

def show_masks(noisy, pred, clean, n=5):
    """
    noisy, pred, clean: tensors or arrays of shape (B, H, W, 1)
    """

    noisy = noisy[:n].numpy() if hasattr(noisy, "numpy") else noisy[:n]
    pred  = pred[:n]
    clean = clean[:n].numpy() if hasattr(clean, "numpy") else clean[:n]

    # binarize AE output for visualization
    pred_bin = (pred > 0.5).astype("float32")

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))

    for i in range(n):
        # Noisy input
        axes[i, 0].imshow(noisy[i, ..., 0], cmap="gray")
        axes[i, 0].set_title("Noisy mask")
        axes[i, 0].axis("off")

        # Autoencoder output (binarized)
        axes[i, 1].imshow(pred_bin[i, ..., 0], cmap="gray")
        axes[i, 1].set_title("AE output")
        axes[i, 1].axis("off")

        # Clean target
        axes[i, 2].imshow(clean[i, ..., 0], cmap="gray")
        axes[i, 2].set_title("Clean mask")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()