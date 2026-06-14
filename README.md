# MNIST Denoising Autoencoder

**Week-6 Assessment — Deep Learning (Autoencoders & GANs)**

A convolutional **denoising autoencoder** built in PyTorch that removes noise
from handwritten-digit images. The network is trained on the MNIST dataset: it
receives a noise-corrupted image as input and learns to reconstruct the clean
original.

## How it works

A denoising autoencoder has two halves:

- **Encoder** — compresses a `28×28` image down to a small latent representation
  using strided convolutions (`28→14→7`, channels `1→32→64`).
- **Decoder** — reconstructs the image back to `28×28` using transposed
  convolutions (`7→14→28`, channels `64→32→1`).

By feeding the network **noisy** images but scoring it against the **clean**
originals (MSE loss), the model is forced to learn the underlying structure of
each digit and discard the random noise. Fresh Gaussian noise is sampled on every
batch so the model never sees the same corruption twice.

```
clean digit ──add noise──▶ noisy digit ──▶ [ Encoder → latent → Decoder ] ──▶ denoised digit
                                                                                ▲
                                                          loss compared to the clean digit
```

## Dataset

MNIST as PNG images (60,000 train / 10,000 test), from Kaggle:
<https://www.kaggle.com/datasets/awsaf49/mnist-dataset>

Download and unzip it into a `data/` folder so the layout is:

```
data/mnist_png/training/<0-9>/*.png
data/mnist_png/testing/<0-9>/*.png
```

(The `data/` folder is git-ignored — it is not committed to the repo.)

## Setup & run

```bash
pip install -r requirements.txt
jupyter notebook denoising_autoencoder.ipynb
```

Run all cells. The notebook automatically uses an Apple-Silicon (MPS) or CUDA GPU
if available, otherwise the CPU. Training is 10 epochs.

## Results

The trained model cleans up unseen noisy test images, recovering clear,
recognisable digits. After 10 epochs the test reconstruction error is
**MSE ≈ 0.012**.

**Denoising on unseen test images** (noisy input → denoised output → clean target):

![Denoising results](assets/denoising_results.png)

**Training loss:**

![Training loss](assets/loss_curve.png)

The trained weights are saved to `models/denoising_autoencoder.pth`.

## Files

| File | Purpose |
|------|---------|
| `denoising_autoencoder.ipynb` | Main notebook — full pipeline with outputs |
| `build_notebook.py` | Script that generates the notebook |
| `requirements.txt` | Python dependencies |
| `models/` | Saved model weights (created after training) |
