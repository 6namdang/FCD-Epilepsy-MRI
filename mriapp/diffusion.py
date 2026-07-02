"""
Toy 2D diffusion demo -- trained from scratch, fast, on brain slices you've
already downloaded from ds004199.

IMPORTANT HONESTY NOTE: this is NOT a true latent diffusion model. A real
LDM (e.g. the MONAI/Pinaya et al. UK Biobank model) trains a separate VAE
to compress into latent space first, then diffuses in that latent space --
that's what makes it fast enough for 3D volumes and anatomically faithful.
Training a VAE here would add real time, which defeats the "fast" goal.
This is a small pixel-space DDPM (Denoising Diffusion Probabilistic Model)
trained directly on downsampled 2D slices. It demonstrates the diffusion
*mechanism* -- iterative denoising from Gaussian noise -- not a
scientifically valid brain generator. With a handful of subjects and a few
hundred training steps, expect blurry, brain-ish blobs, not real anatomy.
"""
import numpy as np
import nibabel as nib
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler

torch.set_num_threads(4)


def extract_slices(t1_paths: list, img_size: int = 32, slices_per_subject: int = 15,
                    axis: int = 2) -> torch.Tensor:
    """
    Pull a band of axial slices from the middle third of each T1 volume
    (avoids mostly-empty slices near the top/bottom of the head), resize
    to img_size x img_size, normalize to [-1, 1].
    Returns a tensor of shape (N, 1, img_size, img_size), or None if
    nothing usable was extracted.
    """
    all_slices = []
    for path in t1_paths:
        vol = nib.load(path).get_fdata()
        n = vol.shape[axis]
        start, end = int(n * 0.35), int(n * 0.65)
        idxs = np.linspace(start, end, slices_per_subject).astype(int)

        for idx in idxs:
            if axis == 2:
                sl = vol[:, :, idx]
            elif axis == 1:
                sl = vol[:, idx, :]
            else:
                sl = vol[idx, :, :]

            sl = np.nan_to_num(sl)
            if sl.max() <= 0:
                continue
            lo, hi = np.percentile(sl, [1, 99])
            if hi <= lo:
                continue
            sl = np.clip(sl, lo, hi)
            sl = (sl - lo) / (hi - lo)  # -> [0, 1]

            img = Image.fromarray((sl * 255).astype(np.uint8)).resize((img_size, img_size))
            arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # -> [-1, 1]
            all_slices.append(arr)

    if not all_slices:
        return None
    return torch.tensor(np.stack(all_slices))[:, None, :, :]  # (N, 1, H, W)


def build_tiny_unet(img_size: int = 32) -> UNet2DModel:
    """Small enough to train in ~1-2 minutes on CPU."""
    return UNet2DModel(
        sample_size=img_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(32, 64),
        norm_num_groups=16,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )


def train_toy_ddpm(dataset: torch.Tensor, img_size: int = 32, num_train_steps: int = 200,
                    batch_size: int = 8, lr: float = 1e-4, progress_callback=None):
    """
    Trains the tiny UNet to predict added noise (standard DDPM objective).
    progress_callback(step, total_steps, loss) is called periodically if given.
    """
    model = build_tiny_unet(img_size)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    n = dataset.shape[0]
    model.train()
    for step in range(num_train_steps):
        idx = torch.randint(0, n, (min(batch_size, n),))
        clean = dataset[idx]
        noise = torch.randn_like(clean)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (clean.shape[0],)).long()
        noisy = scheduler.add_noise(clean, noise, timesteps)

        pred = model(noisy, timesteps).sample
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if progress_callback and (step % 10 == 0 or step == num_train_steps - 1):
            progress_callback(step + 1, num_train_steps, loss.item())

    model.eval()
    return model


def sample_images(model: UNet2DModel, num_samples: int = 4, img_size: int = 32,
                   num_inference_steps: int = 30) -> np.ndarray:
    """
    DDIM sampling (faster than full DDPM ancestral sampling for the same
    quality) starting from pure Gaussian noise.
    Returns uint8 array of shape (num_samples, img_size, img_size).
    """
    ddim = DDIMScheduler(num_train_timesteps=1000)
    ddim.set_timesteps(num_inference_steps)

    sample = torch.randn(num_samples, 1, img_size, img_size)
    model.eval()
    with torch.no_grad():
        for t in ddim.timesteps:
            noise_pred = model(sample, t).sample
            sample = ddim.step(noise_pred, t, sample).prev_sample

    imgs = sample.clamp(-1, 1).numpy()
    imgs = ((imgs + 1) / 2 * 255).astype(np.uint8)
    return imgs[:, 0, :, :]