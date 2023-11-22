import torch
import numpy as np
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from torchmetrics.image.fid import FrechetInceptionDistance


def create_diffusion_samples(pipeline, num_samples, fid, batch_size=16, device=None, real=False, num_inference_steps=50):
    if device is not None:
        pipeline.to(device)
    count = 0
    for i in range(int(np.ceil(num_samples / batch_size))):
        images = pipeline(batch_size=batch_size, num_inference_steps=num_inference_steps, return_dict=True, output_type='tensor').images
        if isinstance(images, np.ndarray):
            # images = torch.tensor(images.permute(0, 3, 1, 2))
            images = torch.tensor(np.transpose(images, (0, 3, 1, 2)))
        fid.update(images, real=real)
        count += images.shape[0]
        if count >= num_samples:
            break


def create_tile_samples(dataloader, num_samples, fid, real=True):
    count = 0
    while count < num_samples:
        for i, batch in enumerate(dataloader):
            if i == 0 and count == 0:
                fid = fid.to(device=batch['pixel_values'].device)
            fid.update(((batch['pixel_values']+1)/2), real=real)
            count += batch['pixel_values'].shape[0]
            if count >= num_samples:
                break


def calculate_fid(images1, images2, batch_size=64, device=None):
    # Process each set of images
    fid = FrechetInceptionDistance(normalize=True)
    def process_images(images, real=True):
        n_batches = int(np.ceil(len(images) / batch_size))
        for i in range(n_batches):
            batch = images[i * batch_size:(i + 1) * batch_size]
            batch = torch.cat(batch, dim=0).cpu()
            fid.update(batch, real=real)

    process_images(images1, real=False)
    process_images(images2, real=True)

    return fid.compute()


def run_fid(pipeline, dataloader, num_samples=10000, batch_size=32, device="cuda:0", num_inference_steps=50):
    fid = FrechetInceptionDistance(normalize=True)
    # fid.to(device=device)
    pipeline.to(torch.device(device))
    create_diffusion_samples(pipeline, num_samples, fid, batch_size=batch_size, device=device, num_inference_steps=num_inference_steps)
    create_tile_samples(dataloader, num_samples, fid, real=True)
    # create_tile_samples(dataloader, num_samples, fid, real=False)
    # fid = calculate_fid(diff_samples, tile_samples, batch_size=batch_size, device=device)
    # fid.update(torch.cat(tile_samples, dim=0), real=True)
    # fid.update(torch.cat(diff_samples, dim=0), real=False)
    # fid = fid.compute()

    return fid.compute()
