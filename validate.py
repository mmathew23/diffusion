from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
import torch


def evaluate(config, epoch, pipeline: DDPMPipeline):
    images = pipeline(
        batch_size=config.val_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    image_grid = make_image_grid(images, rows=4, cols=len(images)//4 + (1 if len(images) % 4 != 0 else 0))

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")