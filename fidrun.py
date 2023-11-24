import sys
from eval_fid import *
from omegaconf import OmegaConf
import torch
from data.utils import get_dataloader
from alpha_deblend_pipeline import AlphaDeblendPipeline
import os
from datetime import datetime
import fire
import time



def main(
        checkpoint: str,
        num_samples: int = 100,
        device: str = 'cuda:0',
        config_path: str = "conf/alpha_pixel_diffusion.yaml",
):
    print(f'checkpoint {checkpoint}')
    print(f'num samples {num_samples}')
    print(f'device {device}')
    print(f'config path {config_path}')
    parent_path = os.path.dirname(checkpoint)
    # Get current time and format it as 'YYYYMMDDHHMMSS'
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Combine the parent path with the timestamp
    save_file = os.path.join(parent_path, f'fid_{num_samples}_{timestamp}')

    config = OmegaConf.load(config_path)
    train_dataloader = get_dataloader(config.data)
    pipeline = AlphaDeblendPipeline.from_pretrained(checkpoint)
    # if one wants to set `leave=False`
    pipeline.set_progress_bar_config(leave=False)

    # if one wants to disable `tqdm`
    pipeline.set_progress_bar_config(disable=True)
    start_time = time.time()
    fid=run_fid(pipeline=pipeline, dataloader=train_dataloader, num_samples=num_samples, num_inference_steps=50, batch_size=32, device=device)
    end_time = time.time()
    print(f'FID: {fid} calculated in {end_time-start_time} seconds')
    with open(save_file, 'w+') as f:
        f.write(str(fid))
        f.write('\n')

if __name__ == '__main__':
    fire.Fire(main)
