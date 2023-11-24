import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from data.utils import get_dataloader
from diffusers.optimization import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from validate import evaluate
from diffusers import DDIMPipeline, EMAModel
from diffusers.models.lora import LoRACompatibleConv
from accelerate import Accelerator
import os
from tqdm import tqdm
import xformers
import torch.nn.functional as F
from eval_fid import run_fid
from hydra.core.hydra_config import HydraConfig
import shutil


def compute_snr(timesteps, scheduler):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):

    if config.use_ema:
        ema = EMAModel(model.parameters())

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        # Create output directory if needed, and asserted for not None in train
        os.makedirs(config.output_dir, exist_ok=True)
        hydra_dir = os.path.join(HydraConfig.get().runtime.output_dir, '.hydra')
        print(f'copying from hydra dir {hydra_dir}')
        f_name = 'config.yaml'
        shutil.copy2(os.path.join(hydra_dir, f_name), os.path.join(config.output_dir, f_name))
        f_name = 'hydra.yaml'
        shutil.copy2(os.path.join(hydra_dir, f_name), os.path.join(config.output_dir, f_name))
        f_name = 'overrides.yaml'
        shutil.copy2(os.path.join(hydra_dir, f_name), os.path.join(config.output_dir, f_name))

        accelerator.init_trackers(config.model_name)

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if config.use_ema:
        ema.to(accelerator.device)

    global_step = 0

    for epoch in range(config.num_epochs):
        total_steps = len(train_dataloader) // config.gradient_accumulation_steps
        if len(train_dataloader) % config.gradient_accumulation_steps != 0:
            total_steps += 1
        progress_bar = tqdm(total=total_steps, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            images = batch["pixel_values"]

            noise = torch.randn_like(images) + config.noise_offset * torch.randn(
                        (images.shape[0], images.shape[1], 1, 1), device=images.device
                    )

            b = images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (b,), device=images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

                if config.snr_gamma > 0.0:
                    snr = compute_snr(timesteps, noise_scheduler)
                    mse_loss_weights = (
                        torch.stack([snr, config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                else:
                    loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if config.use_ema:
                    ema.step(model.parameters())
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

        if accelerator.is_main_process:
            pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(os.path.join(config.output_dir, f"checkpoints_{epoch+1}"))
                #fid = run_fid(pipeline, train_dataloader, device=accelerator.device)
                #accelerator.log({"fid": fid}, step=global_step)

                if config.use_ema:
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                    pipeline.save_pretrained(os.path.join(config.output_dir, f"ema_checkpoints_{epoch+1}"))
                    ema.restore(model.parameters())

    accelerator.end_training()


def convWithBilinear(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    self.paddingX = (self._reversed_padding_repeated_twice[0], self._reversed_padding_repeated_twice[1], self._reversed_padding_repeated_twice[2], self._reversed_padding_repeated_twice[3])
    # new_h = input.shape[2] + self.paddingX[2] + self.paddingX[3]
    # new_w = input.shape[3] + self.paddingX[0] + self.paddingX[1]
    # working = F.interpolate(input, size=(new_h, new_w), mode='bilinear', align_corners=True)
    working = F.pad(input, self.paddingX, mode='replicate')
    return F.conv2d(working, weight, bias, self.stride, (0, 0), self.dilation, self.groups)


def patch_conv2d(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            if isinstance(module, LoRACompatibleConv) and module.lora_layer is None:
                module.lora_layer = lambda *x: 0

            module._conv_forward = convWithBilinear.__get__(module, torch.nn.Conv2d)
        else:
            # Recursively apply to child modules
            patch_conv2d(module)


@hydra.main(version_base=None, config_path="conf", config_name='pixel_diffusion')
def train(config: DictConfig) -> None:

    train_dataloader = get_dataloader(config.data)
    # _convert_ partial to save listconfigs as lists in unet so that it can be saved
    unet = hydra.utils.instantiate(config.unet, _convert_="partial")
    patch_conv2d(unet)
    print(f'Parameter count: {sum([torch.numel(p) for p in unet.parameters()])}')
    noise_scheduler = hydra.utils.instantiate(config.noise_scheduler)
    optimizer = hydra.utils.instantiate(config.optimizer, params=unet.parameters())
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.lr_scheduler.num_warmup_steps, num_training_steps=len(train_dataloader)*config.num_epochs//config.gradient_accumulation_steps)
    assert config.output_dir is not None, "You need to specify an output directory"

    train_loop(config, unet, noise_scheduler, optimizer, train_dataloader, lr_scheduler)



if __name__ == '__main__':
    train()
