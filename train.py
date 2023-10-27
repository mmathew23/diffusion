import torch
import hydra
from omegaconf import DictConfig
from data.utils import get_dataloader
from diffusers.optimization import get_cosine_schedule_with_warmup
from validate import evaluate
from diffusers import DDPMPipeline
from accelerate import Accelerator
import os
from tqdm import tqdm
import xformers
import torch.nn.functional as F


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        # Create output directory if needed, and asserted for not None in train
        os.makedirs(config.output_dir, exist_ok=True)

        accelerator.init_trackers(config.model_name)

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
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
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)

    accelerator.end_training()


@hydra.main(version_base=None, config_path="conf", config_name='pixel_diffusion')
def train(config: DictConfig) -> None:

    train_dataloader = get_dataloader(config.data)
    unet = hydra.utils.instantiate(config.unet)
    unet.enable_xformers_memory_efficient_attention()
    noise_scheduler = hydra.utils.instantiate(config.noise_scheduler)
    optimizer = hydra.utils.instantiate(config.optimizer, params=unet.parameters())
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.lr_scheduler.num_warmup_steps, num_training_steps=len(train_dataloader)*config.num_epochs)
    assert config.output_dir is not None, "You need to specify an output directory"

    train_loop(config, unet, noise_scheduler, optimizer, train_dataloader, lr_scheduler)



if __name__ == '__main__':
    train()
