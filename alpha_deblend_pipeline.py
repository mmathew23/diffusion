from diffusers import ImagePipelineOutput, DiffusionPipeline, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from typing import Optional, List, Union, Tuple
import torch
import math


class AlphaDeblendPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, method='euler'):
        super().__init__()

        # we ignore this, just having a scheduler for HF compatibility
        scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)
        self.trained_image_size = unet.config.sample_size
        self.method = method

    def step(self, x, t, num_inference_steps=50):
        if self.method == 'euler':
            return self.step_euler(x, t, num_inference_steps=num_inference_steps)
        elif self.method == 'rk':
            return self.step_rk(x, t, num_inference_steps=num_inference_steps)
        else:
            raise NotImplementedError()

    def step_rk(self, x, t, num_inference_steps=50):
        alpha = 1 - math.cos(t / num_inference_steps*math.pi/2)
        alpha_half = 1 - math.cos((t+0.5) / num_inference_steps*math.pi/2)
        d_alpha = alpha_half - alpha
        model_output = self.unet(x, torch.tensor(alpha, device=x.device, dtype=torch.float16)).sample
        x_half = x + d_alpha * model_output
        alpha_1 = 1 - math.cos((t+1) / num_inference_steps*math.pi/2)
        d_alpha = alpha_1 - alpha
        model_output = self.unet(x_half, torch.tensor(alpha_half, device=x.device, dtype=torch.float16)).sample
        return x + d_alpha * model_output

    def step_euler(self, x, t, num_inference_steps=50):
        if t == 0:
            t = 1e-4
        alpha = t / num_inference_steps
        d_alpha = (t+1)/num_inference_steps - alpha
        model_output = self.unet(x, torch.tensor(alpha, device=x.device, dtype=torch.float16)).sample
        return x + d_alpha * model_output

    @torch.no_grad()
    def tile_generate(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        buffer: int = 8,
    ) -> Union[ImagePipelineOutput, Tuple]:
        mult, rem = divmod(self.unet.config.sample_size - self.trained_image_size, self.trained_image_size-buffer)
        mult += 1 if rem>0 else 0
        output_shape = (self.trained_image_size-buffer) * mult + self.trained_image_size
        image_shape = (batch_size, self.unet.config.in_channels, output_shape, output_shape)
        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)
        # image += 0.1 * torch.randn(
        #                 (image.shape[0], image.shape[1], 1, 1), device=image.device)


        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(range(num_inference_steps)):
            tiles = torch.nn.functional.unfold(
                image,
                kernel_size=(self.trained_image_size, self.trained_image_size),
                stride=(self.trained_image_size-buffer, self.trained_image_size-buffer),
            )
            #current shape b, c*k*k, n where n is the number of tiles
            bs, ckk, num_tiles = tiles.shape
            tiles = tiles.view(bs, 3, self.trained_image_size, self.trained_image_size, num_tiles).permute(4, 0, 1, 2, 3)
            for tile_num in range(tiles.shape[0]):
                tiles[tile_num] = self.step(tiles[tile_num], t, num_inference_steps)
            tiles = tiles.permute(1, 2, 3, 4, 0).view(bs, ckk, num_tiles)
            image = torch.nn.functional.fold(
                tiles,
                output_size=(output_shape, output_shape),
                kernel_size=(self.trained_image_size, self.trained_image_size),
                stride=(self.trained_image_size-buffer, self.trained_image_size-buffer),
            )
            # fold up ones to know how to weight the overlapped patches for mean
            ones = torch.ones_like(tiles)
            mean_weight = torch.nn.functional.fold(
                ones,
                output_size=(output_shape, output_shape),
                kernel_size=(self.trained_image_size, self.trained_image_size),
                stride=(self.trained_image_size-buffer, self.trained_image_size-buffer),
            )
            image /= mean_weight

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu()
        if output_type == "pil":
            image = self.numpy_to_pil(image.permute(0, 2, 3, 1).numpy())
        if output_type == "numpy":
            image = self.numpy_to_tensor(image.permute(0, 2, 3, 1).numpy())

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)
        # image += 0.1 * torch.randn(
        #                 (image.shape[0], image.shape[1], 1, 1), device=image.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(range(num_inference_steps)):
            image = self.step(image, t, num_inference_steps)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu()
        if output_type == "pil":
            image = self.numpy_to_pil(image.permute(0, 2, 3, 1).numpy())
        if output_type == "numpy":
            image = self.numpy_to_tensor(image.permute(0, 2, 3, 1).numpy())

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
