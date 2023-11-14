from diffusers import ImagePipelineOutput, DiffusionPipeline, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from typing import Optional, List, Union, Tuple
import torch


class AlphaDeblendPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()

        # we ignore this, just having a scheduler for HF compatibility
        scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)

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
            alpha_start = (t / num_inference_steps)
            alpha_end = ((t+1) / num_inference_steps)

            # 1. predict noise model_output
            model_output = self.unet(image, torch.tensor(alpha_start, device=image.device, dtype=torch.float16)).sample
            image = image + (alpha_end-alpha_start) * model_output

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu()
        if output_type == "pil":
            image = self.numpy_to_pil(image.permute(0, 2, 3, 1).numpy())
        if output_type == "numpy":
            image = self.numpy_to_tensor(image.permute(0, 2, 3, 1).numpy())

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
