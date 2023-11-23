import sys
import torch
import torch.nn as nn
import os
from alpha_deblend_pipeline import AlphaDeblendPipeline
from diffusers import DDIMPipeline
from diffusers.models.lora import LoRACompatibleConv
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from torch.nn.modules.utils import _pair


def asymmetricConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    self.paddingX = (self._reversed_padding_repeated_twice[0], self._reversed_padding_repeated_twice[1], self._reversed_padding_repeated_twice[2], self._reversed_padding_repeated_twice[3])
    working = F.pad(input, self.paddingX, mode='circular')
    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)

# Set the environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
def patch_conv2d_to_circular(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            if isinstance(module, LoRACompatibleConv) and module.lora_layer is None:
                module.lora_layer = lambda *x: 0

            module._conv_forward = asymmetricConv2DConvForward.__get__(module, torch.nn.Conv2d)
        else:
            # Recursively apply to child modules
            patch_conv2d_to_circular(module)


def inference(checkpoint):
    pipeline = AlphaDeblendPipeline.from_pretrained(checkpoint)
    # pipeline = DDIMPipeline.from_pretrained(checkpoint)
    pipeline.to(torch_dtype=torch.float16)
    # patch_conv2d_to_circular(pipeline.unet)
    # Set the maximum split size for the caching allocator in megabytes
    # freeU needs to be tuned for each model
    # register_free_upblock2d(pipeline)

    pipeline.unet.config.sample_size = 1024
    pipeline.to(torch.device("cuda:0"))
    images = pipeline(
        batch_size=1,
        num_inference_steps=50,
        # generator=torch.cuda.manual_seed(0),
    ).images
    # images = pipeline.tile_generate(
    #     batch_size=1,
    #     num_inference_steps=50,
    #     # generator=torch.cuda.manual_seed(0),
    # ).images


    directory = os.path.dirname(checkpoint)
    test = os.path.join(directory, "test_samples")
    os.makedirs(test, exist_ok=True)
    for i, img in enumerate(images):
        img.save(f"{test}/image_{i}.png")


if __name__ == '__main__':
    inference(sys.argv[1])
