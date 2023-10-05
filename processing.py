from transformers import FeatureExtractionMixin, ImageFeatureExtractionMixin
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    make_list_of_images,
    valid_images,
    to_numpy_array,
    infer_channel_dimension_format,

)
from transformers.image_transforms import to_channel_dimension_format
from transformers import BatchFeature
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import functional as F
import random


class RandomRotate90:
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        angle = random.choice(self.angles)
        return F.rotate(img, angle)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class DiffusionFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    def __init__(self, size=256, **kwargs):
        super().__init__(**kwargs)

        self.size = size
        #  The standard mean and STD of just 0.5's
        #  DEFAULT are the actual imagenet values
        self.image_mean = IMAGENET_STANDARD_MEAN
        self.image_std = IMAGENET_STANDARD_STD

    def __call__(self, images: ImageInput, **kwargs) -> BatchFeature:
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        images = [to_numpy_array(image) for image in images]
        input_data_format = infer_channel_dimension_format(images[0])
        image_mean, image_std = self.image_mean, self.image_std
        #  Normalize before cropping?
        images = [
            self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
            for image in images
        ]
        data_format = 'channels_first'
        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type='pt')

    def get_transforms(self):
        transforms = Compose([
            RandomCrop(self.size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotate90()
        ])

        return transforms
