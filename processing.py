from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Normalize
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


class DiffusionImageProcessor:
    def __init__(self, size=256):
        self.size = size
        #  The standard mean and STD of just 0.5's
        #  DEFAULT are the actual imagenet values
        self.image_mean = IMAGENET_STANDARD_MEAN
        self.image_std = IMAGENET_STANDARD_STD
        self.transforms = Compose([
            ToTensor(),
            Normalize(mean=self.image_mean, std=self.image_std),
            RandomCrop(self.size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotate90(),
        ])

    def get_transforms(self):
        def transform(example):
            example['pixel_values'] = [self.transforms(img) for img in example['image']]
            return example
        return transform
