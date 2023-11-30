from omegaconf import DictConfig, ListConfig
from hydra.utils import instantiate
from torchvision.transforms import Normalize, ToTensor, Compose, CenterCrop, RandomCrop, Resize
from torchvision.transforms.functional import get_dimensions, pad, crop
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, WeightedRandomSampler


class StridedRandomCrop(RandomCrop):
    @staticmethod
    def get_params(img, output_size, stride):
        _, h, w = get_dimensions(img)
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h//stride, size=(1,)).item() * stride
        j = torch.randint(0, w//stride, size=(1,)).item() * stride
        return i, j, th, tw

    def __init__(self, size, stride=128, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
        self.stride = stride

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size, self.stride)

        return crop(img, i, j, h, w)


def get_augmentations(cfg: DictConfig):
    augs = [instantiate(aug) for aug in cfg]
    return Compose(augs)


def get_transforms(cfg: ListConfig):
    augs = get_augmentations(cfg)
    transforms = Compose([
        # Resize(128),
        # CenterCrop(1024),
        augs,
        ToTensor(),
        Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    def apply_transforms(samples):
        samples['pixel_values'] = [transforms(sample) for sample in samples['image']]
        return samples

    return apply_transforms


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
    }


def get_dataloader(config: DictConfig):
    transforms = get_transforms(config.augmentations)
    dataset = load_dataset(**config.dataset)
    if 'filter' in config:
        if config.filter.by == 'index':
            dataset = dataset.select([config.filter.idx])
        else:
            #  assume filtering by set
            dataset = dataset.filter(lambda x: x['set'] == config.filter.idx)
    dataset.set_transform(transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        sampler=WeightedRandomSampler([1]*len(dataset), num_samples=config.dataloader.dataset_length),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return dataloader
