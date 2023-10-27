from omegaconf import DictConfig, ListConfig
from hydra.utils import instantiate
from torchvision.transforms import Normalize, ToTensor, Compose
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, WeightedRandomSampler


def get_augmentations(cfg: DictConfig):
    augs = [instantiate(aug) for aug in cfg]
    return Compose(augs)


def get_transforms(cfg: ListConfig):
    augs = get_augmentations(cfg)
    transforms = Compose([
        augs,
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
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
    dataset.set_transform(transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        sampler=WeightedRandomSampler([1]*len(dataset), num_samples=config.dataloader.dataset_length),
        collate_fn=collate_fn
    )
    return dataloader
