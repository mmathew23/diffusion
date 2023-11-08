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
    dataset = dataset.filter(lambda x: x['set'] == 1 and x['image'] == dataset[0]['image'])
    dataset.set_transform(transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        sampler=WeightedRandomSampler([1]*len(dataset), num_samples=config.dataloader.dataset_length),
        collate_fn=collate_fn
    )
    return dataloader
