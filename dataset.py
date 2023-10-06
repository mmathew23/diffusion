import torch


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'image': torch.tensor([x['image'] for x in batch])
    }