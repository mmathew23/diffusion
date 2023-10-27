import hydra
from omegaconf import DictConfig
from data.utils import get_dataloader
from diffusers.optimization import get_cosine_schedule_with_warmup


@hydra.main(version_base=None, config_path="conf", config_name=None)
def train(config: DictConfig) -> None:

    train_dataloader = get_dataloader(config.data)
    unet = hydra.utils.instantiate(config.unet)
    noise_scheduler = hydra.utils.instantiate(config.scheduler)
    optimizer = hydra.utils.instantiate(config.optimizer, params=unet.parameters())
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.scheduler.num_warmup_steps, num_training_steps=len(train_dataloader)*config.num_epochs)



if __name__ == '__main__':
    train()
