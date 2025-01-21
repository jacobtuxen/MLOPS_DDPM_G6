import hydra
import torch
import torch.utils
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from pokemon_ddpm import _PATH_TO_CONFIG, _PATH_TO_DATA, _PATH_TO_MODELS, _PATH_TO_SWEEP
from pokemon_ddpm.data import PokemonDataset
from pokemon_ddpm.model import PokemonDDPM
from pokemon_ddpm.utils import setup_wandb_sweep


def train(
    model: LightningModule,
    train_set: torch.utils.data.Dataset,
    epochs: int = 10,
    batch_size: int = 32,
    wandb_active=False,
) -> None:
    train_dataloader = DataLoader(train_set, batch_size=batch_size)
    trainer = Trainer(max_epochs=epochs)
    trainer.fit(model, train_dataloader)
    model.save_model(_PATH_TO_MODELS)


@hydra.main(config_path=str(_PATH_TO_CONFIG), config_name="train_config.yaml")
def main(cfg):
    model = PokemonDDPM()

    if cfg.use_wandb:
        setup_wandb_sweep(
            train_fn=train,
            sweep_file_path=_PATH_TO_SWEEP,
            model=model,
        )
    else:
        train(
            model=model,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            train_set=PokemonDataset(_PATH_TO_DATA),
        )


if __name__ == "__main__":
    main()
