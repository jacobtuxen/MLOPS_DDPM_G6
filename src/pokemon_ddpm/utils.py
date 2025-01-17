import os

from dotenv import load_dotenv

import wandb
from pokemon_ddpm import _PATH_TO_DATA
from pokemon_ddpm.data import PokemonDataset
from pokemon_ddpm.model import get_models


def setup_wandb_sweep(train_loop):
    """Setup wandb for logging."""
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key is None:
        raise ValueError("WANDB_API_KEY not found in the environment. Make sure it is set in your .env file.")

    wandb.login(key=wandb_api_key)
    config = {
        "lr": 1e-4,
        "lr_warmup_steps": 1000,
        "batch_size": 32,
        "epochs": 10,
        "save_model": False,
    }
    sweep_id = wandb.sweep(sweep=config, project="pokemon-ddpm")
    wandb.agent(sweep_id, function=sweep_train)


def sweep_train(config=None):
    """Function to be executed during a wandb sweep."""
    with wandb.init(config=config):
        config = wandb.config
        _, unet = get_models(model_name=None, device=config.device)

        wandb.init(project="pokemon-ddpm", config=config)

        # train(
        #     model=unet,
        #     lr=config.lr,
        #     lr_warmup_steps=config.lr_warmup_steps,
        #     batch_size=config.batch_size,
        #     epochs=config.epochs,
        #     save_model=config.save_model,
        #     train_set=PokemonDataset(_PATH_TO_DATA),
        #     wandb_active=True,
        # )
        wandb.finish()


def log_training(epoch, epoch_loss, wandb_active):
    """Log the training loss to wandb."""
    if wandb_active:
        wandb.log({"train/epoch": epoch, "train/loss": epoch_loss})
