import os

from dotenv import load_dotenv

import wandb
from pokemon_ddpm import _PATH_TO_DATA
from pokemon_ddpm.data import PokemonDataset
from pokemon_ddpm.model import get_models


def setup_wandb_sweep(train_fn: callable, sweep_file_path: str, model: any) -> None:
    """
    Setup wandb for logging and configure sweeps for hyperparameter tuning.

    Args:
        train_fn (callable): The training function that accepts a config dictionary.
        sweep_file_path (str): Path to the sweep.yaml file.
        model (any): The model to be used for training.
    """
    # Load the sweep configuration from the YAML file
    import yaml
    with open(sweep_file_path, "r") as file:
        sweep_config = yaml.safe_load(file)

    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key is None:
        raise ValueError("WANDB_API_KEY not found in the environment. Make sure it is set in your .env file.")

    wandb.login(key=wandb_api_key)

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config.get("project", "default_project"))

    # Function to execute training with WandB sweep
    def sweep_train_fn():
        wandb.init()
        config = wandb.config

        # Call the training function with the sweep configuration
        train_fn(
            model=model,
            lr=config["lr"],
            lr_warmup_steps=config["lr_warmup_steps"],
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            save_model=config.get("save_model", False),
            train_set=PokemonDataset(_PATH_TO_DATA),
            wandb_active=True,
        )

        wandb.finish()

    # Start the sweep
    wandb.agent(sweep_id, function=sweep_train_fn)


def log_training(epoch: int, epoch_loss: float, wandb_active: bool = True, model=None, lr_scheduler=None, train_dataloader=None) -> None:
    """Enhanced logging for training metrics.

    Args:
        epoch (int): The current epoch number.
        epoch_loss (float): The loss value for the current epoch.
        wandb_active (bool): Whether wandb is active.
        model (nn.Module): The model being trained.
        lr_scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        train_dataloader (torch.utils.data.DataLoader): The training dataloader.
    """
    
    if wandb_active:
        logs = {
            "train/loss": epoch_loss,
            "train/avg_batch_loss": epoch_loss / len(train_dataloader),
            "train/lr": lr_scheduler.get_last_lr()[0] if lr_scheduler else None,
        }

        
        if model is not None:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            logs["train/gradient_norm"] = total_norm ** 0.5

        wandb.log(logs)