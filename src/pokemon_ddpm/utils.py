import os

import torch
import yaml
from dotenv import load_dotenv
from pytorch_lightning.callbacks import Callback
from torch.nn.utils import prune

import wandb
from pokemon_ddpm import _PATH_TO_OUTPUT


def setup_wandb_sweep(train_fn: callable, sweep_file_path: str, model: any, train_set: any, epochs: int, device: str):
    """Setup wandb for logging and configure sweeps for hyperparameter tuning.

    Args:
        train_fn (callable): The training function that accepts a config dictionary.
        sweep_file_path (str): Path to the sweep.yaml file.
        model (any): The model to be used for training.
        train_set (any): The training dataset.
        epochs (int): The number of epochs to train the model for.
        device (str): The device to train the model on.

    """

    with open(sweep_file_path, "r") as file:
        sweep_config = yaml.safe_load(file)

    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key is None:
        raise ValueError("WANDB_API_KEY not found in the environment. Make sure it is set in your .env file.")

    wandb.login(key=wandb_api_key)

    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config.get("project", "pokemon-ddpm"))

    def sweep_train_fn():
        wandb.init()
        config = wandb.config

        train_fn(
            model=model,
            batch_size=config["batch_size"],
            epochs=epochs,
            train_set=train_set,
            device=device,
            wandb_active=True,
        )

        wandb.finish()

    wandb.agent(sweep_id, function=sweep_train_fn)


class SampleCallback(Callback):
    def __init__(self):
        if not os.path.exists(_PATH_TO_OUTPUT):
            os.makedirs(_PATH_TO_OUTPUT)

    def on_train_epoch_end(self, trainer, pl_module):
        sample = pl_module.sample()
        sample[0][0].save(_PATH_TO_OUTPUT / "sample.png")


class PruneCallback(Callback):
    def __init__(self, amount: float = 0.5):
        self.amount = amount

    def on_train_epoch_end(self, trainer, pl_module):
        for name, module in pl_module.unet.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                prune.l1_unstructured(module, name="weight", amount=self.amount)
                prune.remove(module, name="weight")
        print("Pruned model.")


class QuantizeCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        for name, module in pl_module.unet.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.quantization.quantize_dynamic(module, {torch.nn.Linear}, dtype=torch.qint8)
        print("Quantized model.")
