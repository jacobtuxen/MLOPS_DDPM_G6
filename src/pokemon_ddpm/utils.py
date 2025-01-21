import os

from dotenv import load_dotenv

import wandb


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
    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config.get("project", "pokemon-ddpm"))

    # Function to execute training with WandB sweep
    def sweep_train_fn():
        wandb.init()
        config = wandb.config

        # Call the training function with the sweep configuration
        train_fn(
            model=model,
            batch_size=config["batch_size"],
            epochs=epochs,
            train_set=train_set,
            device=device,
            wandb_active=True,
        )

        wandb.finish()

    # Start the sweep
    wandb.agent(sweep_id, function=sweep_train_fn)
