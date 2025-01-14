from pathlib import Path

import matplotlib.pyplot as plt
import torch


def visualize_datapoints(path: Path) -> None:
    """A simple function that visualizes random datapoints of the dataset."""

    # Load the tensor from the file
    data = torch.load(path)

    # Print the shape of the tensor
    print(data.shape)

    # Select a datapoint to plot (e.g., the first one)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, ax in enumerate(axes.flat):
        datapoint = data[torch.randint(0, len(data), (1,)).item()]
        ax.imshow(datapoint.permute(1, 2, 0))  # Assuming the tensor is in (C, H, W) format
        ax.axis("off")
        ax.set_title(f"Sample {i + 1}")
    fig.savefig("reports/figures/visualized_datapoints.png")
    plt.tight_layout()
    plt.show()


def model_samples(model, num_samples):
    """Generate samples from the model."""
    model.eval()
    samples = model.sample(num_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 6))

    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].permute(1, 2, 0))
        ax.axis("off")
        ax.set_title(f"Sample {i + 1}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = "data/processed/images.pt"
    visualize_datapoints(path)
