# TO DO LIST

- [x] Task 1: Fix model.py currently it is an empty shell/bad unet
- [x] Task 2: Fix data folder, and data.py (maybe remove processed) (Mads)
- [ ] Task 3: introduce logging to train loop (wandb) (Mads)
- [ ] Task 4: Train model, probably use HPC (maybe GC after)
- [x] Task 5: Downgrade to Pytorch 2.2.2
- [x] Task 6: Linting checks updated to github actions
- [x] Task 7: Add unit tests, and GA
- [ ] Task 8: dvc + Move data to GC
- [x] Task 9: add hydra



# Project Description: MLOps

## Project Description

### Overall Goal
The overall goal of this project is to create a diffusion model that can generate new image sprites of Pokémon. The image sprites will be kept at low resolution (32 × 32) to ensure fast training and inference.

### Frameworks and Tools
- **Deep Learning Framework**: PyTorch for all deep learning-related code.
- **Project Structure**: Cookiecutter template provided by Nicki, including:
  - Git for version control and collaboration.
- **Code Quality**:
  - Ruff for linting, adhering to PEP 8 standards.
- **Training and Metrics**:
  - Weights & Biases (WandB) for logging training progress and metrics.
- **Configuration Management**:
  - Hydra for simplifying hyperparameter management.
- **Environment Management**:
  - Docker for ensuring compatibility.

### Data
We will use the [Pokemon Images](https://www.kaggle.com) dataset from Kaggle. This dataset consists of:
- 819 transparent Pokémon images in `.png` format.
- Original size: 256 × 256.
- Training and inference size: Resized to 32 × 32.

### Models
For generating images, we will use a simple DDPM (Denoising Diffusion Probabilistic Model) with no pre-training. Specifically, we will use the model provided by Hugging Face: [DDPM Diffusers Pipeline](https://huggingface.co/docs/diffusers/api/pipelines/ddpm).

If the model requires additional data for effective training, we have the option to fetch a pretrained network as well.


# pokemon_ddpm

A ddpm using Pokémon dataset

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
