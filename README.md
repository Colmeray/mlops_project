# Project
The goal of this project is to build a machine learning pipeline that can classify houseplant species from images. We will use the Kaggle “house plant species” dataset, which contains roughly 14800 images across 47 species. The images are both close up and full size and vary in quality. The amount of pictures in each class also varies from around 60-550. This class imbalance is something we have to take into account when designing our pipeline.

The broader vision is that this classifier could be one component in a larger application that helps users identify plants in their home and then gives information about fx light requirements, weekly watering reminders etc. However, we focus specifically on the core classification task in this project: given an image predict the correct plant species.

Since we are working with images, we are going to start off with relying on a CNN based approach. We will primarily use pytorch to implement and train our model and we plan to rely on supporting libraries such as torchvision and timm (pytorch image models) to enable transfer learning with pretrained architectures. This could help us achieve good performance with limited training time by finetuning models that have already learned useful visual features. If applicable we will also try to implement more complex models such as vision transformers (ViT)

Beyond the model itself, we will build a clean and reproducible pipeline covering data loading, preprocessing, training, and evaluation. We will ofcourse also be using some of the frameworks covered in this course such as Weights & Biases (wandb) for experiment tracking and logging (e.g., metrics, hyperparameters, and model checkpoints), so results are easy to compare and reproduce. We will use Hydra to manage experiment configurations (model/training/data settings) and make it easy to run different setups via config files and command-line overrides without editing code. And ofcourse also other frameworks learned later in the course.

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
│   └── README.md/
│   └── reports.py/
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

# Instructions

### initialization and data
Start off by cloning the remote repo and getting the right dependecies with uv
```bash
git clone https://github.com/Colmeray/mlops_project.git
cd mlops_project
uv sync
```

To get the kaggle dataset downloaded locally in data folder start of with running
```bash
uv run python src/project/data.py ensure-dataset --data-dir data/raw
```

Preprocessing the data (just label mapping for now) run which stores the meta data in the data/preprocessed folder:
```bash
 uv run python src/project/data.py preprocess data/raw/house_plant_species data/preprocessed
```






Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
