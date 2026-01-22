import os
from pathlib import Path

from PIL import Image
from omegaconf import OmegaConf

from project.data import preprocess
from project.train import train_impl


def test_training_runs_one_batch(tmp_path: Path):
    os.environ["WANDB_MODE"] = "disabled"

    # Create tiny raw dataset
    raw_root = tmp_path / "raw"
    (raw_root / "ClassA").mkdir(parents=True)
    (raw_root / "ClassB").mkdir(parents=True)

    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(raw_root / "ClassA" / "1.jpg")
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(raw_root / "ClassB" / "1.jpg")

    # Preprocess into tmp folder
    processed_root = tmp_path / "preprocessed"
    preprocess(raw_root=raw_root, out_root=processed_root)

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "val_frac": 0.5,
            "epochs": 1,
            "batch_size": 2,
            "lr": 1e-4,
            "model": {"name": "simple", "freeze_features": True},
            "wandb": {"enable": False},
            "smoke_test": True,
            # override paths for training
            "raw_root": str(raw_root),
            "processed_root": str(processed_root),
        }
    )

    metrics = train_impl(cfg, max_batches=1)

    assert metrics["train_loss"] == metrics["train_loss"]  # not NaN
    assert metrics["train_loss"] > 0
    assert 0.0 <= metrics["train_acc"] <= 1.0
